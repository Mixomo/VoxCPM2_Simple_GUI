import os
import sys
# Set environment variables BEFORE importing torch to ensure Inductor respects them
os.environ["TORCHINDUCTOR_CUDAGRAPH_TREES"] = "0"

import json
import yaml
import datetime
import subprocess
import threading
import gradio as gr
import torch
import torch._inductor.config as inductor_config
inductor_config.triton.cudagraphs = False

from pathlib import Path
from typing import Optional

# Add src to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


# Default pretrained model path: prefer VoxCPM2 if it exists, fallback to VoxCPM1.5
_v2_path = project_root / "models" / "openbmb__VoxCPM2"
_v15_path = project_root / "models" / "openbmb__VoxCPM1.5"
default_pretrained_path = str(_v2_path if _v2_path.exists() else _v15_path)
 
# Setup persistent cache for torch.compile
compile_cache_dir = project_root / "models" / ".cache"
HAS_COMPILE_CACHE = compile_cache_dir.exists() and any(compile_cache_dir.iterdir())
_cache_kernel_count = 0
if HAS_COMPILE_CACHE:
    # Recursively count files in the cache directory to show progress/status
    for root, dirs, files in os.walk(compile_cache_dir):
        _cache_kernel_count += len(files)

compile_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache_dir)




print("----------------------------------------------------------------", file=sys.stderr)
if HAS_COMPILE_CACHE:
    print(f"INFO: Persistent compilation cache detected at {compile_cache_dir} ({_cache_kernel_count} kernels)", file=sys.stderr)
    print("INFO: Models will use this cache for hardware-accelerated inference.", file=sys.stderr)
else:
    print(f"NOTICE: Compilation cache will be built at {compile_cache_dir}", file=sys.stderr)
    print("NOTICE: The first generation will be slow due to one-time model optimization.", file=sys.stderr)
    print("NOTICE: Subsequent generations will be significantly faster.", file=sys.stderr)
print("----------------------------------------------------------------", file=sys.stderr)

# Enable TF32 for better performance on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig
import numpy as np
from faster_whisper import WhisperModel
import librosa
import soundfile as sf
import shutil
import glob
import random
from huggingface_hub import snapshot_download



# Global variables
current_model: Optional[VoxCPM] = None
asr_model: Optional[WhisperModel] = None
training_process: Optional[subprocess.Popen] = None
training_log = ""
training_finished_played = False # Track if completion chime was played
def play_done_chime():
    """Plays a notification sound on Linux using various possible backends."""
    try:
        # Try ALSA (usually present)
        if shutil.which("aplay"):
            # Use a standard system sound if it exists
            test_sounds = [
                "/usr/share/sounds/alsa/Front_Center.wav",
                "/usr/share/sounds/freedesktop/stereo/complete.oga",
                "/usr/share/sounds/gnome/default/alerts/glass.ogg"
            ]
            for s in test_sounds:
                if os.path.exists(s):
                    subprocess.Popen(["aplay", "-q", s], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
        
        # Try PulseAudio
        if shutil.which("paplay"):
            subprocess.Popen(["paplay", "--hint", "complete"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return

        # Final fallback: terminal bell
        print('\a', end='', flush=True)
    except:
        pass

def handle_sample_selection_ui(sample_name):
    """Wrapper for sample selection that also controls Style/Control visibility"""
    audio, text = load_sample(sample_name)
    control_visible = (sample_name == "None" or not sample_name)
    return audio, text, gr.update(visible=control_visible)

current_lora_config: Optional[LoRAConfig] = None
current_base_model_path: Optional[str] = None
current_is_lora: bool = False

# Model Mapping
VOXCPM_MODELS = {
    "VoxCPM-2.0": "openbmb/VoxCPM2",
    "VoxCPM-1.5": "openbmb/VoxCPM1.5",
    "VoxCPM-0.5B": "openbmb/VoxCPM-0.5B",
}


def get_timestamp_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def detect_sample_rate(pretrained_path: str) -> Optional[int]:
    """Read audio_vae_config.sample_rate from the model's config.json.

    This is the AudioVAE *encoder* input rate, which is the correct rate for
    resampling training data.  Returns None when detection fails.
    """
    config_file = os.path.join(pretrained_path, "config.json")
    if not os.path.isfile(config_file):
        return None
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg["audio_vae_config"]["sample_rate"])
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"Warning: failed to detect sample_rate from {config_file}: {e}", file=sys.stderr)
        return None


def get_or_load_asr_model():
    global asr_model
    if asr_model is None:
        print("Loading ASR model (Faster-Whisper Large-v3)...", file=sys.stderr)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Using float16 for CUDA to save VRAM and speed up, int8 for CPU
        compute_type = "float16" if device == "cuda" else "int8"
        
        # Local model path
        whisper_model_name = "Systran/faster-whisper-large-v3"
        dest_path = project_root / "models" / whisper_model_name.replace("/", "--")
        
        if not dest_path.exists():
            print(f"Downloading ASR model to {dest_path}...", file=sys.stderr)
            snapshot_download(repo_id=whisper_model_name, local_dir=str(dest_path), local_dir_use_symlinks=False)
            
        asr_model = WhisperModel(str(dest_path), device=device, compute_type=compute_type)
    return asr_model


def unload_asr_model():
    global asr_model
    if asr_model is not None:
        print("Unloading ASR model to free VRAM...", file=sys.stderr)
        del asr_model
        asr_model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def unload_model(*args):
    import torch
    global current_model, current_lora_config, current_base_model_path, current_is_lora
    
    print("Executing aggressive VRAM cleanup...", file=sys.stderr)
    unload_asr_model()
    current_is_lora = False
    
    if current_model is not None:
        print("Unloading VoxCPM model instance...", file=sys.stderr)
        try:
            # Move to CPU first to signal CUDA to free memory
            if hasattr(current_model, "tts_model") and current_model.tts_model is not None:
                current_model.tts_model.to("cpu")
        except Exception:
            pass
            
        del current_model
        current_model = None
        
    current_lora_config = None
    current_base_model_path = None
    
    # Force deep cleanup regardless of status box
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
            
    # Reset torch._dynamo to clear compiled graph cache
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
    except Exception:
        pass
        
    print("VRAM cleanup complete.", file=sys.stderr)
    return "VRAM and Models cleared successfully."


def get_sample_choices():
    samples_dir = project_root / "samples"
    if not samples_dir.exists():
        os.makedirs(samples_dir, exist_ok=True)
        return []
    
    audio_extections = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
    choices = []
    for f in os.listdir(samples_dir):
        if f.lower().endswith(audio_extections):
            name = os.path.splitext(f)[0]
            if name not in choices:
                choices.append(name)
    return ["None"] + sorted(choices)


def load_sample(sample_name):
    if not sample_name or sample_name == "None":
        return None, ""
    samples_dir = project_root / "samples"
    
    # Try different audio extensions
    audio_path = None
    for ext in [".wav", ".mp3", ".flac", ".m4a", ".ogg"]:
        p = samples_dir / f"{sample_name}{ext}"
        if p.exists():
            audio_path = p
            break
            
    if not audio_path:
        return None, ""

    txt_path = samples_dir / f"{sample_name}.txt"
    json_path = samples_dir / f"{sample_name}.json"
    
    text = ""
    # Try JSON first (handles both 'text' and 'Text' keys)
    if json_path.exists():
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
            try:
                with open(json_path, "r", encoding=encoding) as f:
                    data = json.load(f)
                    # Support multiple key variations common in JSON samples
                    text = data.get("Text") or data.get("text") or data.get("content") or data.get("transcription") or ""
                if text: break
            except: pass
    
    # Try TXT if JSON failed or was empty (plain text)
    if not text and txt_path.exists():
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            try:
                with open(txt_path, "r", encoding=encoding) as f:
                    text = f.read().strip()
                if text: break
            except: pass
            
    return str(audio_path.absolute()), text


def save_prep_sample(audio_path, sample_name, transcription):
    if not audio_path or not sample_name:
        return "Error: Missing audio or sample name."
    samples_dir = project_root / "samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    # Preserve original extension
    ext = os.path.splitext(audio_path)[1] or ".wav"
    dest_audio = samples_dir / f"{sample_name}{ext}"
    dest_json = samples_dir / f"{sample_name}.json"
    dest_txt = samples_dir / f"{sample_name}.txt"
    
    shutil.copy(audio_path, dest_audio)
    
    # Save as JSON
    with open(dest_json, "w", encoding="utf-8") as f:
        json.dump({"text": transcription}, f, ensure_ascii=False, indent=2)
        
    # Save as plaintext TXT (as requested)
    with open(dest_txt, "w", encoding="utf-8") as f:
        f.write(transcription)
        
    return f"Sample '{sample_name}' saved successfully (Audio + JSON + TXT)!"


def recognize_audio(audio_path):
    if not audio_path:
        return ""
    try:
        model = get_or_load_asr_model()
        segments, info = model.transcribe(audio_path, beam_size=5)
        text = "".join([s.text for s in segments]).strip()
        return text
    except Exception as e:
        print(f"ASR Error: {e}", file=sys.stderr)
        return ""


def launch_tensorboard(speaker_name):
    if not speaker_name:
        return "Please select or type an Output Directory Name first."
    
    log_dir = project_root / "lora" / speaker_name / "logs"
    if not log_dir.exists():
        return f"No logs found yet at {log_dir}. Start training first."
    
    import webbrowser
    port = 6006
    # Try to start tensorboard
    cmd = f"tensorboard --logdir \"{log_dir}\" --port {port}"
    # Use subprocess.Popen to run in background
    try:
        subprocess.Popen(cmd, shell=True)
        # Give it a second to start
        import time
        time.sleep(2)
        webbrowser.open(f"http://localhost:{port}")
        return f"TensorBoard launched for {speaker_name} at http://localhost:{port}"
    except Exception as e:
        return f"Error launching TensorBoard: {e}"


def scan_datasets():
    """Scan the datasets folder for .jsonl manifest files."""
    datasets_root = project_root / "datasets"
    if not datasets_root.exists():
        os.makedirs(datasets_root, exist_ok=True)
        return []
    
    manifests = []
    # Recursively find all .jsonl files
    for root, dirs, files in os.walk(datasets_root):
        for f in files:
            if f.endswith(".jsonl"):
                full_path = os.path.join(root, f)
                # Use absolute path for reliability in training scripts
                manifests.append(os.path.abspath(full_path))
    return sorted(manifests)


def get_existing_lora_projects():
    """Scan the lora/ directory for existing training project subfolders."""
    lora_dir = project_root / "lora"
    if not lora_dir.exists():
        os.makedirs(lora_dir, exist_ok=True)
        return []
    
    # Get direct subdirectories of lora/
    projects = [d for d in os.listdir(lora_dir) if os.path.isdir(os.path.join(lora_dir, d))]
    return sorted(projects)


def download_voxcpm_model(model_id_or_name):
    """Download model from HF if needed and return local path."""
    # Resolve name from mapping if exists, otherwise assume it's a repo ID
    repo_id = VOXCPM_MODELS.get(model_id_or_name, model_id_or_name)
    
    # Store models in project_root/models/repo_name
    safe_name = repo_id.replace("/", "--")
    dest_path = project_root / "models" / safe_name
    
    if not dest_path.exists():
        print(f"Downloading {repo_id} to {dest_path}...", file=sys.stderr)
        snapshot_download(repo_id=repo_id, local_dir=str(dest_path), local_dir_use_symlinks=False)
    
    return str(dest_path.absolute())


def prepare_voxcpm_dataset(source_folder, dataset_name, val_split=0.1, batch_size=16, progress=gr.Progress()):
    if not source_folder or not os.path.isdir(source_folder):
        return "Error: Please provide a valid source folder path."
    if not dataset_name or dataset_name.strip() == "":
        return "Error: Please provide a target dataset name."

    # Create target directory in datasets/
    datasets_root = project_root / "datasets"
    target_dir = datasets_root / dataset_name
    os.makedirs(target_dir, exist_ok=True)

    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]:
        audio_files.extend(glob.glob(os.path.join(source_folder, ext)))
        audio_files.extend(glob.glob(os.path.join(source_folder, ext.upper())))

    if not audio_files:
        return "Error: No audio files found in the source folder."

    # Sort files to ensure speaker continuity if files are sequentially named
    audio_files.sort()

    total = len(audio_files)
    all_data = []
    
    progress(0.1, desc="Initializing Whisper Large-v3...")
    model = get_or_load_asr_model() # Pre-load to avoid timeout in loop
    
    try:
        from faster_whisper import BatchedInferencePipeline
        pipeline = BatchedInferencePipeline(model=model, use_vad_model=True)
    except:
        pipeline = None

    # 1. Audio Processing & Transcription
    previous_audio = None
    for i, audio_path in enumerate(audio_files):
        filename = os.path.basename(audio_path)
        dest_audio = target_dir / (os.path.splitext(filename)[0] + ".wav")
        
        progress(0.1 + (0.8 * (i / total)), desc=f"Whisper Transcribing {i+1}/{total}: {filename}")
        
        try:
            # Load and normalize (VoxCPM needs mono)
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            max_val = np.max(np.abs(y))
            if max_val > 1e-6:
                y = y / max_val * 0.95
            
            # Save normalized mono wav
            sf.write(str(dest_audio), y, sr)
            duration = len(y) / sr
            
            # Transcribe
            if pipeline:
                segments, info = pipeline.transcribe(str(dest_audio), batch_size=batch_size, beam_size=5)
            else:
                segments, info = model.transcribe(str(dest_audio), beam_size=5)
                
            text = "".join([s.text for s in segments]).strip()
            if text:
                entry = {
                    "audio": str(dest_audio.absolute()).replace("\\", "/"),
                    "text": text,
                    "duration": round(duration, 3),
                    "dataset_id": 0
                }
                if previous_audio is not None:
                    entry["ref_audio"] = previous_audio
                
                all_data.append(entry)
                previous_audio = entry["audio"]
        except Exception as e:
            print(f"Error processing {filename}: {e}", file=sys.stderr)

    if not all_data:
        return "Error: Failed to process or transcribe any audio."

    # 2. Split into Train and Validation
    random.shuffle(all_data)
    num_val = int(len(all_data) * val_split)
    
    if num_val == 0 and val_split > 0:
        num_val = 1 # At least one if split is > 0
    
    val_data = all_data[:num_val]
    train_data = all_data[num_val:]

    # 3. Save Manifests
    train_manifest_path = target_dir / "train_manifest.jsonl"
    val_manifest_path = target_dir / "val_manifest.jsonl"
    
    with open(train_manifest_path, "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    if val_data:
        with open(val_manifest_path, "w", encoding="utf-8") as f:
            for entry in val_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    result_msg = f"Success! Dataset prepared.\n- Total: {len(all_data)}\n- Training: {len(train_data)}\n- Validation: {len(val_data)}\n\nPaths:\n- Train JSONL: {train_manifest_path}\n"
    if val_data:
        result_msg += f"- Valid JSONL: {val_manifest_path}\n"
        
    return result_msg + "\nYou can now use these paths in the Training tab."


def scan_lora_checkpoints(root_dir="lora", with_info=False):
    """
    Scans for LoRA checkpoints in the lora directory.

    Args:
        root_dir: Directory to scan for LoRA checkpoints
        with_info: If True, returns list of (path, base_model) tuples

    Returns:
        List of checkpoint paths, or list of (path, base_model) tuples if with_info=True
    """
    checkpoints = []
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)

    # Look for lora_weights.safetensors recursively
    for root, dirs, files in os.walk(root_dir):
        if "lora_weights.safetensors" in files:
            # Use the relative path from root_dir as the ID
            rel_path = os.path.relpath(root, root_dir)

            if with_info:
                # Try to read base_model from lora_config.json
                base_model = None
                lora_config_file = os.path.join(root, "lora_config.json")
                if os.path.exists(lora_config_file):
                    try:
                        with open(lora_config_file, "r", encoding="utf-8") as f:
                            lora_info = json.load(f)
                        base_model = lora_info.get("base_model", "Unknown")
                    except (json.JSONDecodeError, OSError):
                        pass
                checkpoints.append((rel_path, base_model))
            else:
                checkpoints.append(rel_path)

    # Also check for checkpoints in the default location if they exist
    default_ckpt = "checkpoints/finetune_lora"
    if os.path.exists(os.path.join(root_dir, default_ckpt)):
        # This might be covered by the walk, but good to be sure
        pass

    return sorted(checkpoints, reverse=True)


def refresh_loras():
    checkpoints_with_info = scan_lora_checkpoints(with_info=True)
    # checkpoints_with_info is a list of strings or (rel_path, base_model) tuples
    choices = ["None"]
    for ckpt in checkpoints_with_info:
        if isinstance(ckpt, tuple):
            choices.append(ckpt[0])
        else:
            choices.append(ckpt)
            
    print(f"Refreshed LoRA list: {len(checkpoints_with_info)} checkpoints found", file=sys.stderr)
    return gr.update(choices=choices, value="None")


def load_lora_config_from_checkpoint(lora_path):
    """Load LoRA config from lora_config.json if available."""
    lora_config_file = os.path.join(lora_path, "lora_config.json")
    if os.path.exists(lora_config_file):
        try:
            with open(lora_config_file, "r", encoding="utf-8") as f:
                lora_info = json.load(f)
            lora_cfg_dict = lora_info.get("lora_config", {})
            if lora_cfg_dict:
                return LoRAConfig(**lora_cfg_dict), lora_info.get("base_model")
        except Exception as e:
            print(f"Warning: Failed to load lora_config.json: {e}", file=sys.stderr)
    return None, None


def get_default_lora_config():
    """Return default LoRA config for hot-swapping support."""
    return LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        r=32,
        alpha=16,
        target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
        target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def load_model(pretrained_path, lora_path=None):
    global current_model, current_lora_config, current_base_model_path, current_is_lora
    print(f"Loading model from {pretrained_path}...", file=sys.stderr)

    lora_config = None
    lora_weights_path = None

    if lora_path:
        full_lora_path = os.path.join("lora", lora_path)
        if os.path.exists(full_lora_path):
            lora_weights_path = full_lora_path
            # Try to load LoRA config from lora_config.json
            lora_config, _ = load_lora_config_from_checkpoint(full_lora_path)
            if lora_config:
                print(f"Loaded LoRA config from {full_lora_path}/lora_config.json", file=sys.stderr)
            else:
                # Fallback to default config for old checkpoints
                lora_config = get_default_lora_config()
                print("Using default LoRA config (lora_config.json not found)", file=sys.stderr)

    # Always init with a default LoRA config to allow hot-swapping later
    if lora_config is None:
        lora_config = get_default_lora_config()

    current_lora_config = lora_config
    current_base_model_path = pretrained_path

    current_model = VoxCPM.from_pretrained(
        hf_model_id=pretrained_path,
        load_denoiser=False,
        optimize=True,
        lora_config=lora_config,
        lora_weights_path=lora_weights_path,
    )
    current_is_lora = bool(lora_path)
    return "Model loaded successfully!"


def run_inference(text, prompt_wav, prompt_text, lora_selection, cfg_scale, steps, seed, model_choice=None, control=None, split_by_paragraph=False, progress=gr.Progress(), **kwargs):
    global current_model, current_lora_config, current_base_model_path, current_is_lora
    
    is_lora_request = bool(lora_selection and lora_selection != "None")
    current_is_lora = is_lora_request
    
    # Check if we have an existing cache to inform the user
    cache_exists = any(compile_cache_dir.iterdir()) if compile_cache_dir.exists() else False
    
    if not cache_exists:
        progress(0, desc="[First Run] Initializing hardware acceleration (torch.compile)... Please be patient, this might take up to 5 minutes!")
        print("NOTICE: Building compilation cache for the first time. This will take up to 5 minutes, but happens only once.", file=sys.stderr)
    else:
        progress(0, desc="Hardware acceleration detected. Check the console...")
        print("INFO: Using existing persistent compilation cache for optimized inference.", file=sys.stderr)

    
    # Determine the target base model path
    try:
        base_model_path = download_voxcpm_model(model_choice) if model_choice else default_pretrained_path
    except Exception as e:
        return None, f"Error resolving model: {e}"

    # Determine target lora_config
    target_lora_config = None
    if lora_selection and lora_selection != "None":
        full_lora_path = os.path.join("lora", lora_selection)
        target_lora_config, saved_base_model = load_lora_config_from_checkpoint(full_lora_path)
        
        if saved_base_model and os.path.exists(saved_base_model):
            base_model_path = saved_base_model

    if target_lora_config is None:
        target_lora_config = get_default_lora_config()

    # Check if we need to reload the model entirely
    need_reload = False
    if current_model is None:
        need_reload = True
    elif current_base_model_path != base_model_path:
        need_reload = True
    elif current_lora_config is None:
        need_reload = True
    else:
        # Compare rank and alpha. If they don't match, hot-swapping won't work, need to reload.
        if target_lora_config.r != current_lora_config.r or target_lora_config.alpha != current_lora_config.alpha:
            need_reload = True
            print(f"Force reloading model: LoRA configs differ ({target_lora_config.r} vs {current_lora_config.r})", file=sys.stderr)

    if need_reload:
        try:
            # Explicitly force a clean state before loading to avoid "loading on top"
            print("Force clearing VRAM before model reload...", file=sys.stderr)
            unload_model()
            
            print(f"Loading base model: {base_model_path}", file=sys.stderr)
            # Use a local temporary variable to prevent polluting global if it fails
            temp_model = VoxCPM.from_pretrained(
                hf_model_id=base_model_path,
                load_denoiser=False,
                optimize=True,
                lora_config=target_lora_config,
                lora_weights_path=os.path.join("lora", lora_selection) if (lora_selection and lora_selection != "None") else None
            )
            
            current_model = temp_model
            current_base_model_path = base_model_path
            current_lora_config, _ = load_lora_config_from_checkpoint(os.path.join("lora", lora_selection)) if (lora_selection and lora_selection != "None") else (get_default_lora_config(), None)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Failed to load model from {base_model_path}: {str(e)}"
            print(error_msg, file=sys.stderr)
            # Ensure we are clean if load fails
            unload_model()
            return None, error_msg

    # Handle LoRA hot-swapping
    assert current_model is not None, "Model must be loaded before inference"
    if lora_selection and lora_selection != "None":
        full_lora_path = os.path.join("lora", lora_selection)
        print(f"Hot-loading LoRA: {full_lora_path}", file=sys.stderr)
        try:
            current_model.load_lora(full_lora_path)
            current_model.set_lora_enabled(True)
        except Exception as e:
            print(f"Error loading LoRA: {e}", file=sys.stderr)
            return None, f"Error loading LoRA: {e}"
    else:
        print("Disabling LoRA", file=sys.stderr)
        current_model.set_lora_enabled(False)

    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Handle prompt parameters: must be both None or both have values
    final_prompt_wav = None
    final_prompt_text = None

    if prompt_wav and prompt_wav.strip():
        # Reference audio provided
        final_prompt_wav = prompt_wav

        # If reference text not provided, try automatic recognition
        if not prompt_text or not prompt_text.strip():
            print("Reference audio provided but text missing, auto-recognizing...", file=sys.stderr)
            try:
                final_prompt_text = recognize_audio(prompt_wav)
                unload_asr_model() # Free VRAM for inference
                if final_prompt_text:
                    print(f"ASR result: {final_prompt_text}", file=sys.stderr)
                else:
                    return None, "Error: Could not recognize reference audio content, please fill text manually."
            except Exception as e:
                return None, f"Error: ASR failed - {str(e)}"
        else:
            final_prompt_text = prompt_text.strip()
    
    # Prepare paragraphs based on split logic
    if split_by_paragraph:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    else:
        paragraphs = [text.strip()]
    
    if not paragraphs or (len(paragraphs) == 1 and not paragraphs[0]):
        return None, "Error: No text provided for generation."

    num_clips = len(paragraphs)
    audio_segments = []
    sample_rate = 16000 # default fallback
    
    # Prepare paragraphs based on split logic
    if split_by_paragraph:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    else:
        paragraphs = [text.strip()]
    
    if not paragraphs or (len(paragraphs) == 1 and not paragraphs[0]):
        return None, "Error: No text provided for generation."

    num_clips = len(paragraphs)
    audio_segments = []
    sample_rate = 16000 # default fallback
    
    # Standard generation with optional control instruction
    final_control = (control or kwargs.get("control") or "").strip()

    try:
        for i, para in enumerate(paragraphs):
            current_text = f"({final_control}){para}" if final_control else para
            progress((i / num_clips), desc=f"Generating clip {i+1}/{num_clips} ({len(para)} chars)...")
            
            audio_np = current_model.generate(
                text=current_text,
                prompt_wav_path=final_prompt_wav,
                prompt_text=final_prompt_text,
                cfg_value=cfg_scale,
                inference_timesteps=steps,
            )
            
            audio_segments.append(audio_np)
            sample_rate = current_model.tts_model.sample_rate
            
            # Add 0.5s silence between paragraphs if there are multiple
            if num_clips > 1 and i < num_clips - 1:
                silence = np.zeros(int(sample_rate * 0.5), dtype=audio_np.dtype)
                audio_segments.append(silence)
                
        # Concatenate all segments
        play_done_chime()
        final_audio = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
        audio_int16 = (final_audio * 32767).astype(np.int16)
        
        return (sample_rate, audio_int16), f"Generation Success ({num_clips} clips)"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

def process_audio_array(audio_data):
    """Converts audio to mono and normalizes volume to [-1, 1]"""
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
        
    return audio_data

def generate_dialogue(lora_selection, cfg_scale, steps, seed, model_choice, split_para, row_count, silence_duration, *args, progress=gr.Progress()):
    num_max = 20
    samples = args[:num_max]
    controls = args[num_max:2*num_max]
    texts = args[2*num_max:3*num_max]
    
    segments = []
    for i in range(int(row_count)):
        s = samples[i]
        c = controls[i]
        t = texts[i]
        if s and t:
            segments.append((s, c, t))
            
    if not segments:
        return None, "Please add at least one speaker and text."
        
    all_audio_segments = []
    final_sr = 16000 # default
    
    for i, (sample_name, control, text) in enumerate(segments):
        progress((i / len(segments)), desc=f"Processing segment {i+1}/{len(segments)} ({sample_name})...")
        
        # Load sample
        ref_audio, ref_text = load_sample(sample_name)
        if not ref_audio and sample_name != "None":
            print(f"Sample {sample_name} not found, skipping segment {i+1}")
            continue
            
        result, status = run_inference(
            text=text, 
            prompt_wav=ref_audio, 
            prompt_text=ref_text, 
            lora_selection=lora_selection, 
            cfg_scale=cfg_scale, 
            steps=steps, 
            seed=seed, 
            model_choice=model_choice, 
            control=control, 
            progress=progress, 
            split_by_paragraph=split_para
        )
        
        if result is not None:
            sr, audio_int16 = result
            final_sr = sr
            audio_data = audio_int16.astype(np.float32) / 32767.0
            audio_data = process_audio_array(audio_data)
            all_audio_segments.append(audio_data)
            
            if silence_duration > 0:
                silence = np.zeros(int(sr * silence_duration), dtype=np.float32)
                all_audio_segments.append(silence)
        else:
            return None, f"Error in segment {i+1} ({sample_name}): {status}"
            
    if all_audio_segments:
        if silence_duration > 0 and len(all_audio_segments) > 1:
            combined = np.concatenate(all_audio_segments[:-1])
        else:
            combined = np.concatenate(all_audio_segments)
            
        combined = process_audio_array(combined)
        audio_int16 = (combined * 32767).astype(np.int16)
        play_done_chime()
        return (final_sr, audio_int16), f"Dialogue generated successfully with {len(segments)} segments!"
    
    return None, "No audio generated."

def add_dialogue_row_at(index, count, *args):
    num = 20
    samples = list(args[:num])
    controls = list(args[num:2*num])
    texts = list(args[2*num:3*num])
    
    if count < num:
        samples.insert(index + 1, samples[index])
        controls.insert(index + 1, "")
        texts.insert(index + 1, "")
        samples.pop()
        controls.pop()
        texts.pop()
        count += 1
    
    updates = [count]
    updates.extend([gr.update(value=samples[i], visible=(i < count)) for i in range(num)])
    updates.extend([gr.update(value=controls[i], visible=(i < count and (samples[i] == "None" or not samples[i]))) for i in range(num)])
    updates.extend([gr.update(value=texts[i], visible=(i < count)) for i in range(num)])
    updates.extend([gr.update(visible=(i < count)) for i in range(num)])
    return updates

def rem_dialogue_row_at(index, count, *args):
    num = 20
    samples = list(args[:num])
    controls = list(args[num:2*num])
    texts = list(args[2*num:3*num])
    
    if count > 1:
        samples.pop(index)
        controls.pop(index)
        texts.pop(index)
        samples.append(None)
        controls.append("")
        texts.append("")
        count -= 1
        
    updates = [count]
    updates.extend([gr.update(value=samples[i], visible=(i < count)) for i in range(num)])
    updates.extend([gr.update(value=controls[i], visible=(i < count and (samples[i] == "None" or not samples[i]))) for i in range(num)])
    updates.extend([gr.update(value=texts[i], visible=(i < count)) for i in range(num)])
    updates.extend([gr.update(visible=(i < count)) for i in range(num)])
    return updates

def clone_dialogue_row_at(index, count, *args):
    num = 20
    samples = list(args[:num])
    controls = list(args[num:2*num])
    texts = list(args[2*num:3*num])
    
    if count < num:
        samples.insert(index + 1, samples[index])
        controls.insert(index + 1, controls[index])
        texts.insert(index + 1, texts[index])
        samples.pop()
        controls.pop()
        texts.pop()
        count += 1
        
    updates = [count]
    updates.extend([gr.update(value=samples[i], visible=(i < count)) for i in range(num)])
    updates.extend([gr.update(value=controls[i], visible=(i < count and (samples[i] == "None" or not samples[i]))) for i in range(num)])
    updates.extend([gr.update(value=texts[i], visible=(i < count)) for i in range(num)])
    updates.extend([gr.update(visible=(i < count)) for i in range(num)])
    return updates

def start_training(
    model_choice,
    train_manifest,
    val_manifest,
    learning_rate,
    num_iters,
    batch_size,
    lora_rank,
    lora_alpha,
    save_interval,
    output_name="",
    # Advanced options
    grad_accum_steps=1,
    num_workers=2,
    log_interval=10,
    valid_interval=1000,
    weight_decay=0.01,
    warmup_steps=100,
    max_steps=None,
    sample_rate=44100,
    max_grad_norm=1.0,
    # LoRA advanced
    enable_lm=True,
    enable_dit=True,
    enable_proj=False,
    dropout=0.0,
    tensorboard_path="",
    # Distribution options
    hf_model_id="",
    distribute=False,
):
    global training_process, training_log

    if training_process is not None and training_process.poll() is None:
        return "Training is already running!"

    # Unload models to free VRAM for training
    unload_model()

    # Download or resolve model path
    try:
        pretrained_path = download_voxcpm_model(model_choice)
    except Exception as e:
        return f"Error resolving model: {e}"

    if output_name and output_name.strip():
        timestamp = output_name.strip()
    else:
        timestamp = get_timestamp_str()

    save_dir_obj = project_root / "lora" / timestamp
    save_dir = str(save_dir_obj)
    checkpoints_dir = str(save_dir_obj / "checkpoints")
    logs_dir = str(save_dir_obj / "logs")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Auto-detect sample_rate from model config.json to prevent mismatch
    detected_sr = detect_sample_rate(pretrained_path)
    if detected_sr is not None:
        if int(sample_rate) != detected_sr:
            training_log += (
                f"[Auto-fix] sample_rate changed from {int(sample_rate)} to {detected_sr} "
                f"(read from {pretrained_path}/config.json audio_vae_config.sample_rate)\n"
            )
        sample_rate = detected_sr

    # Create config dictionary
    resolved_max_steps = int(max_steps) if max_steps not in (None, "", 0) else int(num_iters)

    # Auto-detect out_sample_rate from model config
    out_sample_rate = 0
    config_file = os.path.join(pretrained_path, "config.json")
    if os.path.isfile(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            out_sr = cfg.get("audio_vae_config", {}).get("out_sample_rate")
            if out_sr:
                out_sample_rate = int(out_sr)
        except Exception:
            pass

    config = {
        "pretrained_path": pretrained_path,
        "train_manifest": os.path.abspath(train_manifest) if train_manifest else None,
        "val_manifest": os.path.abspath(val_manifest) if val_manifest else None,
        "sample_rate": int(sample_rate),
        "out_sample_rate": out_sample_rate,
        "batch_size": int(batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "num_workers": int(num_workers),
        "num_iters": int(num_iters),
        "log_interval": int(log_interval),
        "valid_interval": int(valid_interval),
        "save_interval": int(save_interval),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "warmup_steps": int(warmup_steps),
        "max_steps": resolved_max_steps,
        "max_grad_norm": float(max_grad_norm),
        "save_path": checkpoints_dir,
        "tensorboard": tensorboard_path if tensorboard_path else logs_dir,
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "lora": {
            "enable_lm": bool(enable_lm),
            "enable_dit": bool(enable_dit),
            "enable_proj": bool(enable_proj),
            "r": int(lora_rank),
            "alpha": int(lora_alpha),
            "dropout": float(dropout),
            "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"],
        },
    }

    # Add distribution options if provided
    if hf_model_id and hf_model_id.strip():
        config["hf_model_id"] = hf_model_id.strip()
    if distribute:
        config["distribute"] = True

    config_path = os.path.join(save_dir, "train_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    cmd = [sys.executable, "scripts/train_voxcpm_finetune.py", "--config_path", config_path]

    training_log = f"Starting training process for model: {model_choice}...\nConfig saved to {config_path}\nOutput dir: {save_dir}\n"

    def run_process():
        global training_process, training_log
        training_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        assert training_process.stdout is not None
        for line in training_process.stdout:
            training_log += line
            print(line, end="", flush=True) # Log to CMD console
            if len(training_log) > 100000:
                training_log = training_log[-100000:]

        training_process.wait()
        training_log += f"\nTraining finished with code {training_process.returncode}"

    threading.Thread(target=run_process, daemon=True).start()

    return f"Training started! Project: {timestamp}"


def get_training_log():
    return training_log


def stop_training():
    global training_log
    if training_process is not None and training_process.poll() is None:
        training_process.terminate()
        training_log += "\nTraining terminated by user."
        return "Training stopped."
    return "No training running."


def check_training_status():
    global training_process, training_log, training_finished_played
    log = training_log
    chime = None
    
    # If process was running and now it's finished
    if training_process is not None and training_process.poll() is not None:
        if not training_finished_played:
            training_finished_played = True


    elif training_process is not None and training_process.poll() is None:
        # Currently running
        training_finished_played = False
        
    return log, None


# --- GUI Layout ---
with gr.Blocks(title="VoxCPM - Simple GUI | Inference + LoRa Training") as app:
    # --- Initial State for Samples ---
    sample_choices = get_sample_choices()
    default_sample_name = sample_choices[0] if sample_choices else None
    default_audio, default_text = load_sample(default_sample_name)
    
    # --- Title Section ---
    with gr.Row(elem_classes="title-section"):
        with gr.Column(scale=4):
            gr.Markdown("""
            # VoxCPM - Simple GUI | Inference + LoRa Training
            """)
        with gr.Column(scale=1):
            gr.Markdown("""
            [📖 Documentation](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html)
            """)

    with gr.Tabs(elem_classes="tabs") as tabs:
        # === Dataset Preparation Tab ===
        with gr.Tab("📂 Dataset Preparation", id="tab_dataset") as tab_dataset:
            gr.Markdown("### 🛠️ Dataset Creation & Auto-Transcription")
            with gr.Row():
                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown(
                        "### 📂 Dataset Creation for Training\n"
                        "This workflow prepares your raw audio files for the LoRA fine-tuning process.\n\n"
                        "**This will:**\n"
                        "1. **Copy** audios to the `datasets/` folder.\n"
                        "2. **Normalize** volume to -0.95 dB.\n"
                        "3. **Convert** to Mono if needed.\n"
                        "4. **Auto-Transcribe** using Faster-Whisper Large-v3.\n"
                        "5. **Generate** `train_manifest.jsonl` and `val_manifest.jsonl` manifests."
                    )
                    
                    gr.Markdown("#### 📁 Folder Selection")
                    with gr.Row():
                        src_folder = gr.Textbox(
                            label="Source Audio Folder",
                            placeholder="C:\\Users\\Voice\\AudioFolder...",
                            scale=4
                        )
                        explorer_btn = gr.Button("📂 Browse", scale=1)
                    
                    def open_folder_explorer():
                        try:
                            import tkinter as tk
                            from tkinter import filedialog
                            root = tk.Tk()
                            root.attributes('-topmost', 1)
                            root.withdraw()
                            path = filedialog.askdirectory(title="Select Source Audio Folder")
                            root.destroy()
                            return path if path else gr.update()
                        except:
                            return gr.update()
                            
                    explorer_btn.click(fn=open_folder_explorer, inputs=[], outputs=[src_folder])

                    dataset_name_input = gr.Textbox(
                        label="Target Dataset Name",
                        value="my_new_dataset",
                        info="This will create a folder in 'datasets/' with your audios and manifest.jsonl."
                    )
                    
                    val_split_slider = gr.Slider(
                        label="Validation Split Ratio",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.1,
                        step=0.05,
                        info="Ratio of dataset to use for validation (e.g. 0.1 = 10%)."
                    )
                    
                    batch_size_slider = gr.Slider(
                        label="Whisper Batch Size",
                        minimum=1,
                        maximum=64,
                        value=16,
                        step=1,
                        info="Batch size for Faster-Whisper batched inference (if supported)."
                    )
                    
                    prep_btn = gr.Button("Process & Transcribe", variant="primary", elem_classes="button-primary")
                    
                with gr.Column(scale=3, elem_classes="form-section"):
                    gr.Markdown("#### 📊 Processing Status")
                    prep_status_out = gr.TextArea(
                        label="",
                        placeholder="Waiting for start (Whisper Large-v3)...",
                        lines=12,
                        interactive=False,
                        elem_classes="input-field"
                    )

            prep_btn.click(
                fn=prepare_voxcpm_dataset,
                inputs=[src_folder, dataset_name_input, val_split_slider, batch_size_slider],
                outputs=[prep_status_out]
            )

        # === Training Tab ===
        with gr.Tab("🚀 Training") as tab_train:
            gr.Markdown("### 🎯 Model Training Configuration")

            with gr.Row():
                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown("#### 📁 Model & Dataset Selection")
                    
                    train_model_select = gr.Dropdown(
                        label="VoxCPM Foundation Model",
                        choices=list(VOXCPM_MODELS.keys()),
                        value="VoxCPM-2.0",
                        info="System will download the model automatically from HuggingFace if not present."
                    )

                    with gr.Row():
                        train_manifest = gr.Dropdown(
                            label="Train Manifest (jsonl)",
                            choices=scan_datasets(),
                            value=scan_datasets()[0] if scan_datasets() else None,
                            allow_custom_value=True,
                            scale=8,
                            info="Select a manifest from 'datasets/' folder."
                        )
                        refresh_train_btn = gr.Button("🔄", scale=1, min_width=50)

                    with gr.Row():
                        val_manifest = gr.Dropdown(
                            label="Validation Manifest (Optional)",
                            choices=scan_datasets(),
                            value=None,
                            allow_custom_value=True,
                            info="Optional validation manifest. Leave empty if not used.",
                            scale=8
                        )
                        refresh_val_btn = gr.Button("🔄", scale=1, min_width=50)

                    with gr.Row():
                        output_name = gr.Dropdown(
                            label="Output Directory Name",
                            choices=get_existing_lora_projects(),
                            value="",
                            allow_custom_value=True,
                            scale=8,
                            info="Training results will be saved in lora/[name]. Selecting an existing folder will resume training."
                        )
                        refresh_out_btn = gr.Button("🔄", scale=1, min_width=50)
                    
                    # Refresh logic
                    def refresh_manifests():
                        manifests = scan_datasets()
                        return gr.update(choices=manifests), gr.update(choices=manifests)

                    def refresh_projects():
                        return gr.update(choices=get_existing_lora_projects())

                    refresh_train_btn.click(refresh_manifests, outputs=[train_manifest, val_manifest])
                    refresh_val_btn.click(refresh_manifests, outputs=[train_manifest, val_manifest])
                    refresh_out_btn.click(refresh_projects, outputs=[output_name])

                    gr.Markdown("#### ⚙️ Core Hyperparameters")

                    with gr.Row():
                        lr = gr.Dropdown(
                            label="Learning Rate",
                            choices=["1e-3", "5e-4", "1e-4", "5e-5", "1e-5"],
                            value="1e-4",
                            allow_custom_value=True,
                            elem_classes="input-field",
                            info="1e-4 is typically best for LoRA."
                        )
                        num_iters = gr.Number(
                            label="Max Steps / Iterations",
                            value=2000,
                            precision=0,
                            elem_classes="input-field",
                            info="Total steps. 500-2000 is usually enough for a single voice."
                        )
                        batch_size = gr.Dropdown(
                            label="Batch Size",
                            choices=["1", "2", "4", "8", "12", "16"],
                            value="1",
                            allow_custom_value=True,
                            elem_classes="input-field",
                            info="Lower value saves VRAM. Use 1-4 for 24GB GPUs."
                        )

                    with gr.Row():
                        lora_rank = gr.Dropdown(
                            label="LoRA Rank (r)",
                            choices=["8", "16", "32", "64", "128"],
                            value="32",
                            allow_custom_value=True,
                            elem_classes="input-field",
                            info="Model capacity. 32 for speakers, 64 for styles."
                        )
                        lora_alpha = gr.Dropdown(
                            label="LoRA Alpha",
                            choices=["8", "16", "32", "64", "128"],
                            value="16",
                            allow_custom_value=True,
                            elem_classes="input-field",
                            info="Scale of LoRA layers. Usually set to Rank or half-Rank."
                        )
                        save_interval = gr.Number(
                            label="Save Interval (Steps)",
                            value=500,
                            precision=0,
                            elem_classes="input-field",
                            info="Saves a checkpoint every X steps."
                        )

                    with gr.Row():
                        start_btn = gr.Button("▶️ Start Training", variant="primary", elem_classes="button-primary")
                        stop_btn = gr.Button("⏹️ Stop Training", variant="stop", elem_classes="button-stop")
                        tb_btn = gr.Button("📊 TensorBoard", variant="secondary")

                    with gr.Accordion("🔧 Advanced Settings", open=False, elem_classes="accordion"):
                        with gr.Row():
                            sample_rate = gr.Dropdown(
                                label="Sample Rate (Hz)",
                                choices=["16000", "44100", "48000"],
                                value="16000",
                                allow_custom_value=True,
                                info="Encoder input rate: 16k for V2, 44.1k for V1.5."
                            )
                            grad_accum_steps = gr.Number(
                                label="Grad Accum Steps",
                                value=1,
                                precision=0,
                                info="Simulate larger batch size by accumulating gradients."
                            )
                        with gr.Row():
                            num_workers = gr.Number(label="Data Loader Workers", value=2, precision=0)
                            log_interval = gr.Number(label="Logging Interval", value=10, precision=0)
                        with gr.Row():
                            valid_interval = gr.Number(label="Validation Interval", value=500, precision=0)
                            weight_decay = gr.Number(label="Weight Decay", value=0.01)
                            warmup_steps = gr.Number(label="Warmup Steps", value=100, precision=0)
                        with gr.Row():
                            max_steps = gr.Number(label="Max Steps (Overrides iterations if > 0)", value=0, precision=0)
                            max_grad_norm = gr.Number(label="Max Grad Norm", value=1.0)
                        with gr.Row():
                            tensorboard_path = gr.Textbox(label="Custom Tensorboard Path", value="", placeholder="Default: [save_dir]/logs")
                            enable_lm = gr.Checkbox(label="Enable LoRA on LM", value=True)
                            enable_dit = gr.Checkbox(label="Enable LoRA on DiT", value=True)
                        with gr.Row():
                            enable_proj = gr.Checkbox(label="Enable Projection LoRA", value=False)
                            dropout = gr.Number(label="LoRA Dropout", value=0.0)

                        gr.Markdown("#### Distribution / Hub Settings")
                        with gr.Row():
                            hf_model_id = gr.Textbox(
                                label="HuggingFace Repo ID", value="", placeholder="e.g. openbmb/VoxCPM2"
                            )
                            distribute = gr.Checkbox(label="Distribute Mode", value=False)

                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown("#### 📟 Training Console Logs")
                    logs_out = gr.TextArea(
                        label="",
                        lines=22,
                        max_lines=35,
                        interactive=False,
                        elem_classes="input-field",
                        show_label=False,
                    )

            def on_pretrained_path_change(choice):
                path = download_voxcpm_model(choice)
                sr = detect_sample_rate(path)
                if sr is not None:
                    return gr.update(value=sr)
                return gr.update()

            train_model_select.change(
                on_pretrained_path_change,
                inputs=[train_model_select],
                outputs=[sample_rate],
            )

            start_btn.click(
                start_training,
                inputs=[
                    train_model_select,
                    train_manifest,
                    val_manifest,
                    lr,
                    num_iters,
                    batch_size,
                    lora_rank,
                    lora_alpha,
                    save_interval,
                    output_name,
                    # advanced
                    grad_accum_steps,
                    num_workers,
                    log_interval,
                    valid_interval,
                    weight_decay,
                    warmup_steps,
                    max_steps,
                    sample_rate,
                    max_grad_norm,
                    enable_lm,
                    enable_dit,
                    enable_proj,
                    dropout,
                    tensorboard_path,
                    # distribution
                    hf_model_id,
                    distribute,
                ],
                outputs=[logs_out],
            )
            stop_btn.click(stop_training, outputs=[logs_out])
            tb_btn.click(launch_tensorboard, inputs=[output_name], outputs=[logs_out])
            timer = gr.Timer(1)
            timer.tick(check_training_status, outputs=[logs_out])

        # === Voice Clone Tab ===
        with gr.Tab("🔊 Voice Clone") as tab_infer:
            gr.Markdown("### 🎙️ Unified Voice Synthesis & Dialogue Builder")
            gr.Markdown("> **Optimization Note:** Hardware acceleration (torch.compile) is active. The first generation after loading a new model or LoRA rank may take **up to 5 minutes** to optimize. Subsequent runs will be near-instant.")
            
            # --- GLOBAL SETTINGS ---
            with gr.Accordion("⚙️ Global Generation Settings", open=True, elem_classes="accordion"):
                with gr.Row():
                    infer_base_model = gr.Dropdown(
                        label="Core Model", 
                        choices=list(VOXCPM_MODELS.keys()), 
                        value="VoxCPM-2.0",
                        info="Select the base foundation model.",
                        scale=2
                    )
                    infer_lora = gr.Dropdown(
                        label="Voice Clone (LoRA)",
                        choices=["None"] + [ckpt[0] for ckpt in scan_lora_checkpoints(with_info=True)],
                        value="None",
                        info="Select 'None' for standard VoxCPM voice.",
                        scale=2
                    )
                    refresh_infer_lora_btn = gr.Button("🔄 Refresh Available Voices", variant="secondary", scale=1)
                
                with gr.Row():
                    infer_cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
                    infer_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, value=10, step=1)
                    infer_seed = gr.Number(label="Seed (-1 for Random)", value=-1, precision=0)
                
                with gr.Row():
                    with gr.Column():
                        split_para_check = gr.Checkbox(label="Split by Paragraphs (Recommended for long texts)", value=False)
                        gr.Markdown("ℹ️ *To apply splits, you must press **Enter** after each sentence or point where you want a cut; each line break will generate an independent audio clip that will be automatically merged.*")
                    with gr.Column():
                        dialogue_silence_slider = gr.Slider(0, 5, value=0.5, step=0.1, label="Silence between speakers (s)")
                        if HAS_COMPILE_CACHE:
                            gr.Markdown(f"✅ **Cache Found:** ({_cache_kernel_count} kernels)")
                        else:
                            gr.Markdown(f"⚠️ **No Cache:** First PyTorch run ~5 min.")
            
            # --- SUB-TABS ---
            with gr.Tabs():
                # --- SINGLE INFERENCE ---
                with gr.Tab("Single Inference"):
                    with gr.Row():
                        # --- Left Column: Reference & Voice Selection ---
                        with gr.Column(scale=1, elem_classes="form-section"):
                            gr.Markdown("#### 🎙️ Voice & Reference")
                            with gr.Row():
                                infer_sample_select = gr.Dropdown(
                                    choices=sample_choices,
                                    value=default_sample_name,
                                    label="Quick Sample Select",
                                    info="Load a reference from your 'samples' library.",
                                    scale=4
                                )
                                refresh_infer_sample_btn = gr.Button("🔄", scale=1, min_width=50)
                            
                            infer_ref_audio = gr.Audio(label="Reference Audio", type="filepath", value=default_audio)
                            infer_ref_text = gr.Textbox(
                                label="Reference Text (Transcript)", 
                                placeholder="Automatic transcription if left empty...", 
                                lines=2,
                                value=default_text
                            )

                        # --- Right Column: Content & Generation Settings ---
                        with gr.Column(scale=1, elem_classes="form-section"):
                            gr.Markdown("#### ✍️ Target Speech")
                            infer_text = gr.Textbox(
                                label="Target Text", 
                                placeholder="Enter the text you want the AI to speak...", 
                                lines=4,
                                value="Hello! I can speak with any voice you provide as a reference."
                            )
                            
                            infer_control = gr.Textbox(
                                label="Style / Control (Optional) Only works without a reference sample / text loaded.", 
                                placeholder="e.g. A happy energetic tone / Whispering softly...", 
                                lines=2,
                                visible=(default_sample_name == "None")
                            )
                            
                            infer_clips_count = gr.Markdown(
                                value="*1 clip detected*",
                                visible=False,
                                elem_classes="clips-count-mini"
                            )

                            infer_gen_btn = gr.Button("⚡ Generate Speech", variant="primary", size="lg", elem_classes="button-primary")

                # --- DIALOGUE BUILDER ---
                with gr.Tab("Dialogue Builder"):
                    gr.Markdown("#### 💬 Multi-Speaker Dialogue")
                    gr.Markdown("Select different speakers for each turn. Use the buttons on the right to manage segments.")
                    
                    dialogue_row_count = gr.State(2)
                    MAX_DIALOGUE_SEGMENTS = 20
                    dialogue_samples = []
                    dialogue_controls = []
                    dialogue_texts = []
                    dialogue_rows = []
                    dialogue_add_btns = []
                    dialogue_rem_btns = []
                    dialogue_clone_btns = []
                    
                    for i in range(MAX_DIALOGUE_SEGMENTS):
                        with gr.Row(visible=(i < 2)) as row:
                            with gr.Column(scale=3):
                                s = gr.Dropdown(choices=sample_choices, label=f"Speaker {i+1}", value=default_sample_name if i < 2 else "None")
                                c = gr.Textbox(
                                    placeholder=f"Style {i+1}...", 
                                    label=f"Control {i+1} (Only works without sample)", 
                                    lines=1,
                                    visible=(default_sample_name == "None" if i < 2 else True)
                                )
                            t = gr.Textbox(placeholder=f"Enter text for speaker {i+1}...", label=f"Text {i+1}", scale=7, lines=5)
                            with gr.Row(scale=1):
                                add_btn = gr.Button("➕", variant="secondary", size="sm", elem_classes=["green-btn"])
                                clone_btn = gr.Button("📋", variant="secondary", size="sm")
                                rem_btn = gr.Button("🗑️", variant="stop", size="sm", elem_classes=["red-btn"])
                            
                        dialogue_rows.append(row)
                        dialogue_samples.append(s)
                        dialogue_controls.append(c)
                        dialogue_texts.append(t)
                        dialogue_add_btns.append(add_btn)
                        dialogue_rem_btns.append(rem_btn)
                        dialogue_clone_btns.append(clone_btn)

                    # Bind events after all components are created
                    for i in range(MAX_DIALOGUE_SEGMENTS):
                        dialogue_add_btns[i].click(
                            fn=add_dialogue_row_at,
                            inputs=[gr.State(i), dialogue_row_count] + dialogue_samples + dialogue_controls + dialogue_texts,
                            outputs=[dialogue_row_count] + dialogue_samples + dialogue_controls + dialogue_texts + dialogue_rows
                        )
                        dialogue_rem_btns[i].click(
                            fn=rem_dialogue_row_at,
                            inputs=[gr.State(i), dialogue_row_count] + dialogue_samples + dialogue_controls + dialogue_texts,
                            outputs=[dialogue_row_count] + dialogue_samples + dialogue_controls + dialogue_texts + dialogue_rows
                        )
                        dialogue_clone_btns[i].click(
                            fn=clone_dialogue_row_at,
                            inputs=[gr.State(i), dialogue_row_count] + dialogue_samples + dialogue_controls + dialogue_texts,
                            outputs=[dialogue_row_count] + dialogue_samples + dialogue_controls + dialogue_texts + dialogue_rows
                        )
                        
                        # Add visibility handler for dialogue builder rows
                        def make_row_handler(idx):
                            def handler(sample_name):
                                visible = (sample_name == "None")
                                return gr.update(visible=visible)
                            return handler
                            
                        dialogue_samples[i].change(
                            fn=make_row_handler(i),
                            inputs=[dialogue_samples[i]],
                            outputs=[dialogue_controls[i]]
                        )
                        
                    dialogue_gen_btn = gr.Button("⚡ Generate Dialogue", variant="primary", size="lg", elem_classes="button-primary")

            with gr.Row():
                infer_audio_out = gr.Audio(label="Generated Audio")
                infer_status_out = gr.Textbox(label="System Status", interactive=False)
            
            # --- Unified Event Handlers ---
            def smart_asr_unified(audio, current_text):
                if current_text and current_text.strip():
                    return current_text
                return recognize_audio(audio)

            infer_sample_select.change(
                fn=handle_sample_selection_ui, 
                inputs=[infer_sample_select], 
                outputs=[infer_ref_audio, infer_ref_text, infer_control]
            )
            refresh_infer_sample_btn.click(lambda: gr.update(choices=get_sample_choices()), outputs=[infer_sample_select])
            refresh_infer_lora_btn.click(refresh_loras, outputs=[infer_lora])
            infer_ref_audio.change(fn=smart_asr_unified, inputs=[infer_ref_audio, infer_ref_text], outputs=[infer_ref_text])

            def update_clips_count(text, enabled):
                if not enabled:
                    return gr.update(visible=False)
                if not text:
                    return gr.update(visible=True, value="*1 clip detected*")
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                count = len(paragraphs)
                label = f"**{count} clips detected**" if count > 1 else "*1 clip detected*"
                return gr.update(visible=True, value=label)

            infer_text.change(update_clips_count, inputs=[infer_text, split_para_check], outputs=[infer_clips_count])
            split_para_check.change(update_clips_count, inputs=[infer_text, split_para_check], outputs=[infer_clips_count])
            
            # Sync model download
            infer_base_model.change(fn=download_voxcpm_model, inputs=[infer_base_model], outputs=[])
            
            infer_gen_btn.click(
                run_inference,
                inputs=[
                    infer_text,
                    infer_ref_audio,
                    infer_ref_text,
                    infer_lora,
                    infer_cfg,
                    infer_steps,
                    infer_seed,
                    infer_base_model,
                    infer_control,
                    split_para_check
                ],
                outputs=[infer_audio_out, infer_status_out],
            )

            dialogue_gen_btn.click(
                generate_dialogue,
                inputs=[
                    infer_lora,
                    infer_cfg,
                    infer_steps,
                    infer_seed,
                    infer_base_model,
                    split_para_check,
                    dialogue_row_count,
                    dialogue_silence_slider
                ] + dialogue_samples + dialogue_controls + dialogue_texts,
                outputs=[infer_audio_out, infer_status_out]
            )
            

        # === Prep Samples Tab ===
        with gr.Tab("🎙️ Prep Samples", id="tab_prep_samples") as tab_prep_samples:
            gr.Markdown(
                "### 🎙️ Reference Audio Library\n"
                "Manage and prepare audio samples for voice cloning. You can record, upload, transcribe, and organize your samples here.\n\n"
                "**Guide:**\n"
                "1. Use **Single Editor** to add or edit individual reference audios.\n"
                "2. Once uploaded, click **✨ Transcribe** to automatically get the text via Faster-Whisper.\n"
                "3. Review and edit the transcription manually if needed.\n"
                "4. Assign a **Sample ID** and click **💾 Save Sample** to add it to your permanent library."
            )
            
            with gr.Row():
                with gr.Column(scale=1, elem_classes="form-section"):
                    gr.Markdown("#### 📂 Your Samples")
                    sample_dropdown = gr.Dropdown(
                        choices=sample_choices,
                        value=default_sample_name,
                        label="Select Sample",
                        interactive=True
                    )
                    refresh_samples_btn = gr.Button("🔄 Refresh List", size="sm")
                    
                
                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown("#### 🎙️ Transcription & Editor")

                    gr.Markdown(
                        "### 🎙️ Add or Edit Audio\n"
                        "Use the **'X'** (top right of the player) to clear the preview and drag or click to upload a new audio.\n"
                        "Once uploaded, click **✨ Transcribe** to get the text, then **💾 Save Sample** to add it to your library."
                    )
                    
                    
                    prep_audio_player = gr.Audio(label="Audio Editor (Use Trim icon to edit)", type="filepath", interactive=True, value=default_audio)
                    prep_transcription = gr.Textbox(
                        label="Reference Text / Transcription",
                        placeholder="Transcription will appear here, or enter/edit text manually...",
                        lines=4,
                        interactive=True,
                        value=default_text
                    )
                    with gr.Row():
                        transcribe_prep_btn = gr.Button("✨ Transcribe", variant="secondary")
                        save_sample_name = gr.Textbox(label="Sample ID", placeholder="e.g. news_anchor_1", scale=2, value=default_sample_name)
                        save_sample_btn = gr.Button("💾 Save Sample", variant="primary", scale=1)
                    
                    prep_op_status = gr.Textbox(label="Operation Status", interactive=False)

            def on_sample_change(name):
                audio, text = load_sample(name)
                return gr.update(value=audio), gr.update(value=text), gr.update(value=name)

            sample_dropdown.change(on_sample_change, inputs=[sample_dropdown], outputs=[prep_audio_player, prep_transcription, save_sample_name])
            refresh_samples_btn.click(lambda: gr.update(choices=get_sample_choices()), outputs=[sample_dropdown])
            
            transcribe_prep_btn.click(fn=recognize_audio, inputs=[prep_audio_player], outputs=[prep_transcription])
            save_sample_btn.click(fn=save_prep_sample, inputs=[prep_audio_player, save_sample_name, prep_transcription], outputs=[prep_op_status]).then(
                fn=lambda: gr.update(choices=get_sample_choices()), outputs=[sample_dropdown]
            )

CUSTOM_CSS = """
.green-btn { background-color: #2e8b57 !important; color: white !important; }
.red-btn { background-color: #8b0000 !important; color: white !important; }
"""

if __name__ == "__main__":
    # Ensure lora directory exists
    os.makedirs("lora", exist_ok=True)
    app.queue().launch(
        server_name="127.0.0.1", 
        server_port=7860,
        inbrowser=True,
        css=CUSTOM_CSS
    )
