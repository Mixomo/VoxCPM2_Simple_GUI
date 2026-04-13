import os
import sys
import json
import yaml
import datetime
import subprocess
import threading
import gradio as gr
import torch
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
has_cache = compile_cache_dir.exists() and any(compile_cache_dir.iterdir())
compile_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache_dir)
# Disable CUDA graph trees to allow safe model unload/reload within the same process.
# CUDA graphs use thread-local state that corrupts on model recreation, causing AssertionError.
# All other torch.compile optimizations (kernel fusion, etc.) remain fully active.
os.environ["TORCHINDUCTOR_CUDAGRAPH_TREES"] = "0"

print("----------------------------------------------------------------", file=sys.stderr)
if has_cache:
    print(f"INFO: Persistent compilation cache detected at {compile_cache_dir}", file=sys.stderr)
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
CHIME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "inference_training_done.wav")

def play_done_chime():
    if os.path.exists(CHIME_PATH):
        try:
            import winsound
            winsound.PlaySound(CHIME_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"Failed to play chime: {e}", file=sys.stderr)
    else:
        try:
            import winsound
            winsound.MessageBeep()
        except: pass
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
        asr_model = WhisperModel("large-v3", device=device, compute_type=compute_type)
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
    return sorted(choices)


def load_sample(sample_name):
    if not sample_name:
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


def run_inference(text, prompt_wav, prompt_text, lora_selection, cfg_scale, steps, seed, model_choice=None, control=None, progress=gr.Progress(), **kwargs):
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
    
    # Standard generation with optional control instruction
    final_control = (control or kwargs.get("control") or "").strip()
    final_text = f"({final_control}){text}" if final_control else text

    try:
        audio_np = current_model.generate(
            text=final_text,
            prompt_wav_path=final_prompt_wav,
            prompt_text=final_prompt_text,
            cfg_value=cfg_scale,
            inference_timesteps=steps,
        )
        play_done_chime()
        return (current_model.tts_model.sample_rate, audio_np), "Generation Success"
    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"Error: {str(e)}"


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
            play_done_chime()
    elif training_process is not None and training_process.poll() is None:
        # Currently running
        training_finished_played = False
        
    return log, None


# --- GUI Layout ---
with gr.Blocks(title="VoxCPM - Simple GUI | Inference + LoRa Training") as app:
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

        # === Inference Tab ===
        with gr.Tab("🔊 Inference") as tab_infer:
            gr.Markdown("### 🎙️ Unified Voice Synthesis")
            gr.Markdown("> **Optimization Note:** Hardware acceleration (torch.compile) is active. The first generation after loading a new model or LoRA rank may take **up to 5 minutes** to optimize. Subsequent runs will be near-instant.")
            
            with gr.Row():
                # --- Left Column: Reference & Voice Selection ---
                with gr.Column(scale=1, elem_classes="form-section"):
                    gr.Markdown("#### 🎙️ Voice & Reference")
                    with gr.Row():
                        infer_sample_select = gr.Dropdown(
                            choices=get_sample_choices(),
                            label="Quick Sample Select",
                            info="Load a reference from your 'samples' library.",
                            scale=4
                        )
                        refresh_infer_sample_btn = gr.Button("🔄", scale=1, min_width=50)
                    
                    infer_ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                    infer_ref_text = gr.Textbox(
                        label="Reference Text (Transcript)", 
                        placeholder="Automatic transcription if left empty...", 
                        lines=2
                    )
                    
                    gr.Markdown("---")
                    infer_lora = gr.Dropdown(
                        label="Voice Clone (LoRA)",
                        choices=["None"] + [ckpt[0] for ckpt in scan_lora_checkpoints(with_info=True)],
                        value="None",
                        info="Select 'None' for standard VoxCPM voice."
                    )
                    refresh_infer_lora_btn = gr.Button("🔄 Refresh Available Voices", variant="secondary", size="sm")

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
                        label="Style / Control (Optional)", 
                        placeholder="e.g. A happy energetic tone / Whispering softly...", 
                        lines=2
                    )
                    
                    with gr.Accordion("⚙️ Advanced Parameters", open=False):
                        infer_cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
                        infer_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, value=10, step=1)
                        infer_seed = gr.Number(label="Seed (-1 for Random)", value=-1, precision=0)
                        infer_base_model = gr.Dropdown(
                            label="Core Model", 
                            choices=list(VOXCPM_MODELS.keys()), 
                            value="VoxCPM-2.0",
                            info="Select the base foundation model."
                        )

                    infer_gen_btn = gr.Button("⚡ Generate Speech", variant="primary", size="lg", elem_classes="button-primary")

            with gr.Row():
                infer_audio_out = gr.Audio(label="Generated Audio")
                infer_status_out = gr.Textbox(label="System Status", interactive=False)
            
            # --- Unified Event Handlers ---
            def smart_asr_unified(audio, current_text):
                if current_text and current_text.strip():
                    return current_text
                return recognize_audio(audio)

            infer_sample_select.change(load_sample, inputs=[infer_sample_select], outputs=[infer_ref_audio, infer_ref_text])
            refresh_infer_sample_btn.click(lambda: gr.update(choices=get_sample_choices()), outputs=[infer_sample_select])
            refresh_infer_lora_btn.click(refresh_loras, outputs=[infer_lora])
            infer_ref_audio.change(fn=smart_asr_unified, inputs=[infer_ref_audio, infer_ref_text], outputs=[infer_ref_text])
            
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
                    infer_control
                ],
                outputs=[infer_audio_out, infer_status_out],
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
                        choices=get_sample_choices(),
                        value=get_sample_choices()[0] if get_sample_choices() else None,
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
                    
                    
                    prep_audio_player = gr.Audio(label="Audio Editor (Use Trim icon to edit)", type="filepath", interactive=True)
                    prep_transcription = gr.Textbox(
                        label="Reference Text / Transcription",
                        placeholder="Transcription will appear here, or enter/edit text manually...",
                        lines=4,
                        interactive=True
                    )
                    with gr.Row():
                        transcribe_prep_btn = gr.Button("✨ Transcribe", variant="secondary")
                        save_sample_name = gr.Textbox(label="Sample ID", placeholder="e.g. news_anchor_1", scale=2)
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

if __name__ == "__main__":
    # Ensure lora directory exists
    os.makedirs("lora", exist_ok=True)
    app.queue().launch(
        server_name="127.0.0.1", 
        server_port=7860,
        inbrowser=True
    )
