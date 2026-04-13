# 🎙️ VoxCPM WebUI: Voice Cloning & Fine-Tuning

A comprehensive WebUI for fine-tuning and interacting with **VoxCPM** models. This application leverages Efficient LoRA (Low-Rank Adaptation) capabilities to enable high-quality voice cloning.

<img src="./assets/1_dataset_tab.png">

<img src="./assets/2_train_tab.png">

<img src="./assets/3_inference_tab.png">

<img src="./assets/4_inference_tab_lora.png">

<img src="./assets/5_prep_samples_tab.png">

---

## 🔄 Application Workflow

1.  **Data Preparation:** Upload/record audio and generate transcriptions using `Faster-Whisper`.
2.  **Fine-Tuning:** Configure LoRA or Full Fine-Tuning parameters and train your voice adapter.
3.  **Inference:** Generate speech using the base model combined with your trained weights.

*   **Persistent Torch & Triton cache**: Integration of `triton-windows` and a custom kernel caching system in `models/.cache`, enabling the full power of `torch.compile` for inference speed-up.
*   **Building the persistent cache for the first time, might take a up to 5 minutes. This is a one time process. Once cached, Subsequent generations will be significantly faster.**

---

## ⚙️ System Requirements & Hardware

### 💻 Software Dependencies
* **Python:** 3.10 – 3.11 (Recommended for stability during training).
* **PyTorch:** 2.5.0+
* **CUDA:** 12.0+
* **Format Support:** `.wav` is recommended.

### 🔌 Hardware Setup (VRAM Requirements)

| Model | LoRA Training | Full Fine-Tuning |
| :--- | :--- | :--- |
| **VoxCPM 1.5 (750M)** | ~12 GB VRAM | ~24 GB VRAM |
| **VoxCPM 2.0 (2B)** | ~20 GB VRAM | ~40 GB VRAM |

---

## 📊 Dataset & Audio Specifications

### 🎯 Clip Requirements
* **Format:** `.wav` is highly recommended. Other formats supported by `torchaudio` also work.
* **Duration:** **3–30 seconds** per clip is the "sweet spot."
    * *Warning:* Clips < 1s produce unstable results.
    * *Warning:* Very long clips increase VRAM usage and may be filtered by `max_batch_tokens`.
* **Sample Rate:** The dataloader resamples automatically. Your config `sample_rate` must match the **AudioVAE encoder** input:
    * **VoxCPM 1.0:** 16kHz
    * **VoxCPM 1.5:** 44.1kHz
    * **VoxCPM 2.0:** 16kHz (The encoder operates at 16kHz; the decoder outputs 48kHz).

### ✨ Preprocessing Tips
* **Trim Trailing Silence:** Keep silence to **< 0.5 seconds**. Excessive trailing silence is the leading cause of "infinite generation" issues after fine-tuning.
* **Normalize Volume:** Ensure consistent levels across all training samples.
* **Clean Transcripts:** Text must match audio **exactly**. Inaccurate transcripts degrade both cloning quality and text adherence.
* **Remove Noise:** The model is highly sensitive to background noise. Use clean, isolated voice recordings.

---

## 🛠️ Installation & Execution (Windows)

This project utilizes `uv` for lightning-fast dependency management.

### Setup Steps
1.  **Run Installer:** Double-click `install.bat`.
    * This installs `uv` via Winget (if not present).
    * Synchronizes the environment and installs all required libraries automatically.
2.  **Launch App:** Double-click `start.bat`.
3.  **Access:** Navigate to `http://127.0.0.1:7860` in your web browser.
