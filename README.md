# Voxtral Real-time Speech-to-Text (vLLM)

This project implements a high-performance, real-time speech transcription system using the **Mistral Voxtral-Mini-4B-Realtime-2602** model. It is designed to run on NVIDIA GPU servers (specifically tested on dual RTX A6000s) and provides a low-latency Gradio web interface.



## 🌟 Project Highlights
* **True Real-time:** Continuous streaming transcription using Mistral's sliding-window attention architecture.
* **Remote Optimized:** Engineered for stable performance across long distances (e.g., Munich ↔ Nuremberg).
* **High Efficiency:** Leverages vLLM's PagedAttention and optimized CUDA kernels.
* **Local & Private:** 100% self-hosted; no data leaves your server.

---

## 🛠 Installation & Setup

### 1. System Requirements
* **OS:** Linux (Ubuntu recommended)
* **GPU:** NVIDIA GPU with 16GB+ VRAM (RTX A6000 used in this setup)
* **Dependencies:** `ffmpeg` must be installed on the host system.
  ```bash
  sudo apt update && sudo apt install ffmpeg -y

### 2. Environment Setup
Navigate to your project directory and create a Python virtual environment.
  ```bash
  mkdir ~/voxtral && cd ~/voxtral
  python3 -m venv venv
  source venv/bin/activate




### 3. Install Packages
Install the nightly vLLM build and the required audio/UI libraries.
  ```bash
  pip install --upgrade pip
  pip install -U vllm --pre --extra-index-url [https://wheels.vllm.ai/nightly](https://wheels.vllm.ai/nightly)
  pip install gradio==5.15.0 websockets numpy soxr librosa soundfile mistral_common>=1.9.0
    
### 📂 File Structure
Place these three files in your ~/voxtral/ directory:

app.py: The Gradio web interface with WebSocket client logic.

start_vllm.sh: Shell script to launch the vLLM inference engine.

start_ui.sh: Shell script to launch the Gradio frontend.

Make scripts executable:
  ```bash
  chmod +x start_vllm.sh start_ui.sh
    



### 🚀 How to Run
1. Launch the Backend (Inference)
In your first terminal, start the vLLM server:

  ```bash
  ./start_vllm.sh




Wait for the log: INFO: Uvicorn running on http://0.0.0.0:8000.

### 2. Launch the Frontend (UI)
In a second terminal, start the web interface:

  ```bash
  ./start_ui.sh




### 3. Connect via SSH Tunnel (optional Remote Access)
If you are accessing the server from a different location, open a terminal on your local computer:

  ```bash
  ssh -L 7634:localhost:7634 ingmar@<your-server-ip>





### 4. Open in Browser
Visit http://localhost:7634. Note: You must use localhost so the browser permits microphone access over an unencrypted connection.

### ⚙️ Technical Configuration
GPU Management: Configured for --tensor-parallel-size 1 to ensure stability and compatibility with other concurrent GPU tasks (like ComfyUI or Stable Diffusion).

VRAM Optimization: Uses --max-model-len 32768 to balance high context (approx. 45 mins of audio) with memory efficiency.

Latency: Audio is resampled to 16kHz and streamed via WebSockets for sub-second response times.
