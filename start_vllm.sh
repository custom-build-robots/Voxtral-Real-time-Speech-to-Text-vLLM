#!/bin/bash
cd ~/voxtral
source venv/bin/activate
export VLLM_DISABLE_COMPILE_CACHE=1

# Wir starten erst einmal auf EINER Karte, um den Bus-Konflikt auszuschließen
# Das ist für Voxtral (4B Modell) auf einer A6000 (48GB) völlig ausreichend!
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --enforce-eager