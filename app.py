import asyncio, base64, json, queue, threading, numpy as np, websockets, gradio as gr

# Config
VLLM_HOST, VLLM_PORT = "localhost", 8000
SAMPLE_RATE = 16_000
MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"

audio_queue = queue.Queue()
transcription_text = ""
is_running = False

async def websocket_handler():
    global transcription_text, is_running
    url = f"ws://{VLLM_HOST}:{VLLM_PORT}/v1/realtime"
    try:
        async with websockets.connect(url, ping_interval=5, ping_timeout=10) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "session.update", "model": MODEL_ID}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            
            async def send_audio():
                loop = asyncio.get_event_loop()
                while is_running:
                    try:
                        chunk = await loop.run_in_executor(None, lambda: audio_queue.get(timeout=0.1))
                        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
                    except queue.Empty:
                        continue

            async def receive_transcription():
                global transcription_text
                async for message in ws:
                    data = json.loads(message)
                    if data.get("type") == "transcription.delta":
                        transcription_text += data["delta"]

            await asyncio.gather(send_audio(), receive_transcription())
    except Exception as e:
        print(f"WS Error: {e}")

def start_recording():
    global transcription_text, is_running
    transcription_text, is_running = "", True
    def run_async():
        asyncio.run(websocket_handler())
    
    threading.Thread(target=run_async, daemon=True).start()
    return gr.update(interactive=False), gr.update(interactive=True), ""

def stop_recording():
    global is_running
    is_running = False
    return gr.update(interactive=True), gr.update(interactive=False), transcription_text

# NEU: Funktion zum Leeren der Box
def clear_box():
    global transcription_text
    transcription_text = ""
    return ""

def process_audio(audio):
    if audio is None or not is_running:
        return transcription_text
    
    sr, y = audio
    if len(y.shape) > 1: y = y.mean(axis=1)
    
    y = y.astype(np.float32) / 32767.0
    if sr != SAMPLE_RATE:
        y = np.interp(np.linspace(0, len(y)-1, int(len(y)*SAMPLE_RATE/sr)), np.arange(len(y)), y)
    
    pcm16 = (y * 32767).astype(np.int16)
    payload = base64.b64encode(pcm16.tobytes()).decode("utf-8")
    
    audio_queue.put(payload)
    return transcription_text

with gr.Blocks(title="Voxtral A6000") as demo:
    gr.Markdown(f"# Voxtral Real-time STT\nStabile Verbindung: München ↔ Nürnberg")
    with gr.Row():
        btn_start = gr.Button("Start", variant="primary")
        btn_stop = gr.Button("Stop", variant="stop", interactive=False)
    
    mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy")
    txt = gr.Textbox(label="Live-Transkript", lines=10)
    
    # NEU: Clear Button
    btn_clear = gr.Button("Clear Transcript", variant="secondary")

    btn_start.click(start_recording, outputs=[btn_start, btn_stop, txt])
    btn_stop.click(stop_recording, outputs=[btn_start, btn_stop, txt])
    
    # NEU: Verknüpfung des Clear Buttons
    btn_clear.click(clear_box, outputs=[txt])
    
    mic.stream(process_audio, inputs=[mic], outputs=[txt], show_progress="hidden")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7634)