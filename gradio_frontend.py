import base64
import json
import os
import shutil
import time

import gradio as gr
import websockets

# --- Configuration ---
SERVER_URL = "ws://localhost:8000"
TEMP_CLIENT_DIR = "client_temp_files_gradio"
os.makedirs(TEMP_CLIENT_DIR, exist_ok=True)


# --- Helper functions ---
def _file_to_base64(filepath):
    if not filepath or not os.path.exists(filepath):
        print(f"Error: File not found or invalid: {filepath}")
        return None
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding file {filepath} to base64: {e}")
        return None


async def _connect_and_send(url, payload):
    """Helper to connect to a WebSocket, send payload, and receive response."""
    try:
        async with websockets.connect(url) as websocket:
            await websocket.send(json.dumps(payload))
            response_str = await websocket.recv()
            return json.loads(response_str)
    except websockets.exceptions.ConnectionClosedOK:
        print("WebSocket connection closed gracefully by server.")
        return {"status": "error", "message": "Connection closed by server."}
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket connection closed with error: {e}")
        return {"status": "error", "message": f"Connection closed with error: {e}"}
    except Exception as e:
        print(f"WebSocket error: {e}")
        return {"status": "error", "message": f"WebSocket connection failed: {e}"}


# --- Gradio Interface Functions (Batch Only) ---


async def process_batch_mode(image_path, audio_path, progress=gr.Progress()):
    if not image_path or not audio_path:
        return None, "Please upload both an image and an audio file."

    progress(0.1, desc="Encoding inputs...")
    image_b64 = _file_to_base64(image_path)
    audio_b64 = _file_to_base64(audio_path)

    if not image_b64 or not audio_b64:
        # Return current None/message and also print for server logs
        print(
            "Failed to encode image or audio to base64. Check file paths and content."
        )
        return (
            None,
            "Failed to encode image or audio to base64. Check file paths and content.",
        )

    payload = {"image_base64": image_b64, "audio_base64": audio_b64}

    progress(0.3, desc="Sending request to server...")
    # Use the batch endpoint URL
    response = await _connect_and_send(f"{SERVER_URL}/ws/lipsync/batch", payload)

    if response.get("status") == "success" and response.get("video_base64"):
        progress(0.7, desc="Decoding video and saving...")
        output_video_b64 = response["video_base64"]

        # Ensure unique filename for batch output
        timestamp = int(time.time())
        image_name = (
            os.path.basename(image_path).split(".")[0] if image_path else "image"
        )
        audio_name = (
            os.path.basename(audio_path).split(".")[0] if audio_path else "audio"
        )
        output_video_filename = (
            f"batch_output_{timestamp}_{image_name}_{audio_name}.mp4"
        )
        output_video_path = os.path.join(TEMP_CLIENT_DIR, output_video_filename)

        try:
            video_bytes = base64.b64decode(output_video_b64)
            with open(output_video_path, "wb") as f:
                f.write(video_bytes)
            progress(1.0, desc="Done!")
            return output_video_path, "Batch processing complete."
        except Exception as e:
            # Return current None/message and also print for server logs
            print(f"Error decoding or saving video: {e}")
            return None, f"Error decoding or saving video: {e}"
    else:
        # Return current None/message and also print for server logs
        server_message = response.get("message", "Unknown error")
        print(f"Server error: {server_message}")
        return None, f"Server error: {server_message}"


# --- Gradio UI Definition (Batch Only) ---
with gr.Blocks(
    css="#video_output_batch {max-width: 720px; max-height: 480px;}"
) as demo:
    gr.Markdown(
        """<h1 align='center'>MuseTalk Batch Lip-Sync</h1>
        <p align='center'>This Gradio app acts as a client to the MuseTalk FastAPI backend's batch processing mode.</p>
        """
    )

    # Removed Tabs, keeping only the Batch content
    gr.Markdown("## Batch Mode: Generate a full video from an image and an audio file.")
    with gr.Row():
        with gr.Column():
            # Default paths relative to the project root
            default_image = os.path.join(
                "MuseTalk", "assets", "demo", "musk", "musk.png"
            )
            default_audio = os.path.join("MuseTalk", "data", "audio", "yongen.wav")

            # Ensure default paths exist, if not, set to None or print warning
            if not os.path.exists(default_image):
                print(
                    f"Warning: Default image not found at {default_image}. Please provide an image."
                )
                default_image = None
            if not os.path.exists(default_audio):
                print(
                    f"Warning: Default audio not found at {default_audio}. Please provide audio."
                )
                default_audio = None

            batch_image_input = gr.Image(
                type="filepath", label="Input Avatar Image", value=default_image
            )
            batch_audio_input = gr.Audio(
                type="filepath", label="Input Driving Audio", value=default_audio
            )
            batch_button = gr.Button("Generate Batch Video", variant="primary")
        with gr.Column():
            # Keep the element ID for CSS
            batch_output_video = gr.Video(
                label="Generated Video", elem_id="video_output_batch"
            )
            batch_status_message = gr.Textbox(label="Status")

    batch_button.click(
        fn=process_batch_mode,
        inputs=[batch_image_input, batch_audio_input],
        outputs=[batch_output_video, batch_status_message],
    )


# --- Launch Gradio App ---
if __name__ == "__main__":
    # Remove client_temp_files_gradio on startup to ensure a clean slate
    if os.path.exists(TEMP_CLIENT_DIR):
        try:
            shutil.rmtree(TEMP_CLIENT_DIR)
            print(
                f"Cleaned up previous Gradio client temp directory: {TEMP_CLIENT_DIR}"
            )
        except Exception as e:
            print(f"Warning: Could not clean up {TEMP_CLIENT_DIR}: {e}")
    os.makedirs(TEMP_CLIENT_DIR, exist_ok=True)  # Recreate it

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
