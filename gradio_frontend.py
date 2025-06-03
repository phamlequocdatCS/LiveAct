# lipsync_project/gradio_frontend.py

import gradio as gr
import os
import asyncio
import websockets
import json
import base64
import io
import cv2
import numpy as np
from pydub import AudioSegment
import imageio
import shutil
import time

# --- Configuration ---
SERVER_URL = "ws://localhost:8000" # Ensure this matches your FastAPI server's address
TEMP_CLIENT_DIR = "client_temp_files_gradio" # Changed to avoid conflict with client_test_stream.py
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

def _base64_to_cv2_image_rgb(base64_string):
    """Decodes a base64 JPEG string to an OpenCV image (BGR), then converts to RGB."""
    try:
        img_bytes = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_np_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Decodes JPEG to BGR
        if img_np_bgr is None:
            raise ValueError("cv2.imdecode returned None. Corrupt image data?")
        img_np_rgb = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        return img_np_rgb
    except Exception as e:
        print(f"Error decoding base64 to image for display: {e}")
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

# --- Gradio Interface Functions ---

async def process_batch_mode(image_path, audio_path, progress=gr.Progress()):
    if not image_path or not audio_path:
        return None, "Please upload both an image and an audio file."

    progress(0.1, desc="Encoding inputs...")
    image_b64 = _file_to_base64(image_path)
    audio_b64 = _file_to_base64(audio_path)

    if not image_b64 or not audio_b64:
        return None, "Failed to encode image or audio to base64. Check file paths and content."

    payload = {
        "image_base64": image_b64,
        "audio_base64": audio_b64
    }

    progress(0.3, desc="Sending request to server...")
    response = await _connect_and_send(f"{SERVER_URL}/ws/lipsync/batch", payload)

    if response.get("status") == "success" and response.get("video_base64"):
        progress(0.7, desc="Decoding video and saving...")
        output_video_b64 = response["video_base64"]
        # Ensure unique filename for batch output
        output_video_path = os.path.join(TEMP_CLIENT_DIR, f"batch_output_{int(time.time())}_{os.path.basename(image_path).split('.')[0]}.mp4")
        
        try:
            video_bytes = base64.b64decode(output_video_b64)
            with open(output_video_path, "wb") as f:
                f.write(video_bytes)
            progress(1.0, desc="Done!")
            return output_video_path, "Batch processing complete."
        except Exception as e:
            return None, f"Error decoding or saving video: {e}"
    else:
        return None, f"Server error: {response.get('message', 'Unknown error')}"

async def process_stream_mode(image_path, audio_info, progress=gr.Progress(track_tqdm=True)):
    if not image_path or audio_info is None: # audio_info can be None if no file is uploaded and mic is not used
        return None, "Please upload an image and provide audio (from file or microphone)."

    # audio_info is the path to the audio file (string) if uploaded or recorded
    audio_path = audio_info 
    if not os.path.exists(audio_path):
        return None, f"Audio file not found at: {audio_path}. Please ensure audio is recorded or uploaded correctly."

    session_id = None
    temp_session_frames_dir = os.path.join(TEMP_CLIENT_DIR, f"stream_frames_{int(time.time())}_{os.path.basename(image_path).split('.')[0]}")
    os.makedirs(temp_session_frames_dir, exist_ok=True)
    
    # This list will store RGB frames for imageio, or paths to saved BGR frames if cv2.imwrite is used inside loop
    frames_for_video_stitching = [] 

    try:
        # --- 1. Init Stream ---
        progress(0.1, desc="Initializing stream session...")
        image_b64 = _file_to_base64(image_path)
        if not image_b64:
            raise ValueError("Failed to encode image for stream initialization.")

        init_payload = {
            "type": "init",
            "image_base64": image_b64,
            "desired_fps": 25 # Or get from a Gradio input
        }

        async with websockets.connect(f"{SERVER_URL}/ws/lipsync/stream") as websocket:
            await websocket.send(json.dumps(init_payload))
            init_response_str = await websocket.recv()
            init_response = json.loads(init_response_str)

            if init_response.get("status") == "success":
                session_id = init_response["session_id"]
                progress(0.2, desc=f"Session initialized: {session_id}. Processing audio chunks...")
            else:
                raise RuntimeError(f"Stream init failed: {init_response.get('message', 'Unknown error')}")

            # --- 2. Stream Audio Chunks ---
            progress(0.3, desc="Reading audio file...")
            try:
                full_audio = AudioSegment.from_file(audio_path)
            except Exception as e:
                raise ValueError(f"Error loading audio file '{audio_path}': {e}")

            CHUNK_DURATION_MS = 100 # Send 100ms chunks
            num_chunks = (len(full_audio) + CHUNK_DURATION_MS -1) // CHUNK_DURATION_MS # Calculate total chunks for tqdm

            for i in progress.tqdm(range(num_chunks), desc="Sending audio chunks & receiving frames"):
                start_ms = i * CHUNK_DURATION_MS
                end_ms = start_ms + CHUNK_DURATION_MS
                chunk = full_audio[start_ms : end_ms]
                
                audio_chunk_io = io.BytesIO()
                chunk.export(audio_chunk_io, format="wav") # Export to WAV bytes in memory
                audio_chunk_b64 = base64.b64encode(audio_chunk_io.getvalue()).decode("utf-8")

                audio_chunk_payload = {
                    "type": "audio_chunk",
                    "session_id": session_id,
                    "audio_base64": audio_chunk_b64
                }
                
                await websocket.send(json.dumps(audio_chunk_payload))
                
                response_str = await websocket.recv()
                response_data = json.loads(response_str)

                if response_data.get("type") == "video_frames" and response_data.get("frames_base64_jpeg"):
                    frames_b64_jpeg = response_data["frames_base64_jpeg"]
                    for frame_b64 in frames_b64_jpeg:
                        # _base64_to_cv2_image_rgb decodes JPEG (which is BGR) and returns RGB
                        frame_np_rgb = _base64_to_cv2_image_rgb(frame_b64)
                        if frame_np_rgb is not None:
                            frames_for_video_stitching.append(frame_np_rgb) # Append RGB frame directly for imageio
                elif response_data.get("status") == "error":
                     print(f"Server error processing chunk: {response_data.get('message', 'Unknown error')}")
                else:
                    print(f"Warning: Unexpected response for audio chunk: {response_data.get('message', response_str)}")
                
                await asyncio.sleep(max(0, (CHUNK_DURATION_MS / 1000.0) - 0.02)) # Small adjustment for processing overhead

            # --- 3. End Stream ---
            progress(0.9, desc="Ending stream and stitching video...")
            end_payload = {
                "type": "end_stream",
                "session_id": session_id
            }
            await websocket.send(json.dumps(end_payload))
            end_stream_response_str = await websocket.recv() 
            print(f"End stream response: {end_stream_response_str}")


            # --- 4. Stitch Frames into Final Video ---
            if not frames_for_video_stitching:
                # Check if temp_session_frames_dir exists and has files if frames_for_video_stitching is empty
                # This part is now less relevant as we directly append RGB frames.
                raise RuntimeError("No frames were generated or collected during streaming.")
            
            output_video_path = os.path.join(TEMP_CLIENT_DIR, f"stream_output_{int(time.time())}_{os.path.basename(image_path).split('.')[0]}.mp4")
            writer = imageio.get_writer(output_video_path, fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
            
            for frame_rgb in frames_for_video_stitching: # frames_for_video_stitching contains RGB frames
                writer.append_data(frame_rgb)
            writer.close()
            
            progress(1.0, desc="Stream processing complete!")
            return output_video_path, "Streaming complete. Video generated."

    except websockets.exceptions.ConnectionClosedOK:
        message = "Stream connection closed gracefully by server."
        print(message)
        return None, message
    except websockets.exceptions.ConnectionClosedError as e:
        message = f"Stream connection closed with error: {e}"
        print(message)
        return None, message
    except Exception as e:
        message = f"Error during streaming: {e}"
        print(message)
        # If an error occurs, try to clean up session on server if session_id was obtained
        if session_id:
            print(f"Attempting to clean up session {session_id} on server due to error...")
            try:
                async def cleanup_session_on_error():
                    uri = f"{SERVER_URL}/ws/lipsync/stream"
                    async with websockets.connect(uri) as ws_cleanup:
                        end_payload = {"type": "end_stream", "session_id": session_id}
                        await ws_cleanup.send(json.dumps(end_payload))
                        await ws_cleanup.recv() # Wait for server ack
                        print(f"Cleanup request sent for session {session_id}")
                asyncio.run(cleanup_session_on_error()) # Run in new event loop if current one is closing
            except Exception as cleanup_e:
                print(f"Error sending cleanup for session {session_id}: {cleanup_e}")
        return None, message
    finally:
        # Clean up temporary frames directory (if it was used for saving intermediate jpegs)
        # This is less relevant now as frames are kept in memory, but good for robustness
        if os.path.exists(temp_session_frames_dir):
            try:
                shutil.rmtree(temp_session_frames_dir)
                print(f"Cleaned up {temp_session_frames_dir}")
            except Exception as e_clean:
                print(f"Error cleaning up temp stream dir {temp_session_frames_dir}: {e_clean}")

# --- Gradio UI Definition ---
with gr.Blocks(css="#video_output_batch, #video_output_stream {max-width: 720px; max-height: 480px;}") as demo:
    gr.Markdown(
        """<h1 align='center'>MuseTalk: Real-Time High-Fidelity Video Dubbing</h1>
        <p align='center'>This Gradio app acts as a client to the MuseTalk FastAPI backend. It allows you to use both batch and real-time streaming modes.</p>
        """
    )

    with gr.Tabs():
        with gr.TabItem("Batch Mode"):
            gr.Markdown("## Batch Mode: Generate a full video from an image and an audio file.")
            with gr.Row():
                with gr.Column():
                    batch_image_input = gr.Image(type="filepath", label="Input Avatar Image", value=os.path.join("MuseTalk", "assets", "demo", "musk", "musk.png"))
                    batch_audio_input = gr.Audio(type="filepath", label="Input Driving Audio", value=os.path.join("MuseTalk", "data", "audio", "yongen.wav"))
                    batch_button = gr.Button("Generate Batch Video", variant="primary")
                with gr.Column():
                    batch_output_video = gr.Video(label="Generated Video", elem_id="video_output_batch")
                    batch_status_message = gr.Textbox(label="Status")
            
            batch_button.click(
                fn=process_batch_mode,
                inputs=[batch_image_input, batch_audio_input],
                outputs=[batch_output_video, batch_status_message]
            )

        with gr.TabItem("Streaming Mode"):
            gr.Markdown("## Streaming Mode: Stream audio from file/microphone and receive real-time video frames.")
            gr.Markdown("Note: The video will be stitched and displayed only after the entire audio stream is processed.")
            with gr.Row():
                with gr.Column():
                    stream_image_input = gr.Image(type="filepath", label="Input Avatar Image", value=os.path.join("MuseTalk", "assets", "demo", "musk", "musk.png"))
                    # Corrected: Ensure your Gradio version supports the 'source' argument
                    # If not, you might need to remove 'source' or use a different component for microphone.
                    stream_audio_input = gr.Audio(type="filepath", label="Input Driving Audio (or Microphone)", sources=["microphone", "upload"])
                    stream_button = gr.Button("Start Streaming Lipsync", variant="primary")
                with gr.Column():
                    stream_output_video = gr.Video(label="Generated Streamed Video", elem_id="video_output_stream")
                    stream_status_message = gr.Textbox(label="Status")

            stream_button.click(
                fn=process_stream_mode,
                inputs=[stream_image_input, stream_audio_input],
                outputs=[stream_output_video, stream_status_message]
            )

# --- Launch Gradio App ---
if __name__ == "__main__":
    # Remove client_temp_files_gradio on startup to ensure a clean slate
    if os.path.exists(TEMP_CLIENT_DIR):
        try:
            shutil.rmtree(TEMP_CLIENT_DIR)
            print(f"Cleaned up previous Gradio client temp directory: {TEMP_CLIENT_DIR}")
        except Exception as e:
            print(f"Warning: Could not clean up {TEMP_CLIENT_DIR}: {e}")
    os.makedirs(TEMP_CLIENT_DIR, exist_ok=True) # Recreate it

    # Ensure asyncio event loop is managed correctly if Gradio runs in a separate thread
    # For simple cases, Gradio handles this, but for complex async, explicit loop management might be needed.
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    
    demo.launch(
        server_name="0.0.0.0", # Allow access from network if needed
        server_port=7860,    # Gradio will run on this port
        share=False,         # Set to True to get a public Gradio link (useful for sharing)
        # In newer Gradio, you might not need to specify loop explicitly here.
    )