# lipsync_project/client_test_stream.py

import asyncio
import websockets
import json
import base64
import os
import io
import cv2
import numpy as np
from pydub import AudioSegment
import imageio # For final video stitching
import shutil
import time

# --- Configuration ---
# Ensure this matches your FastAPI server's address
# For local testing, it's typically ws://localhost:8000
# For Gradio, it will connect to the same server that launched it if both are on same machine.
SERVER_URL = "ws://localhost:8000" 
TEMP_CLIENT_DIR = "client_temp_files"
os.makedirs(TEMP_CLIENT_DIR, exist_ok=True)

# --- Helper functions ---
def file_to_base64(filepath):
    if not filepath or not os.path.exists(filepath):
        print(f"File not found or invalid: {filepath}")
        return None
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding file {filepath} to base64: {e}")
        return None

def base64_to_cv2_image_rgb(base64_string):
    """Decodes a base64 JPEG string to an OpenCV image (BGR), then converts to RGB."""
    try:
        img_bytes = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_np_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_np_bgr is None:
            raise ValueError("cv2.imdecode returned None. Corrupt image data?")
        img_np_rgb = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB
        return img_np_rgb
    except Exception as e:
        print(f"Error decoding base64 to image: {e}")
        return None

async def send_lipsync_stream_data():
    uri = f"{SERVER_URL}/ws/lipsync/stream"
    
    # --- IMPORTANT: UPDATE THESE PATHS ---
    # Use a clear image of a single face and a short audio for testing
    image_path = os.path.join("MuseTalk", "assets", "demo", "musk", "musk.png")
    audio_path = os.path.join("MuseTalk", "data", "audio", "yongen.wav") # A slightly longer audio is good for streaming test
    
    # --- Audio Chunking Parameters ---
    # Duration of each audio chunk to send (in milliseconds)
    CHUNK_DURATION_MS = 100 

    print(f"Loading initial image from: {image_path}")
    image_b64 = file_to_base64(image_path)
    print(f"Loading audio for streaming from: {audio_path}")
    
    if not image_b64:
        print("Failed to encode initial image to base64. Exiting.")
        return

    try:
        full_audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}. Ensure it's a valid audio format.")
        return

    # --- WebSocket connection and message handling ---
    print(f"Connecting to {uri}...")
    session_id = None # Initialize session_id here
    total_frames_received = 0 # Initialize here to prevent UnboundLocalError
    
    output_dir = os.path.join(TEMP_CLIENT_DIR, "stream_output_frames") # Specific dir for this run
    os.makedirs(output_dir, exist_ok=True) # Ensure it exists for this run

    try:
        async with websockets.connect(uri) as websocket:
            print("Sending init message...")
            init_payload = {
                "type": "init",
                "image_base64": image_b64,
                "desired_fps": 25 # Can be adjusted
            }
            await websocket.send(json.dumps(init_payload))
            
            init_response = await websocket.recv()
            init_data = json.loads(init_response)
            
            if init_data.get("status") == "success":
                session_id = init_data["session_id"]
                print(f"Session initialized with ID: {session_id}. Starting audio stream...")
                
                for i in range(0, len(full_audio), CHUNK_DURATION_MS):
                    chunk = full_audio[i : i + CHUNK_DURATION_MS]
                    
                    audio_chunk_io = io.BytesIO()
                    chunk.export(audio_chunk_io, format="wav")
                    audio_chunk_b64 = base64.b64encode(audio_chunk_io.getvalue()).decode("utf-8")

                    audio_chunk_payload = {
                        "type": "audio_chunk",
                        "session_id": session_id,
                        "audio_base64": audio_chunk_b64
                    }
                    await websocket.send(json.dumps(audio_chunk_payload))
                    
                    # Wait for response (video frames) for this chunk
                    response_str = await websocket.recv()
                    try:
                        response_data = json.loads(response_str)
                        if response_data.get("type") == "video_frames" and response_data.get("frames_base64_jpeg"):
                            frames_b64 = response_data["frames_base64_jpeg"]
                            print(f"Received {len(frames_b64)} video frames for audio chunk {i}-{i+CHUNK_DURATION_MS}ms.")
                            for j, frame_b64 in enumerate(frames_b64):
                                # FIX: Use the RGB decoding helper function
                                frame_np_rgb = base64_to_cv2_image_rgb(frame_b64)
                                if frame_np_rgb is not None:
                                    # FIX: Convert RGB to BGR before saving with cv2.imwrite
                                    frame_np_bgr = cv2.cvtColor(frame_np_rgb, cv2.COLOR_RGB2BGR)
                                    frame_filename = os.path.join(output_dir, f"frame_{total_frames_received:08d}.jpg")
                                    cv2.imwrite(frame_filename, frame_np_bgr)
                                    total_frames_received += 1
                        else:
                            print(f"Unexpected response for audio chunk: {response_data.get('message', response_str)}")
                    except json.JSONDecodeError:
                        print(f"Received non-JSON response for audio chunk: {response_str[:100]}...")
                    except Exception as e:
                        print(f"Error processing video frames: {e}")

                    await asyncio.sleep(CHUNK_DURATION_MS / 1000.0) # Simulate real-time delay

                # --- End stream ---
                print("Finished sending audio chunks. Sending end_stream message.")
                end_payload = {
                    "type": "end_stream",
                    "session_id": session_id
                }
                await websocket.send(json.dumps(end_payload))
                end_response = await websocket.recv()
                print(f"End stream response: {end_response}")
                
            else:
                print(f"Failed to initialize session: {init_data.get('message', init_response)}")

    except websockets.exceptions.ConnectionClosedOK:
        print("WebSocket connection closed gracefully by server.")
    except Exception as e:
        print(f"An error occurred while connecting or streaming: {e}")
    finally:
        # Final cleanup or reporting
        print(f"Total frames received and saved: {total_frames_received}")
        print(f"Saved all frames to {output_dir}. You can combine them into a video using ffmpeg:")
        print(f"ffmpeg -framerate 25 -i {output_dir}/frame_%08d.jpg -c:v libx264 -pix_fmt yuv420p output_streamed_video.mp4")
        # You might want to automatically combine the video here for easier testing
        # using imageio or moviepy if desired.
        
        # Simple auto-stitch if frames were saved
        if total_frames_received > 0:
            final_output_video_path = os.path.join(TEMP_CLIENT_DIR, f"final_stream_output_{int(time.time())}.mp4")
            try:
                # Need to read images in BGR if saved as BGR, then convert to RGB for imageio
                # Or read directly in RGB if saved in RGB. Assuming cv2.imwrite saved BGR.
                # So imageio needs RGB.
                frames_for_stitch = []
                for i in range(total_frames_received):
                    frame_path = os.path.join(output_dir, f"frame_{i:08d}.jpg")
                    frame_bgr = cv2.imread(frame_path)
                    if frame_bgr is not None:
                        frames_for_stitch.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    else:
                        print(f"Warning: Could not read frame {frame_path} for stitching.")
                
                if frames_for_stitch:
                    writer = imageio.get_writer(final_output_video_path, fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
                    for frame_rgb in frames_for_stitch:
                        writer.append_data(frame_rgb)
                    writer.close()
                    print(f"Automatically stitched video to: {final_output_video_path}")
                else:
                    print("No valid frames to stitch.")
            except Exception as stitch_e:
                print(f"Error automatically stitching video: {stitch_e}")
            finally:
                # Clean up individual frame files
                try:
                    shutil.rmtree(output_dir)
                    print(f"Cleaned up temporary frames directory: {output_dir}")
                except Exception as e_clean:
                    print(f"Error cleaning up temp frames dir {output_dir}: {e_clean}")


if __name__ == "__main__":
    # Ensure your FastAPI server (main.py) is running before executing this client.
    asyncio.run(send_lipsync_stream_data())