# lipsync_project/client_test_batch.py

import asyncio
import websockets
import json
import base64
import os

# --- Helper functions ---
def file_to_base64(filepath):
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def base64_to_file(base64_string, output_filepath):
    try:
        file_bytes = base64.b64decode(base64_string)
        with open(output_filepath, "wb") as f:
            f.write(file_bytes)
        print(f"Successfully saved to {output_filepath}")
    except Exception as e:
        print(f"Error decoding or saving file: {e}")

async def send_lipsync_batch_data():
    uri = "ws://localhost:8000/ws/lipsync/batch"
    
    # --- IMPORTANT: UPDATE THESE PATHS ---
    # Use a clear image of a single face and a short audio for testing
    # Relative to this script's location (lipsync_project)
    image_path = os.path.join("MuseTalk", "assets", "demo", "musk", "musk.png")
    # image_path = os.path.join("MuseTalk", "assets", "demo", "yongen", "yongen.jpeg")
    audio_path = os.path.join("MuseTalk", "data", "audio", "yongen.wav")
    # Make sure these files exist on your system!

    print(f"Loading image from: {image_path}")
    image_b64 = file_to_base64(image_path)
    print(f"Loading audio from: {audio_path}")
    audio_b64 = file_to_base64(audio_path)

    if not image_b64 or not audio_b64:
        print("Failed to encode one or both files to base64. Exiting.")
        return

    payload = {
        "image_base64": image_b64,
        "audio_base64": audio_b64
    }

    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as websocket:
        print(f"Sending data: {{'image_base64': '{image_b64[:30]}...', 'audio_base64': '{audio_b64[:30]}...'}}")
        await websocket.send(json.dumps(payload))
        print("Data sent. Waiting for response...")

        response = await websocket.recv()
        print(f"Received from server: {response[:200]}...") # Print first 200 chars

        try:
            response_json = json.loads(response)
            if response_json.get("status") == "success" and response_json.get("video_base64"):
                print("Received success response. Attempting to decode video.")
                output_video_path = "output_batch_video.mp4"
                base64_to_file(response_json["video_base64"], output_video_path)
                print(f"Batch video saved to: {output_video_path}")
            else:
                print(f"Error response: {response_json.get('message', 'No message')}")
        except json.JSONDecodeError:
            print("Received non-JSON response.")
        except Exception as e:
            print(f"An error occurred while processing response: {e}")

if __name__ == "__main__":
    # Ensure your FastAPI server (main.py) is running before executing this client.
    asyncio.run(send_lipsync_batch_data())