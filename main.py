# ruff: noqa: E402

import asyncio
import base64
import json
import os
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import uvicorn
import websockets  # Kept for WebSocket handling
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse  # Kept for the simple test HTML
from pydantic import BaseModel, ValidationError

# --- Add MuseTalk directory to sys.path ---
LIPSYNC_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MUSE_TALK_DIR = os.path.join(LIPSYNC_PROJECT_ROOT, "MuseTalk")
if MUSE_TALK_DIR not in sys.path:
    sys.path.append(MUSE_TALK_DIR)


import torch  # For device checks
from transformers import WhisperModel

from MuseTalk.musetalk.utils.audio_processor import AudioProcessor
from MuseTalk.musetalk.utils.face_parsing import FaceParsing
from MuseTalk.musetalk.utils.utils import load_all_model
from MuseTalk.musetalk_adapter_batch import MuseTalkBatchGenerator

# Directory for temporary files for all operations
TEMP_BASE_DIR = os.path.join(LIPSYNC_PROJECT_ROOT, "temp_files")
os.makedirs(TEMP_BASE_DIR, exist_ok=True)

# Global variables for shared models
shared_models: SimpleNamespace = SimpleNamespace()
batch_args_config: SimpleNamespace = SimpleNamespace()
musetalk_batch_generator_instance: MuseTalkBatchGenerator = None


# --- FastAPI Lifespan Events for Model Loading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global musetalk_batch_generator_instance, shared_models, batch_args_config

    print("FastAPI application startup...")
    try:
        # --- 1. Define common configuration for MuseTalk models ---
        common_cfg = SimpleNamespace()
        common_cfg.gpu_id = 0
        common_cfg.vae_type = "sd-vae"
        common_cfg.unet_model_path = os.path.join("models", "musetalkV15", "unet.pth")
        common_cfg.unet_config = os.path.join("models", "musetalkV15", "musetalk.json")
        common_cfg.whisper_dir = os.path.join("models", "whisper")
        common_cfg.use_float16 = True  # Set to True for faster inference

        # Determine device and store it in shared_models for global access
        shared_models.device_str = (
            f"cuda:{common_cfg.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        shared_models.device = torch.device(shared_models.device_str)
        print(f"Using device: {shared_models.device_str}")

        # Check critical file paths relative to MUSE_TALK_DIR
        # Construct absolute paths based on where the MuseTalk library is
        abs_unet_model_path = os.path.join(MUSE_TALK_DIR, common_cfg.unet_model_path)
        abs_unet_config = os.path.join(MUSE_TALK_DIR, common_cfg.unet_config)
        # Note: vae_type like "sd-vae" maps to a directory name, config is inside.
        # vae config is actually loaded from the VAE path itself by the VAE class
        abs_vae_dir = os.path.join(MUSE_TALK_DIR, "models", common_cfg.vae_type)
        abs_whisper_dir = os.path.join(MUSE_TALK_DIR, common_cfg.whisper_dir)
        abs_face_parse_model_pth = os.path.join(
            MUSE_TALK_DIR, "models", "face-parse-bisent", "79999_iter.pth"
        )
        abs_face_parse_resnet_path = os.path.join(
            MUSE_TALK_DIR, "models", "face-parse-bisent", "resnet18-5c106cde.pth"
        )

        # List paths to check for existence
        paths_to_check = [
            abs_unet_model_path,
            abs_unet_config,
            abs_vae_dir,  # Check the VAE directory exists
            abs_whisper_dir,
            abs_face_parse_model_pth,
            abs_face_parse_resnet_path,
        ]

        print("Checking model file paths...")
        for path_to_check in paths_to_check:
            if not os.path.exists(path_to_check):
                # Print error and mark models as not loaded
                print(f"CRITICAL FILE NOT FOUND: {path_to_check}")
                raise FileNotFoundError(
                    f"MuseTalk critical file/dir not found at application startup: {path_to_check}"
                )
            # else:
            #     print(f"Found: {path_to_check}") # Uncomment for detailed debugging

        # --- 2. Load common models once ---
        print("Loading shared MuseTalk models (VAE, UNet, PE, Whisper, FaceParser)...")
        shared_models.vae, shared_models.unet, shared_models.pe = load_all_model(
            unet_model_path=abs_unet_model_path,
            vae_type=common_cfg.vae_type,  # VAE type is a string, not a path here
            unet_config=abs_unet_config,
            device=shared_models.device_str,
            use_float16=common_cfg.use_float16,
        )

        # Determine and store the weight_dtype
        shared_models.weight_dtype = (
            torch.float16 if common_cfg.use_float16 else torch.float32
        )

        # Apply precision conversion for PE and UNet if use_float16 is true
        if common_cfg.use_float16:
            print("Applying float16 precision to PE and UNet...")
            shared_models.pe = shared_models.pe.half().to(shared_models.device)
            shared_models.unet.model = shared_models.unet.model.half().to(
                shared_models.device
            )
            # VAE is handled internally by the VAE class which respects use_float16
            print("PE and UNet converted to float16 and moved to device.")
        else:
            # Ensure models are on the correct device if not using float16
            shared_models.pe = shared_models.pe.to(shared_models.device)
            # shared_models.vae.vae = shared_models.vae.vae.to(shared_models.device) # VAE class handles this
            shared_models.unet.model = shared_models.unet.model.to(shared_models.device)
            print("Running PE and UNet in float32 precision and moved to device.")

        # AudioProcessor initialization
        shared_models.audio_processor = AudioProcessor(
            feature_extractor_path=abs_whisper_dir
        )
        # Initialize WhisperModel using the absolute path to the directory
        shared_models.whisper = WhisperModel.from_pretrained(abs_whisper_dir)
        shared_models.whisper = (
            shared_models.whisper.to(
                device=shared_models.device, dtype=shared_models.weight_dtype
            )
            .eval()
            .requires_grad_(False)
        )

        # FaceParsing (already patched in MuseTalk to handle absolute paths internally via MUSE_TALK_ROOT_DIR sys.path)
        shared_models.face_parser = FaceParsing(
            left_cheek_width=90,  # Default
            right_cheek_width=90,  # Default
        )
        print("Shared MuseTalk models loaded successfully.")

        # --- 3. Configure and Initialize Batch Mode Generator ---
        batch_args_config.fps = 25
        batch_args_config.audio_padding_length_left = 2
        batch_args_config.audio_padding_length_right = 2
        batch_args_config.batch_size = 8
        batch_args_config.version = "v15"
        batch_args_config.bbox_shift = 0
        batch_args_config.extra_margin = 10
        batch_args_config.parsing_mode = "jaw"  # This is used in blending

        musetalk_batch_generator_instance = MuseTalkBatchGenerator(
            args_config=batch_args_config,
            preloaded_models=shared_models,
            device=shared_models.device_str,  # Use the determined device string
        )
        print("Batch Mode Generator initialized.")

        # Removed Real-Time Mode Configuration

    except Exception as e:
        print(
            f"CRITICAL ERROR: Failed to load MuseTalk models or initialize generators: {e}"
        )
        # Set models to None to indicate failure state
        musetalk_batch_generator_instance = None
        shared_models.vae = None
        shared_models.unet = None
        shared_models.pe = None
        shared_models.audio_processor = None
        shared_models.whisper = None
        shared_models.face_parser = None
        shared_models.device = None
        shared_models.device_str = None
        # Allow the application to start, but endpoints should check for None models.

    yield  # Application runs after this

    # --- Application shutdown ---
    print("FastAPI application shutdown...")
    # No stream sessions to clean up explicitly now.
    # Python's GC will handle model unloading on process exit.


app = FastAPI(lifespan=lifespan)


# --- Pydantic Models for WebSocket Input/Output (Batch Only) ---
class LipSyncBatchInput(BaseModel):
    audio_base64: str
    image_base64: str


# --- HTML for basic testing (optional, for simple browser client) ---
# Updated HTML to only show the batch mode section
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Batch LipSync WebSocket Test</title>
        <style>
            body { font-family: sans-serif; }
            #batchMessages { border: 1px solid #ccc; padding: 10px; height: 300px; overflow-y: scroll; background: #f9f9f9; }
            #batchVideoOutput { max-width: 100%; height: auto; border: 1px solid #ddd; margin-top: 10px; }
            .section { margin-bottom: 20px; padding: 15px; border: 1px solid #eee; border-radius: 5px; background: #fff;}
            h2 { margin-top: 0; }
            textarea, input[type="text"] { width: calc(100% - 20px); padding: 8px; margin-bottom: 10px; border: 1px solid #ccc; border-radius: 4px; }
            button { padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <h1>Batch LipSync WebSocket API Test</h1>

        <div class="section">
            <h2>Batch Mode (/ws/lipsync/batch)</h2>
            <p>Paste Base64 of a person image:</p>
            <textarea id="batchImageBase64" rows="5" placeholder="Base64 image string (e.g., from small PNG/JPG)"></textarea>
            <p>Paste Base64 of an audio file:</p>
            <textarea id="batchAudioBase64" rows="5" placeholder="Base64 audio string (e.g., from short WAV)"></textarea>
            <button onclick="sendBatchMessage()">Send Batch Data</button>
            <h3>Batch Response:</h3>
            <div id="batchMessages" style="height: 100px; overflow-y: scroll; border: 1px solid #eee;"></div>
            <h3>Batch Video Output:</h3>
            <video id="batchVideoOutput" controls autoplay></video>
        </div>

        <script>
            let wsBatch;

            function connectBatchWs() {
                if (wsBatch && wsBatch.readyState === WebSocket.OPEN) return;
                wsBatch = new WebSocket("ws://localhost:8000/ws/lipsync/batch");
                wsBatch.onopen = () => console.log("Batch WS Connected");
                wsBatch.onmessage = function(event) {
                    const messagesDiv = document.getElementById('batchMessages');
                    const videoOutput = document.getElementById('batchVideoOutput');
                    messagesDiv.innerHTML += `<p>Received: ${event.data.substring(0, 200)}...</p>`;
                    try {
                        const data = JSON.parse(event.data);
                        if (data.status === "success" && data.video_base64) {
                             messagesDiv.innerHTML += `<p style="color: green;">Success! Video data received.</p>`;
                            videoOutput.src = `data:video/mp4;base64,${data.video_base64}`;
                            videoOutput.load();
                        } else {
                            messagesDiv.innerHTML += `<p style="color: red;">Error: ${data.message || 'Unknown error'}</p>`;
                        }
                    } catch (e) {
                        messagesDiv.innerHTML += `<p style="color: red;">Failed to parse JSON: ${e.message}</p>`;
                    }
                     messagesDiv.scrollTop = messagesDiv.scrollHeight; // Auto-scroll
                };
                wsBatch.onclose = () => console.log("Batch WS Disconnected");
                wsBatch.onerror = (error) => console.error("Batch WS Error:", error);
            }

            // Connect on page load
            window.onload = () => {
                connectBatchWs();
            };

            function sendBatchMessage() {
                connectBatchWs(); // Ensure connected
                if (wsBatch.readyState !== WebSocket.OPEN) {
                    alert("Batch WebSocket not open. Try again in a moment.");
                    return;
                }
                const imageB64 = document.getElementById("batchImageBase64").value;
                const audioB64 = document.getElementById("batchAudioBase64").value;
                if (!imageB64 || !audioB64) {
                    alert("Please provide both image and audio base64 strings for batch mode.");
                    return;
                }
                const payload = JSON.stringify({ "image_base64": imageB64, "audio_base64": audioB64 });
                wsBatch.send(payload);
                 document.getElementById('batchMessages').innerHTML += `<p>Sending data...</p>`;
            }
        </script>
    </body>
</html>
"""


@app.get("/")
async def get_test_page():
    return HTMLResponse(html)


# --- Batch Mode Endpoint ---
@app.websocket("/ws/lipsync/batch")
async def websocket_lipsync_batch_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to batch lipsync endpoint.")
    request_temp_dir = None  # Initialize outside try for cleanup
    try:
        # In batch mode, a client connects, sends one request, gets one response, and can disconnect.
        # So, we expect one message per connection, or handle multiple if the client sends them.
        # The current structure will process one message and then the loop will wait for the next.
        # If you expect only one request per connection, add `break` after sending response.
        # For simplicity now, let's keep the loop but expect one message per interaction.

        # While True: # Kept the loop to handle multiple requests per connection if needed
        if (
            musetalk_batch_generator_instance is None or shared_models.vae is None
        ):  # Check if models failed to load
            print("Batch generator or shared models not initialized. Sending error.")
            await websocket.send_json(
                {
                    "status": "error",
                    "message": "Server models not loaded. Please check server logs for critical errors during startup.",
                }
            )
            # Consider closing the connection after sending the error
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            return  # Exit the handler for this connection

        data_str = await websocket.receive_text()
        print(f"Batch: Received raw data (first 100 chars): {data_str[:100]}...")

        try:
            input_data = LipSyncBatchInput.model_validate_json(
                data_str
            )  # Use model_validate_json for Pydantic V2+
            image_bytes = base64.b64decode(input_data.image_base64)
            audio_bytes = base64.b64decode(input_data.audio_base64)
            print("Batch: Successfully decoded image and audio base64.")

            # Create a unique temporary directory for this batch request
            request_temp_dir = os.path.join(TEMP_BASE_DIR, "batch_" + str(uuid.uuid4()))
            os.makedirs(request_temp_dir, exist_ok=True)
            print(f"Batch: Created temporary directory: {request_temp_dir}")

            # Save input files temporarily
            temp_image_path = os.path.join(
                request_temp_dir, "input_image.png"
            )  # Or jpg, MuseTalk handles various image types
            temp_audio_path = os.path.join(
                request_temp_dir, "input_audio.wav"
            )  # MuseTalk needs audio file

            # MuseTalk's AudioProcessor uses librosa and expects formats like .wav, .mp3, etc.
            # The input audio base64 could be from any format. Saving it as .wav is safest
            # if MuseTalk's processor prefers it or if further audio processing is needed.
            # Assuming the input audio base64 is of a common format compatible with pydub/ffmpeg implicitly used by MuseTalk.
            # Let's save it as .wav, but this might require format detection or client to send format.
            # For simplicity now, assuming common audio bytes format savable as .wav.
            with open(temp_image_path, "wb") as f_img:
                f_img.write(image_bytes)
            with open(temp_audio_path, "wb") as f_aud:
                f_aud.write(audio_bytes)
            print(f"Batch: Saved temp inputs: {temp_image_path}, {temp_audio_path}")

            try:
                print("Batch: Starting MuseTalk generation...")
                # The generate method in the adapter returns the path to the final video
                generated_video_path = await asyncio.to_thread(
                    musetalk_batch_generator_instance.generate,
                    image_path=temp_image_path,
                    audio_path=temp_audio_path,
                    result_dir=os.path.join(
                        request_temp_dir, "result"
                    ),  # Output sub-directory within the request temp dir
                )
                print(
                    f"Batch: MuseTalk generation complete. Video at: {generated_video_path}"
                )

                if not os.path.exists(generated_video_path):
                    raise FileNotFoundError(
                        f"Generated video file not found after MuseTalk processing at: {generated_video_path}"
                    )

                # Read the generated video file and encode it to base64
                with open(generated_video_path, "rb") as video_file:
                    video_base64 = base64.b64encode(video_file.read()).decode("utf-8")

                response_data = {
                    "status": "success",
                    "message": "Video generated successfully.",
                    "video_base64": video_base64,
                }
                # Send the response back to the client
                await websocket.send_json(response_data)
                print("Batch: Sent video data to client.")

            except (FileNotFoundError, RuntimeError, Exception) as gen_error:
                # Catch potential errors during generation or file handling
                print(f"Batch: Generation or file error: {gen_error}")
                await websocket.send_json(
                    {"status": "error", "message": f"Generation failed: {gen_error}"}
                )

        except ValidationError as e:
            print(f"Batch: Input validation error: {e}")
            await websocket.send_json(
                {"status": "error", "message": f"Invalid input format: {e.errors()}"}
            )
        except base64.binascii.Error as e_b64:
            print(f"Batch: Base64 decoding error: {e_b64}")
            await websocket.send_json(
                {"status": "error", "message": f"Invalid base64 string: {e_b64}"}
            )
        except json.JSONDecodeError as e_json:
            print(f"Batch: JSON decoding error: {e_json}")
            await websocket.send_json(
                {"status": "error", "message": f"Invalid JSON received: {e_json}"}
            )
        except Exception as e_parse:
            # Catch any other unexpected errors during message processing
            print(f"Batch: Unexpected error processing message: {e_parse}")
            await websocket.send_json(
                {
                    "status": "error",
                    "message": f"An unexpected server error occurred: {str(e_parse)}",
                }
            )
        finally:
            # Clean up the temporary directory for this request
            if request_temp_dir and os.path.exists(request_temp_dir):
                try:
                    shutil.rmtree(request_temp_dir)
                    print(f"Batch: Cleaned up temporary directory: {request_temp_dir}")
                except Exception as e_clean:
                    print(
                        f"Batch: Error cleaning up temp directory {request_temp_dir}: {e_clean}"
                    )

    except WebSocketDisconnect:
        print("Batch: Client disconnected from lipsync endpoint.")
        # Ensure cleanup even if client disconnects unexpectedly
        if request_temp_dir and os.path.exists(request_temp_dir):
            try:
                shutil.rmtree(request_temp_dir)
                print(
                    f"Batch: Cleaned up temporary directory {request_temp_dir} on disconnect."
                )
            except Exception as e_clean:
                print(
                    f"Batch: Error cleaning up temp directory {request_temp_dir} on disconnect: {e_clean}"
                )
    except Exception as e:
        print(
            f"Batch: An unexpected error occurred in websocket_lipsync_batch_endpoint: {e}"
        )
        # Attempt to send a final error message before closing
        try:
            if websocket.application_state == websockets.WebSocketState.OPEN:
                await websocket.send_json(
                    {
                        "status": "error",
                        "message": "An unexpected server error occurred.",
                    }
                )
        except Exception as send_err:
            print(f"Batch: Failed to send error on broken socket: {send_err}")


if __name__ == "__main__":
    print("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
