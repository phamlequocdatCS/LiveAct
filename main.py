# lipsync_project/main.py

import asyncio
import base64
from contextlib import asynccontextmanager
import json
import uvicorn
import os
import uuid
import sys
import shutil
from types import SimpleNamespace # To create simple config objects
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError
import websockets

# --- Add MuseTalk directory to sys.path ---
LIPSYNC_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MUSE_TALK_DIR = os.path.join(LIPSYNC_PROJECT_ROOT, "MuseTalk")
if MUSE_TALK_DIR not in sys.path:
    sys.path.append(MUSE_TALK_DIR)

# Import the new adapter classes
try:
    from MuseTalk.musetalk_adapter_batch import MuseTalkBatchGenerator
    from MuseTalk.musetalk_adapter_realtime import RealtimeMuseTalkProcessor

    # MuseTalk's core model loading and utility functions
    from musetalk.utils.utils import load_all_model
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.face_parsing import FaceParsing
    from transformers import WhisperModel
    import torch # For device checks
    import cv2 # For JPEG encoding of frames
    from PIL import Image # For image handling (though not directly used in latest inference)
    
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import MuseTalk adapters or core modules: {e}")
    print(f"Please ensure all patches are applied and the MuseTalk/ directory structure is correct.")
    # Exit or handle gracefully in production
    sys.exit(1)


# Directory for temporary files for all operations
TEMP_BASE_DIR = os.path.join(LIPSYNC_PROJECT_ROOT, "temp_files")
os.makedirs(TEMP_BASE_DIR, exist_ok=True)

# Global variables for shared models and active streaming sessions
shared_models: SimpleNamespace = SimpleNamespace()
batch_args_config: SimpleNamespace = SimpleNamespace()
realtime_args_config: SimpleNamespace = SimpleNamespace()
musetalk_batch_generator_instance: MuseTalkBatchGenerator = None
active_stream_sessions: Dict[str, RealtimeMuseTalkProcessor] = {} # {session_id: processor_instance}


# --- FastAPI Lifespan Events for Model Loading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global musetalk_batch_generator_instance, shared_models, batch_args_config, realtime_args_config

    print("FastAPI application startup...")
    try:
        # --- 1. Define common configuration for MuseTalk models ---
        common_cfg = SimpleNamespace()
        common_cfg.gpu_id = 0 # Matches inference.py default
        common_cfg.vae_type = "sd-vae" # Matches inference.py default
        common_cfg.unet_model_path = os.path.join("models", "musetalkV15", "unet.pth")
        common_cfg.unet_config = os.path.join("models", "musetalkV15", "musetalk.json")
        common_cfg.whisper_dir = os.path.join("models", "whisper") # Matches inference.py default
        common_cfg.use_float16 = False # Set to True for faster inference, matches app.py default

        # Determine device and store it in shared_models for global access
        shared_models.device_str = f"cuda:{common_cfg.gpu_id}" if torch.cuda.is_available() else "cpu"
        shared_models.device = torch.device(shared_models.device_str)

        # Check critical file paths relative to MUSE_TALK_DIR
        abs_unet_model_path = os.path.join(MUSE_TALK_DIR, common_cfg.unet_model_path)
        abs_unet_config = os.path.join(MUSE_TALK_DIR, common_cfg.unet_config)
        # Note: vae_type like "sd-vae" maps to a directory name, config is inside.
        abs_vae_config_path = os.path.join(MUSE_TALK_DIR, "models", common_cfg.vae_type, "config.json")
        abs_whisper_dir = os.path.join(MUSE_TALK_DIR, common_cfg.whisper_dir)
        abs_face_parse_model_pth = os.path.join(MUSE_TALK_DIR, 'models', 'face-parse-bisent', '79999_iter.pth')
        abs_face_parse_resnet_path = os.path.join(MUSE_TALK_DIR, 'models', 'face-parse-bisent', 'resnet18-5c106cde.pth')

        for path_to_check in [abs_unet_model_path, abs_unet_config, abs_vae_config_path, abs_whisper_dir, abs_face_parse_model_pth, abs_face_parse_resnet_path]:
            if not os.path.exists(path_to_check):
                raise FileNotFoundError(f"MuseTalk critical file/dir not found at application startup: {path_to_check}")

        # --- 2. Load common models once ---
        print("Loading shared MuseTalk models (VAE, UNet, PE, Whisper, FaceParser)...")
        shared_models.vae, shared_models.unet, shared_models.pe = load_all_model(
            unet_model_path=abs_unet_model_path, # Use absolute path here
            vae_type=common_cfg.vae_type,
            unet_config=abs_unet_config, # Use absolute path here
            device=shared_models.device_str, # Use shared_models.device_str
            use_float16=common_cfg.use_float16
        )
        
        # Determine and store the weight_dtype
        shared_models.weight_dtype = torch.float16 if common_cfg.use_float16 else torch.float32

        # Apply precision conversion for PE and UNet if use_float16 is true
        if common_cfg.use_float16:
            print("Applying float16 precision to PE and UNet...")
            shared_models.pe = shared_models.pe.half()
            shared_models.unet.model = shared_models.unet.model.half()
            print("PE and UNet converted to float16.")
        else:
            print("Running PE and UNet in float32 precision.")

        # Move to device (ensure they are on correct device after half() conversion)
        shared_models.pe = shared_models.pe.to(shared_models.device)
        shared_models.vae.vae = shared_models.vae.vae.to(shared_models.device)
        shared_models.unet.model = shared_models.unet.model.to(shared_models.device)


        shared_models.audio_processor = AudioProcessor(
            feature_extractor_path=abs_whisper_dir # Use absolute path
        )
        # Initialize WhisperModel using the path to the directory
        shared_models.whisper = WhisperModel.from_pretrained(
            abs_whisper_dir # Use absolute path
        )
        shared_models.whisper = shared_models.whisper.to(
            device=shared_models.device, # Use shared_models.device
            dtype=shared_models.weight_dtype # Use the stored weight_dtype
        ).eval().requires_grad_(False)

        # FaceParsing (already patched to handle absolute paths internally)
        shared_models.face_parser = FaceParsing(
            left_cheek_width=90, # Default, can be exposed in config if needed
            right_cheek_width=90
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
        batch_args_config.parsing_mode = 'jaw'
        
        musetalk_batch_generator_instance = MuseTalkBatchGenerator(
            args_config=batch_args_config,
            preloaded_models=shared_models,
            device=shared_models.device_str # Use shared_models.device_str
        )
        print("Batch Mode Generator initialized.")

        # --- 4. Configure Real-Time Mode Parameters ---
        realtime_args_config.fps = 25
        realtime_args_config.audio_padding_length_left = 2
        realtime_args_config.audio_padding_length_right = 2
        realtime_args_config.batch_size = 1 # Keep this for realtime, process one audio chunk at a time
        realtime_args_config.version = "v15"
        realtime_args_config.bbox_shift = 0
        realtime_args_config.extra_margin = 10
        realtime_args_config.parsing_mode = 'jaw'

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load MuseTalk models or initialize generators: {e}")
        musetalk_batch_generator_instance = None
        shared_models.vae = None # Set components to None to indicate failure
        shared_models.unet = None
        shared_models.pe = None
        shared_models.audio_processor = None
        shared_models.whisper = None
        shared_models.face_parser = None
        shared_models.device = None
        shared_models.device_str = None
        # Do NOT sys.exit() here, let the app start but fail on requests
        # This allows you to inspect the app state.

    yield # Application runs after this

    # --- Application shutdown ---
    print("FastAPI application shutdown...")
    # Make a copy of items for safe iteration if sessions can be modified during cleanup
    for session_id, processor in list(active_stream_sessions.items()):
        if processor: # Check if processor is not None (it might have been popped already)
            await asyncio.to_thread(processor.cleanup)
    active_stream_sessions.clear() # Clear the dictionary
    # No explicit cleanup for shared_models needed, Python's GC will handle it on exit


app = FastAPI(lifespan=lifespan)

# --- Pydantic Models for WebSocket Input/Output ---
class LipSyncBatchInput(BaseModel):
    audio_base64: str
    image_base64: str

class LipSyncStreamInitInput(BaseModel):
    type: str = "init"
    image_base64: str
    desired_fps: int = 25 # Client can specify desired FPS, default 25

class LipSyncStreamAudioChunkInput(BaseModel):
    type: str = "audio_chunk"
    session_id: str
    audio_base64: str

class LipSyncStreamEndInput(BaseModel):
    type: str = "end_stream"
    session_id: str

class LipSyncStreamOutput(BaseModel):
    type: str
    session_id: str
    data: Any # Can be list of base64 frames, or video chunk, or status message


# --- HTML for basic testing (optional, for simple browser client) ---
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>LipSync WebSocket Test</title>
        <style>
            body { font-family: sans-serif; }
            #messages { border: 1px solid #ccc; padding: 10px; height: 300px; overflow-y: scroll; background: #f9f9f9; }
            #videoOutput { max-width: 100%; height: auto; border: 1px solid #ddd; margin-top: 10px; }
            .section { margin-bottom: 20px; padding: 15px; border: 1px solid #eee; border-radius: 5px; background: #fff;}
            h2 { margin-top: 0; }
            textarea, input[type="text"] { width: calc(100% - 20px); padding: 8px; margin-bottom: 10px; border: 1px solid #ccc; border-radius: 4px; }
            button { padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <h1>LipSync WebSocket API Test</h1>

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

        <div class="section">
            <h2>Streaming Mode (/ws/lipsync/stream)</h2>
            <p>Streaming is more complex and best tested with a dedicated client script.</p>
            <p>This demo only shows the init message response.</p>
            <p>Paste Base64 of a person image for stream init:</p>
            <textarea id="streamImageBase66" rows="5" placeholder="Base64 image string (e.g., from small PNG/JPG)"></textarea>
            <button onclick="sendStreamInit()">Send Stream Init</button>
            <h3>Stream Response:</h3>
            <div id="streamMessages" style="height: 100px; overflow-y: scroll; border: 1px solid #eee;"></div>
            <p>After init, you'd send audio chunks.</p>
        </div>

        <script>
            let wsBatch;
            let wsStream;

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
                            videoOutput.src = `data:video/mp4;base64,${data.video_base64}`;
                            videoOutput.load();
                        } else {
                            messagesDiv.innerHTML += `<p style="color: red;">Error: ${data.message || 'Unknown error'}</p>`;
                        }
                    } catch (e) {
                        messagesDiv.innerHTML += `<p style="color: red;">Failed to parse JSON: ${e.message}</p>`;
                    }
                };
                wsBatch.onclose = () => console.log("Batch WS Disconnected");
                wsBatch.onerror = (error) => console.error("Batch WS Error:", error);
            }

            function connectStreamWs() {
                if (wsStream && wsStream.readyState === WebSocket.OPEN) return;
                wsStream = new WebSocket("ws://localhost:8000/ws/lipsync/stream");
                wsStream.onopen = () => console.log("Stream WS Connected");
                wsStream.onmessage = function(event) {
                    const messagesDiv = document.getElementById('streamMessages');
                    messagesDiv.innerHTML += `<p>Received: ${event.data.substring(0, 200)}...</p>`;
                    // In a real stream client, you'd process video frames here
                };
                wsStream.onclose = () => console.log("Stream WS Disconnected");
                wsStream.onerror = (error) => console.error("Stream WS Error:", error);
            }

            // Connect both on page load
            window.onload = () => {
                connectBatchWs();
                connectStreamWs();
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
            }

            function sendStreamInit() {
                connectStreamWs(); // Ensure connected
                if (wsStream.readyState !== WebSocket.OPEN) {
                    alert("Stream WebSocket not open. Try again in a moment.");
                    return;
                }
                const imageB64 = document.getElementById("streamImageBase66").value;
                if (!imageB64) {
                    alert("Please provide an image base64 string for stream init.");
                    return;
                }
                const payload = JSON.stringify({ "type": "init", "image_base64": imageB64, "desired_fps": 25 });
                wsStream.send(payload);
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
    try:
        while True:
            if shared_models.vae is None: # Check if models failed to load
                print("Batch generator models not initialized. Sending error.")
                await websocket.send_json({"status": "error", "message": "Server models not loaded. Please check server logs for critical errors during startup."})
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                break

            data_str = await websocket.receive_text()
            print(f"Batch: Received raw data: {data_str[:100]}...")

            try:
                input_data = LipSyncBatchInput.model_validate_json(data_str) # Use model_validate_json for Pydantic V2+
                image_bytes = base64.b64decode(input_data.image_base64)
                audio_bytes = base64.b64decode(input_data.audio_base64)
                print("Batch: Successfully decoded image and audio.")

                # Create a unique temporary directory for this batch request
                request_temp_dir = os.path.join(TEMP_BASE_DIR, "batch_" + str(uuid.uuid4()))
                os.makedirs(request_temp_dir, exist_ok=True)
                
                # Save input files temporarily
                temp_image_path = os.path.join(request_temp_dir, "input_image.png")
                temp_audio_path = os.path.join(request_temp_dir, "input_audio.wav")
                with open(temp_image_path, "wb") as f_img:
                    f_img.write(image_bytes)
                with open(temp_audio_path, "wb") as f_aud:
                    f_aud.write(audio_bytes)
                print(f"Batch: Saved temp image: {temp_image_path}, temp audio: {temp_audio_path}")

                try:
                    print(f"Batch: Starting MuseTalk generation for request {request_temp_dir}...")
                    generated_video_path = await asyncio.to_thread(
                        musetalk_batch_generator_instance.generate,
                        image_path=temp_image_path,
                        audio_path=temp_audio_path,
                        result_dir=os.path.join(request_temp_dir, "result") # Output sub-directory
                    )
                    print(f"Batch: MuseTalk generation complete. Video at: {generated_video_path}")

                    if not os.path.exists(generated_video_path):
                         raise FileNotFoundError(f"Generated video file not found at: {generated_video_path}")

                    with open(generated_video_path, "rb") as video_file:
                        video_base64 = base64.b64encode(video_file.read()).decode("utf-8")

                    response_data = {
                        "status": "success",
                        "message": "Video generated successfully.",
                        "video_base64": video_base64
                    }
                    await websocket.send_json(response_data)
                    print(f"Batch: Sent video data to client for request {request_temp_dir}.")

                except (FileNotFoundError, RuntimeError) as gen_error:
                    print(f"Batch: Generation error: {gen_error}")
                    await websocket.send_json({"status": "error", "message": f"Generation failed: {gen_error}"})
                except Exception as e:
                    print(f"Batch: Unexpected error during MuseTalk generation or video sending: {e}")
                    await websocket.send_json({"status": "error", "message": f"An unexpected server error occurred during generation: {str(e)}"})
                finally:
                    if os.path.exists(request_temp_dir):
                        try:
                            shutil.rmtree(request_temp_dir)
                            print(f"Batch: Cleaned up temporary directory: {request_temp_dir}")
                        except Exception as e_clean:
                            print(f"Batch: Error cleaning up temp directory {request_temp_dir}: {e_clean}")
                            
            except ValidationError as e:
                print(f"Batch: Input validation error: {e}")
                await websocket.send_json({"status": "error", "message": f"Invalid input format: {e.errors()}"})
            except base64.binascii.Error as e_b64:
                print(f"Batch: Base64 decoding error: {e_b64}")
                await websocket.send_json({"status": "error", "message": f"Invalid base64 string: {e_b64}"})
            except Exception as e_parse:
                print(f"Batch: Error processing message: {e_parse}")
                await websocket.send_json({"status": "error", "message": f"An unexpected server error occurred: {str(e_parse)}"})

    except WebSocketDisconnect:
        print("Batch: Client disconnected from lipsync endpoint.")
    except Exception as e:
        print(f"Batch: An unexpected error occurred in websocket_lipsync_batch_endpoint: {e}")
        try:
            await websocket.send_json({"status": "error", "message": "An unexpected server error occurred."})
        except:
            pass # If sending fails, socket is likely closed or broken


# --- Streaming Mode Endpoint ---
@app.websocket("/ws/lipsync/stream")
async def websocket_lipsync_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_session_id: str = None
    print("Stream: Client connected to stream lipsync endpoint.")
    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                message = json.loads(raw_message)
                message_type = message.get("type")
            except json.JSONDecodeError:
                print("Stream: Received non-JSON message.")
                await websocket.send_json({"status": "error", "message": "Invalid JSON format."})
                continue

            if shared_models.vae is None: # Check if models failed to load
                print("Stream: Shared models not loaded. Sending error.")
                await websocket.send_json({"status": "error", "message": "Server models not loaded. Please check server logs for critical errors during startup."})
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                break

            if message_type == "init":
                try:
                    init_data = LipSyncStreamInitInput.model_validate(message)
                    image_bytes = base64.b64decode(init_data.image_base64)
                    
                    client_session_id = str(uuid.uuid4())
                    if client_session_id in active_stream_sessions:
                        print(f"Stream: Duplicate session ID generated: {client_session_id}. Retrying.")
                        client_session_id = str(uuid.uuid4()) # Regenerate if collision (highly unlikely)

                    print(f"Stream: Initializing processor for new session: {client_session_id}")
                    # Initialize RealtimeMuseTalkProcessor for this session
                    processor = RealtimeMuseTalkProcessor(
                        args_config=realtime_args_config,
                        preloaded_models=shared_models,
                        image_bytes=image_bytes,
                        session_id=client_session_id,
                        temp_base_dir=TEMP_BASE_DIR,
                        device=shared_models.device_str # Use shared_models.device_str
                    )
                    active_stream_sessions[client_session_id] = processor
                    await websocket.send_json({"type": "init_response", "status": "success", "session_id": client_session_id, "message": "Processor initialized. Send audio chunks."})
                    print(f"Stream: Processor initialized for session {client_session_id}.")

                except ValidationError as e:
                    print(f"Stream: Init message validation error: {e}")
                    await websocket.send_json({"type": "init_response", "status": "error", "message": f"Invalid init message: {e.errors()}"})
                except Exception as e:
                    print(f"Stream: Error during init processing: {e}")
                    await websocket.send_json({"type": "init_response", "status": "error", "message": f"Server error during initialization: {str(e)}"})

            elif message_type == "audio_chunk":
                try:
                    chunk_data = LipSyncStreamAudioChunkInput.model_validate(message)
                    current_session_id = chunk_data.session_id
                    
                    if current_session_id not in active_stream_sessions:
                        print(f"Stream: Audio chunk for unknown session ID: {current_session_id}")
                        await websocket.send_json({"type": "chunk_response", "status": "error", "session_id": current_session_id, "message": "Session not found. Please re-initialize."})
                        continue

                    processor = active_stream_sessions[current_session_id]
                    audio_chunk_bytes = base64.b64decode(chunk_data.audio_base64)

                    print(f"Stream: Processing audio chunk for session {current_session_id}...")
                    # Run processing in a separate thread to not block the event loop
                    generated_frames_np = await asyncio.to_thread(processor.process_audio_chunk, audio_chunk_bytes)
                    print(f"Stream: Generated {len(generated_frames_np)} frames for session {current_session_id}.")

                    # RealtimeMuseTalkProcessor.process_audio_chunk returns frames in BGR format.
                    # cv2.imencode expects BGR. So, no conversion is needed here.
                    frames_base64 = []
                    for frame_np_bgr_from_adapter in generated_frames_np: # This is BGR
                        # frame_bgr = cv2.cvtColor(frame_np_rgb, cv2.COLOR_RGB2BGR) # REMOVED: This was incorrect
                        _, buffer = cv2.imencode('.jpg', frame_np_bgr_from_adapter, [int(cv2.IMWRITE_JPEG_QUALITY), 85]) # Encode BGR as JPEG
                        frames_base64.append(base64.b64encode(buffer).decode('utf-8'))
                    
                    await websocket.send_json({
                        "type": "video_frames",
                        "session_id": current_session_id,
                        "frames_base64_jpeg": frames_base64,
                        "frame_fps": realtime_args_config.fps # Inform client of expected FPS
                    })
                    print(f"Stream: Sent {len(frames_base64)} frames for session {current_session_id}.")

                except ValidationError as e:
                    print(f"Stream: Audio chunk validation error: {e}")
                    await websocket.send_json({"type": "chunk_response", "status": "error", "session_id": message.get("session_id", "unknown"), "message": f"Invalid audio chunk message: {e.errors()}"})
                except Exception as e:
                    print(f"Stream: Error during audio chunk processing: {e}")
                    await websocket.send_json({"type": "chunk_response", "status": "error", "session_id": message.get("session_id", "unknown"), "message": f"Server error during chunk processing: {str(e)}"})

            elif message_type == "end_stream":
                try:
                    end_data = LipSyncStreamEndInput.model_validate(message)
                    current_session_id = end_data.session_id
                    if current_session_id in active_stream_sessions:
                        processor_to_remove = active_stream_sessions.pop(current_session_id, None) # Remove from active sessions, default to None if not found
                        if processor_to_remove:
                            await asyncio.to_thread(processor_to_remove.cleanup) # Clean up temp files
                        print(f"Stream: Session {current_session_id} ended and cleaned up.")
                        await websocket.send_json({"type": "end_response", "status": "success", "session_id": current_session_id, "message": "Stream ended. Resources released."})
                    else:
                        print(f"Stream: Attempted to end unknown session: {current_session_id}")
                        await websocket.send_json({"type": "end_response", "status": "error", "session_id": current_session_id, "message": "Session not found."})
                except ValidationError as e:
                    print(f"Stream: End stream validation error: {e}")
                    await websocket.send_json({"type": "end_response", "status": "error", "session_id": message.get("session_id", "unknown"), "message": f"Invalid end stream message: {e.errors()}"})
                except Exception as e:
                    print(f"Stream: Error during end stream processing: {e}")
                    await websocket.send_json({"type": "end_response", "status": "error", "session_id": message.get("session_id", "unknown"), "message": f"Server error during stream end: {str(e)}"})

            else:
                print(f"Stream: Unknown message type received: {message_type}")
                await websocket.send_json({"status": "error", "message": f"Unknown message type: {message_type}"})

    except WebSocketDisconnect:
        print(f"Stream: Client disconnected from stream endpoint. Session: {client_session_id}")
        if client_session_id and client_session_id in active_stream_sessions:
            processor_on_disconnect = active_stream_sessions.pop(client_session_id, None)
            if processor_on_disconnect:
                await asyncio.to_thread(processor_on_disconnect.cleanup)
            print(f"Stream: Cleaned up session {client_session_id} due to disconnect.")
    except Exception as e:
        print(f"Stream: An unexpected error occurred in websocket_lipsync_stream_endpoint: {e}")
        try:
            if websocket.application_state == websockets.WebSocketState.OPEN: # Check if socket is still open
                 await websocket.send_json({"status": "error", "message": "An unexpected server error occurred."})
        except Exception as send_err:
            print(f"Stream: Failed to send error on broken socket: {send_err}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)