# lipsync_project/MuseTalk/musetalk_adapter_realtime.py

import os
import sys
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
import copy
import shutil
import uuid # For temporary audio files
import io # To read bytes into PIL Image

# --- Path Setup (similar to other adapters) ---
MUSE_TALK_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Should be /path/to/MuseTalk
if MUSE_TALK_ROOT_DIR not in sys.path:
    sys.path.insert(0, MUSE_TALK_ROOT_DIR)

# Assume all internal MuseTalk library imports are now fixed to use absolute paths
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.face_parsing import FaceParsing # Only for type hinting if needed here
from musetalk.utils.audio_processor import AudioProcessor # Only for type hinting if needed here
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from transformers import WhisperModel # Only for type hinting if needed here
from PIL import Image # Needed for image processing

class RealtimeMuseTalkProcessor:
    def __init__(self, args_config, preloaded_models, image_bytes, session_id, temp_base_dir="temp_files", device="cuda"):
        print(f"Initializing RealtimeMuseTalkProcessor for session {session_id} with device: {device}")
        self.session_id = session_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.args = args_config
        self.temp_dir = os.path.join(temp_base_dir, str(session_id)) # Dedicated temp dir for this session
        os.makedirs(self.temp_dir, exist_ok=True)

        # Use preloaded models
        self.vae = preloaded_models.vae
        self.unet = preloaded_models.unet
        self.pe = preloaded_models.pe
        self.audio_processor = preloaded_models.audio_processor
        self.whisper = preloaded_models.whisper
        self.fp = preloaded_models.face_parser
        self.timesteps = torch.tensor([0], device=self.device)

        # Ensure models are on the correct device
        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)

        # Prepare the single avatar frame
        print(f"Preparing avatar for session {session_id} from input image...")
        self._prepare_single_frame_avatar(image_bytes)
        print(f"Avatar prepared for session {session_id}.")

    def _prepare_single_frame_avatar(self, image_bytes):
        # Save image bytes to a temp file for MuseTalk's `get_landmark_and_bbox`
        temp_image_path = os.path.join(self.temp_dir, f"avatar_input_{self.session_id}.png")
        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)
        
        # Original image as NumPy array for blending
        self.avatar_original_frame_np = cv2.imread(temp_image_path)
        if self.avatar_original_frame_np is None:
            raise ValueError(f"Could not read input image for session {self.session_id}.")

        # Get landmark and bbox
        bbox_shift_val = 0 if self.args.version == "v15" else self.args.bbox_shift
        coord_list, frame_list = get_landmark_and_bbox([temp_image_path], bbox_shift_val)

        if not coord_list or coord_list[0] == coord_placeholder:
            raise ValueError(f"No face detected or landmarks returned for avatar image in session {self.session_id}.")
        
        self.avatar_bbox = coord_list[0]
        x1, y1, x2, y2 = self.avatar_bbox
        
        # Adjust y2 for v15 extra margin
        if self.args.version == "v15":
            y2_adjusted = y2 + self.args.extra_margin
            y2_adjusted = min(y2_adjusted, self.avatar_original_frame_np.shape[0])
            self.avatar_bbox_adjusted = (x1, y1, x2, y2_adjusted)
        else:
            self.avatar_bbox_adjusted = self.avatar_bbox # No adjustment for v1

        # Crop and resize the face for VAE encoding
        crop_frame = self.avatar_original_frame_np[y1:y2_adjusted, x1:x2] # Use adjusted y2 for crop
        crop_frame_resized = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        
        # Get VAE latent for the avatar
        self.avatar_latent = self.vae.get_latents_for_unet(crop_frame_resized)

        # Prepare blending materials (mask and crop box for pasting)
        # Note: realtime_inference.py Avatar class does this for all frames
        # We do it once for the single avatar image.
        if self.args.version == "v15":
            mode = self.args.parsing_mode
        else:
            mode = "raw" # Default for v1
        
        # Use the actual original image and its final bounding box for mask generation
        self.avatar_mask, self.avatar_mask_crop_box = get_image_prepare_material(
            self.avatar_original_frame_np, list(self.avatar_bbox_adjusted), fp=self.fp, mode=mode
        )
        # Make sure mask is 3-channel if blending expects that
        if len(self.avatar_mask.shape) == 2:
            self.avatar_mask = cv2.cvtColor(self.avatar_mask, cv2.COLOR_GRAY2BGR)


    @torch.no_grad()
    def process_audio_chunk(self, audio_chunk_bytes):
        # Save audio chunk bytes to a temporary .wav file
        temp_audio_path = os.path.join(self.temp_dir, f"audio_chunk_{uuid.uuid4()}.wav")
        with open(temp_audio_path, "wb") as f:
            f.write(audio_chunk_bytes)

        # Extract audio features for this chunk
        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(temp_audio_path, weight_dtype=self.unet.model.dtype)
        os.remove(temp_audio_path) # Clean up audio temp file immediately

        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.unet.model.dtype,
            self.whisper,
            librosa_length,
            fps=self.args.fps,
            audio_padding_length_left=self.args.audio_padding_length_left,
            audio_padding_length_right=self.args.audio_padding_length_right,
        )

        if len(whisper_chunks) == 0:
            print(f"Warning: No valid whisper chunks for audio chunk in session {self.session_id}.")
            return [] # Return empty list if no frames can be generated

        # Prepare batched input for UNet (repeat avatar latent for each frame in the chunk)
        latent_batch = torch.cat([self.avatar_latent] * len(whisper_chunks), dim=0) # Repeat latent
        whisper_batch = torch.stack(whisper_chunks) # Stack audio chunks

        audio_feature_batch = self.pe(whisper_batch.to(self.device)) # PE needs to be on device
        latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)

        # UNet inference
        pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample

        # VAE decode
        decoded_face_frames_np = self.vae.decode_latents(pred_latents) # List of numpy arrays (BGR)

        # Blend each decoded face back onto the original avatar frame
        blended_frames = []
        for decoded_face_np in decoded_face_frames_np:
            # Resize generated face to the original bbox size for pasting
            x1, y1, x2, y2 = self.avatar_bbox_adjusted # Use adjusted bbox for consistent paste size
            resized_decoded_face = cv2.resize(decoded_face_np.astype(np.uint8), (x2 - x1, y2 - y1))

            # Blend
            combined_frame = get_image_blending(
                self.avatar_original_frame_np,
                resized_decoded_face,
                list(self.avatar_bbox_adjusted), # ensure list type for blending
                self.avatar_mask,
                self.avatar_mask_crop_box
            )
            # Ensure dimensions are divisible by 2 if needed for subsequent video creation (though we send raw frames)
            # Not strictly needed if sending JPEG, but good habit for general video ops.
            h, w = combined_frame.shape[:2]
            if w % 2 != 0 or h % 2 != 0:
                target_w = (w // 2) * 2
                target_h = (h // 2) * 2
                combined_frame = cv2.resize(combined_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            blended_frames.append(combined_frame)
            
        return blended_frames # List of NumPy arrays (BGR frames)

    def cleanup(self):
        # Clean up temporary directory for this session
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory for session {self.session_id}: {self.temp_dir}")
            except Exception as e:
                print(f"Error cleaning up temp directory {self.temp_dir} for session {self.session_id}: {e}")