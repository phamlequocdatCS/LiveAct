# lipsync_project/MuseTalk/musetalk_adapter_realtime.py

import os
import sys
import cv2
import torch
import numpy as np
import copy
import shutil
import io # For in-memory audio/image handling
from pydub import AudioSegment # For robust audio processing with chunks

# Ensure the MuseTalk root directory is in sys.path
MUSE_TALK_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Should be /path/to/MuseTalk
if MUSE_TALK_ROOT_DIR not in sys.path:
    sys.path.insert(0, MUSE_TALK_ROOT_DIR)

# Assume all internal MuseTalk library imports are now fixed to use absolute paths
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model

# FIX: Explicitly import functions needed from preprocessing
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder

from transformers import WhisperModel

class RealtimeMuseTalkProcessor:
    def __init__(self, args_config, preloaded_models, image_bytes, session_id, temp_base_dir, device="cuda"):
        print(f"Initializing RealtimeMuseTalkProcessor for session {session_id} on device: {device}")
        self.args = args_config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Models from preloaded_models (loaded once at app startup)
        self.vae = preloaded_models.vae
        self.unet = preloaded_models.unet
        self.pe = preloaded_models.pe
        self.audio_processor = preloaded_models.audio_processor
        self.whisper = preloaded_models.whisper
        self.fp = preloaded_models.face_parser # FaceParsing instance
        self.timesteps = torch.tensor([0], device=self.device)

        # Ensure models are on the correct device
        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)

        self.session_id = session_id
        self.temp_session_dir = os.path.join(temp_base_dir, f"realtime_{session_id}")
        os.makedirs(self.temp_session_dir, exist_ok=True)
        print(f"Session temp directory created: {self.temp_session_dir}")

        # Avatar materials, initialized once per session
        self.base_frame_np = None # Original RGB frame of the avatar image
        self.base_bbox = None     # Bounding box for the face
        self.base_latent = None   # Latent representation of the cropped face
        self.base_mask = None     # Face parsing mask (BGR, from get_image_prepare_material)
        self.base_mask_crop_box = None # Bounding box for the mask

        # Audio buffer for accumulating incoming chunks
        self.full_audio_segment = AudioSegment.empty()
        # Path to the temporary WAV file for accumulated audio
        self.temp_full_audio_path = os.path.join(self.temp_session_dir, "full_audio.wav")
        self.processed_frames_count = 0 # To track how many frames have been generated and sent

        self._prepare_avatar_material(image_bytes)
        print(f"RealtimeMuseTalkProcessor for session {session_id} initialized successfully.")

    def _prepare_avatar_material(self, image_bytes):
        """
        Prepares the avatar material (landmarks, bbox, latent, mask) from the initial image.
        This is done once per session during initialization.
        self.base_frame_np will be stored as RGB.
        self.base_mask will be stored as BGR (output of get_image_prepare_material).
        """
        print("Preparing avatar material from input image...")
        
        input_image_np_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if input_image_np_bgr is None:
            raise ValueError("Could not decode input image bytes. Ensure it's a valid image.")
        
        input_image_np_rgb = cv2.cvtColor(input_image_np_bgr, cv2.COLOR_BGR2RGB)

        temp_input_image_path = os.path.join(self.temp_session_dir, "input_avatar.png")
        cv2.imwrite(temp_input_image_path, input_image_np_bgr) 

        bbox_shift_val = 0 if self.args.version == "v15" else self.args.bbox_shift
        
        coord_list, frame_list = get_landmark_and_bbox([temp_input_image_path], bbox_shift_val)

        if not coord_list or coord_list[0] == coord_placeholder:
            raise ValueError(f"No face detected or landmarks returned for avatar image. "
                             f"Please provide an image with a clear, visible face. Coords: {coord_list}")
        
        self.base_frame_np = frame_list[0] # This is RGB
        self.base_bbox = coord_list[0]     

        x1, y1, x2, y2 = self.base_bbox
        
        if self.args.version == "v15":
            y2_adjusted = y2 + self.args.extra_margin
            y2_adjusted = min(y2_adjusted, self.base_frame_np.shape[0])
            self.base_bbox = [x1, y1, x2, y2_adjusted] 
        else:
            y2_adjusted = y2 

        crop_frame_rgb = self.base_frame_np[y1:y2_adjusted, x1:x2] # crop_frame is RGB
        resized_crop_frame_rgb = cv2.resize(crop_frame_rgb, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        
        self.base_latent = self.vae.get_latents_for_unet(resized_crop_frame_rgb) # VAE expects RGB
        
        if torch.isnan(self.base_latent).any() or torch.isinf(self.base_latent).any():
            print(f"Warning: NaN or Inf detected in base_latent during avatar preparation for session {self.session_id}. Replacing with zeros.")
            self.base_latent = torch.nan_to_num(self.base_latent, nan=0.0, posinf=0.0, neginf=0.0)

        mode = self.args.parsing_mode if self.args.version == "v15" else "raw"
        # get_image_prepare_material expects an RGB frame, but returns a BGR mask
        self.base_mask, self.base_mask_crop_box = get_image_prepare_material(
            self.base_frame_np, # This is RGB
            [x1, y1, x2, y2_adjusted], 
            fp=self.fp, mode=mode
        ) # self.base_mask is BGR
        print("Avatar material prepared successfully.")

    @torch.no_grad()
    def process_audio_chunk(self, audio_chunk_bytes: bytes) -> list[np.ndarray]:
        """
        Processes an incoming audio chunk, accumulates it, and returns newly generated video frames
        corresponding to the *new* audio added since the last call.
        The returned frames are NumPy arrays in BGR format.
        """
        
        try:
            new_audio_segment = AudioSegment.from_file(io.BytesIO(audio_chunk_bytes), format="wav")
            self.full_audio_segment += new_audio_segment
        except Exception as e:
            print(f"Error appending audio chunk for session {self.session_id}: {e}. "
                  f"Ensure audio chunks are valid WAV segments. Skipping this chunk.")
            return [] 

        self.full_audio_segment.export(self.temp_full_audio_path, format="wav")

        whisper_input_features_list, librosa_length = self.audio_processor.get_audio_feature(
            self.temp_full_audio_path, weight_dtype=self.unet.model.dtype
        )

        if not whisper_input_features_list:
            print(f"Warning: No Whisper input features generated for session {self.session_id}. Skipping chunk processing.")
            return []

        sanitized_whisper_input_features_list = []
        for i, input_feature_segment in enumerate(whisper_input_features_list):
            if not isinstance(input_feature_segment, torch.Tensor):
                print(f"Error: Expected torch.Tensor but got {type(input_feature_segment)} at index {i} in whisper_input_features_list. Attempting conversion.")
                try:
                    input_feature_segment = torch.tensor(input_feature_segment, dtype=self.unet.model.dtype, device=self.device)
                except Exception as convert_e:
                    print(f"Failed to convert non-Tensor segment: {convert_e}. This segment will be skipped.")
                    continue 

            if torch.isnan(input_feature_segment).any() or torch.isinf(input_feature_segment).any():
                print(f"Warning: NaN or Inf detected in input_feature_segment (index {i}) for session {self.session_id}. Replacing with zeros.")
                input_feature_segment = torch.nan_to_num(input_feature_segment, nan=0.0, posinf=0.0, neginf=0.0)
            
            sanitized_whisper_input_features_list.append(input_feature_segment)
        
        if not sanitized_whisper_input_features_list:
            print(f"Warning: All Whisper input feature segments were problematic after sanitization for session {self.session_id}. Skipping chunk processing.")
            return []

        all_whisper_chunks_tensor = self.audio_processor.get_whisper_chunk(
            sanitized_whisper_input_features_list, 
            self.device,
            self.unet.model.dtype, 
            self.whisper,
            librosa_length,
            fps=self.args.fps, 
            audio_padding_length_left=self.args.audio_padding_length_left,
            audio_padding_length_right=self.args.audio_padding_length_right,
        )

        if not isinstance(all_whisper_chunks_tensor, torch.Tensor):
             raise TypeError(f"Critical Error: get_whisper_chunk returned {type(all_whisper_chunks_tensor)}, expected torch.Tensor. Please check audio_processor.py.")

        if torch.isnan(all_whisper_chunks_tensor).any() or torch.isinf(all_whisper_chunks_tensor).any():
            print(f"Warning: NaN or Inf detected in consolidated all_whisper_chunks_tensor after get_whisper_chunk. Replacing with zeros.")
            all_whisper_chunks_tensor = torch.nan_to_num(all_whisper_chunks_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        new_prompts_tensor_to_process = all_whisper_chunks_tensor[self.processed_frames_count:]
        
        if new_prompts_tensor_to_process.shape[0] == 0: 
            return [] 

        generated_frames_np_bgr = [] # Will store BGR frames
        
        list_of_new_prompts_for_datagen = [new_prompts_tensor_to_process[j] for j in range(new_prompts_tensor_to_process.shape[0])]

        gen = datagen(
            whisper_chunks=list_of_new_prompts_for_datagen, 
            vae_encode_latents=[self.base_latent], 
            batch_size=self.args.batch_size,
            delay_frame=0,
            device=self.device,
        )

        for i, (whisper_batch, latent_batch) in enumerate(gen):
            whisper_batch = whisper_batch.to(self.device, dtype=self.unet.model.dtype)

            if torch.isnan(whisper_batch).any() or torch.isinf(whisper_batch).any():
                print(f"Warning: NaN or Inf detected in whisper_batch (before PE) for session {self.session_id}. Replacing with zeros.")
                whisper_batch = torch.nan_to_num(whisper_batch, nan=0.0, posinf=0.0, neginf=0.0)

            audio_feature_batch = self.pe(whisper_batch) 

            if torch.isnan(audio_feature_batch).any() or torch.isinf(audio_feature_batch).any():
                print(f"Warning: NaN or Inf detected in audio_feature_batch (after PE) for session {self.session_id}. Replacing with zeros.")
                audio_feature_batch = torch.nan_to_num(audio_feature_batch, nan=0.0, posinf=0.0, neginf=0.0)

            latent_batch = latent_batch.to(self.device, dtype=self.unet.model.dtype)

            pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
            
            if torch.isnan(pred_latents).any() or torch.isinf(pred_latents).any():
                print(f"Warning: NaN or Inf detected in pred_latents for session {self.session_id}. Replacing with zeros.")
                pred_latents = torch.nan_to_num(pred_latents, nan=0.0, posinf=0.0, neginf=0.0)

            recon_frames_rgb = self.vae.decode_latents(pred_latents.to(self.device, dtype=self.vae.vae.dtype)) # VAE returns RGB

            for res_frame_np_rgb in recon_frames_rgb: # res_frame_np_rgb is RGB
                x1, y1, x2, y2 = self.base_bbox
                try:
                    resized_res_frame_rgb = cv2.resize(res_frame_np_rgb.astype(np.uint8), (x2 - x1, y2 - y1)) # resized_res_frame_rgb is RGB
                except Exception as e_resize:
                    print(f"Warning: Could not resize generated frame. Error: {e_resize}. Skipping blending for this frame.")
                    continue
                
                # get_image_blending expects RGB original frame, RGB generated frame,
                # and BGR mask. It returns a BGR combined frame.
                combined_frame_bgr = get_image_blending(
                    self.base_frame_np,          # This is RGB
                    resized_res_frame_rgb,       # This is RGB
                    self.base_bbox,              
                    self.base_mask,              # This is BGR
                    self.base_mask_crop_box      
                ) # combined_frame_bgr is BGR
                generated_frames_np_bgr.append(combined_frame_bgr) # Store BGR frames

        self.processed_frames_count += len(generated_frames_np_bgr)
        
        return generated_frames_np_bgr # Return list of BGR frames

    def cleanup(self):
        """Removes temporary files and directories created for this session."""
        print(f"Cleaning up temporary directory for session {self.session_id}: {self.temp_session_dir}")
        if os.path.exists(self.temp_session_dir):
            try:
                shutil.rmtree(self.temp_session_dir)
                print(f"Successfully cleaned up {self.temp_session_dir}")
            except Exception as e:
                print(f"Error during cleanup of session {self.session_id} temp directory: {e}")