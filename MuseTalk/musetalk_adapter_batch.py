# lipsync_project/MuseTalk/musetalk_adapter_batch.py

import os
import sys
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
import copy
import shutil
import glob
import subprocess # Make sure subprocess is imported

# --- Path Setup ---
MUSE_TALK_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Should be /path/to/MuseTalk
if MUSE_TALK_ROOT_DIR not in sys.path:
    sys.path.insert(0, MUSE_TALK_ROOT_DIR)

# Assume all internal MuseTalk library imports are now fixed to use absolute paths
from MuseTalk.musetalk.utils.blending import get_image
from MuseTalk.musetalk.utils.face_parsing import FaceParsing # Only for type hinting if needed here
from MuseTalk.musetalk.utils.audio_processor import AudioProcessor # Only for type hinting if needed here
from MuseTalk.musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from MuseTalk.musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from transformers import WhisperModel # Only for type hinting if needed here

class MuseTalkBatchGenerator: # Renamed class
    def __init__(self, args_config, preloaded_models, device="cuda"): # Accepts preloaded models
        print(f"Initializing MuseTalkBatchGenerator with device: {device}")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.args = args_config

        # Use preloaded models instead of loading them here
        self.vae = preloaded_models.vae
        self.unet = preloaded_models.unet
        self.pe = preloaded_models.pe
        self.audio_processor = preloaded_models.audio_processor
        self.whisper = preloaded_models.whisper
        self.fp = preloaded_models.face_parser # FaceParsing instance
        self.timesteps = torch.tensor([0], device=self.device)

        # Ensure models are on the correct device if preloaded_models don't guarantee it
        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)

        print("MuseTalkBatchGenerator initialized successfully with preloaded models.")

    @torch.no_grad()
    def generate(self, image_path, audio_path, result_dir): # result_dir is now the consistent name
        args = self.args 

        input_basename = os.path.splitext(os.path.basename(image_path))[0]
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_basename = f"{input_basename}_{audio_basename}"

        result_img_save_path = os.path.join(result_dir, output_basename + "_frames")
        os.makedirs(result_img_save_path, exist_ok=True)

        final_video_path = os.path.join(result_dir, output_basename + ".mp4")

        if get_file_type(image_path) != "image":
            raise ValueError(f"Input video_path '{image_path}' is not a valid image file.")
        input_img_list = [image_path]
        fps = args.fps

        print(f"Extracting audio features for: {audio_path}")
        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path, weight_dtype=self.unet.model.dtype)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.unet.model.dtype, # Use unet's dtype for consistency
            self.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        print(f"Audio features extracted, num_chunks: {len(whisper_chunks)}")

        print(f"Extracting landmarks for image: {image_path}")
        bbox_shift_val = 0 if args.version == "v15" else args.bbox_shift
        
        coord_list, frame_list = get_landmark_and_bbox([image_path], bbox_shift_val)

        if not coord_list or coord_list[0] == coord_placeholder:
            raise ValueError(f"No face detected or landmarks returned for image: {image_path}. Check image clarity and face visibility.")
        print(f"Landmarks extracted. Coords: {coord_list[0]}")

        input_latent_list = []
        bbox, frame = coord_list[0], frame_list[0]
        x1, y1, x2, y2 = bbox
        if args.version == "v15":
            y2 = y2 + args.extra_margin
            y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
        latents = self.vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)
        print("Input image processed to latent.")

        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

        print("Starting batch inference...")
        video_num = len(whisper_chunks)
        batch_size = args.batch_size
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list_cycle,
            batch_size=batch_size,
            delay_frame=0,
            device=self.device,
        )

        res_frame_list = []

        for i, (whisper_batch, latent_batch) in enumerate(gen):
            audio_feature_batch = self.pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

            pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = self.vae.decode_latents(pred_latents)
            for res_frame_np in recon:
                res_frame_list.append(res_frame_np)
        print(f"Inference complete. Generated {len(res_frame_list)} frames.")

        print("Processing and saving generated frames...")
        for i, res_frame_np in enumerate(res_frame_list):
            current_coord_idx = i % len(coord_list_cycle)
            bbox_to_use = coord_list_cycle[current_coord_idx]
            original_frame_to_use = copy.deepcopy(frame_list_cycle[current_coord_idx])

            x1_orig, y1_orig, x2_orig, y2_orig = bbox_to_use
            
            # --- START OF FIX ---
            # Re-apply extra_margin logic to y2 for correct resizing and blending
            # This makes the blending region consistent with how the latent was generated.
            if args.version == "v15":
                y2_for_blending = y2_orig + args.extra_margin
                y2_for_blending = min(y2_for_blending, original_frame_to_use.shape[0])
            else:
                y2_for_blending = y2_orig # In case args.version is ever changed from "v15"
            
            current_y2_crop = y2_for_blending # Use this adjusted y2 for all subsequent operations
            # --- END OF FIX ---

            try:
                resized_res_frame = cv2.resize(res_frame_np.astype(np.uint8), (x2_orig - x1_orig, current_y2_crop - y1_orig))
            except Exception as e_resize:
                print(f"Warning: Could not resize frame {i}. Error: {e_resize}. Skipping frame.")
                continue

            if args.version == "v15":
                combined_frame = get_image(original_frame_to_use, resized_res_frame,
                                           [x1_orig, y1_orig, x2_orig, current_y2_crop],
                                           mode=args.parsing_mode, fp=self.fp)
            else: # v1
                combined_frame = get_image(original_frame_to_use, resized_res_frame,
                                           [x1_orig, y1_orig, x2_orig, y2_orig],
                                           fp=self.fp)
            
            cv2.imwrite(os.path.join(result_img_save_path, f"{str(i).zfill(8)}.png"), combined_frame)
        print(f"All frames processed and saved to {result_img_save_path}")

        temp_video_for_audio_merge = os.path.join(result_dir, "temp_video.mp4")
        
        # FFmpeg command to handle odd dimensions if present, by scaling down to nearest even
        # This aligns with the solution that previously worked for you.
        # vf_options = "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" 
        vf_options = "format=yuv420p" # <--- This line

        cmd_img2video = (f"ffmpeg -y -v warning -r {fps} -f image2 "
                         f"-i \"{os.path.join(result_img_save_path, '%08d.png')}\" "
                         f"-vf \"{vf_options}\" "
                         f"-vcodec libx264 -crf 18 \"{temp_video_for_audio_merge}\"")
        
        print(f"Executing: {cmd_img2video}")
        try:
            subprocess.run(cmd_img2video, shell=True, check=True)
        except subprocess.CalledProcessError as e_ffmpeg_vid:
            print(f"Error during video generation from frames: {e_ffmpeg_vid}")
            if not os.listdir(result_img_save_path):
                 print(f"ERROR: No frames found in {result_img_save_path} for ffmpeg to process.")
            raise RuntimeError(f"ffmpeg failed to create video from frames. Command: {cmd_img2video}. Error: {e_ffmpeg_vid}")


        cmd_combine_audio = (f"ffmpeg -y -v warning -i \"{audio_path}\" -i \"{temp_video_for_audio_merge}\" "
                             f"-c:v copy -c:a aac -strict experimental \"{final_video_path}\"")
        print(f"Executing: {cmd_combine_audio}")
        try:
            subprocess.run(cmd_combine_audio, shell=True, check=True)
        except subprocess.CalledProcessError as e_ffmpeg_aud:
            print(f"Error during audio merge: {e_ffmpeg_aud}")
            raise RuntimeError(f"ffmpeg failed to merge audio. Command: {cmd_combine_audio}. Error: {e_ffmpeg_aud}")

        try:
            shutil.rmtree(result_img_save_path)
            if os.path.exists(temp_video_for_audio_merge):
                os.remove(temp_video_for_audio_merge)
        except Exception as e_clean:
            print(f"Warning: Error during cleanup of intermediate files: {e_clean}")

        if not os.path.exists(final_video_path):
            raise RuntimeError(f"Final video not found at {final_video_path} after processing.")
            
        print(f"MuseTalk generation complete. Final video at: {final_video_path}")
        return final_video_path