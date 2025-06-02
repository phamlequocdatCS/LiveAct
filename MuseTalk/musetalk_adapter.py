# In lipsync_project/MuseTalk/musetalk_adapter.py

import os
import subprocess
import sys
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
import copy # For deepcopy
import shutil # For rmtree
import glob # For finding output video if needed

# --- Path Setup ---
MUSE_TALK_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Should be /path/to/MuseTalk
if MUSE_TALK_ROOT_DIR not in sys.path:
    sys.path.insert(0, MUSE_TALK_ROOT_DIR)

print(f"[musetalk_adapter.py] MUSE_TALK_ROOT_DIR: {MUSE_TALK_ROOT_DIR}")
print(f"[musetalk_adapter.py] Current sys.path: {sys.path}")
# --- End Path Setup ---

try:
    from transformers import WhisperModel # inference.py uses this

    from musetalk.utils.blending import get_image
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model # load_all_model is key
    from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder

    print("[musetalk_adapter.py] Successfully imported MuseTalk components from 'musetalk.utils.*'")
except ImportError as e:
    print(f"[musetalk_adapter.py] Error during import from 'musetalk.utils.*': {e}")
    raise

class MuseTalkGenerator:
    def __init__(self, args_config, device_id="cuda:0"): # args_config will mimic the argparse from inference.py
        print(f"Initializing MuseTalkGenerator with device_id: {device_id}")
        self.device = torch.device(device_id if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.args = args_config # Store the config

        # Load models (mimicking main() in inference.py)
        print("Loading all models (VAE, UNet, PE)...")
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=self.args.unet_model_path, # This path needs to be correct
            vae_type=self.args.vae_type,
            unet_config=self.args.unet_config, # This path needs to be correct
            device=self.device
        )
        self.timesteps = torch.tensor([0], device=self.device)
        print("Models VAE, UNet, PE loaded.")

        if self.args.use_float16:
            print("Using float16 precision...")
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()

        # Ensure models are on the correct device (load_all_model might handle some of this)
        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)
        print("Models moved to device.")

        print("Initializing AudioProcessor...")
        self.audio_processor = AudioProcessor(feature_extractor_path=self.args.whisper_dir) # Path to whisper model
        print("AudioProcessor initialized.")

        print("Loading WhisperModel...")
        self.weight_dtype = self.unet.model.dtype # Get dtype from UNet after potential half()
        self.whisper = WhisperModel.from_pretrained(self.args.whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)
        print("WhisperModel loaded.")

        print("Initializing FaceParsing...")
        if self.args.version == "v15":
            self.fp = FaceParsing(
                left_cheek_width=self.args.left_cheek_width,
                right_cheek_width=self.args.right_cheek_width
            )
        else: # v1
            self.fp = FaceParsing()
        print("FaceParsing initialized.")
        print("MuseTalkGenerator initialized successfully.")

    @torch.no_grad()
    def generate(self, image_path, audio_path, result_dir):
        # This method will now closely follow the logic within the for loop of inference.py's main()
        # 'result_dir' is the unique temp dir for this specific API call (e.g., temp_files/uuid/)
        
        args = self.args # Use the stored config

        # --- Mimic task setup from inference.py ---
        input_basename = os.path.splitext(os.path.basename(image_path))[0]
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_basename = f"{input_basename}_{audio_basename}"

        # temp_dir for this specific generation, within the request's own temp space
        # scripts/inference.py makes a subdir like: os.path.join(args.result_dir, f"{args.version}")
        # We'll use result_dir directly as the base for intermediate files.
        # Example: result_dir = "temp_files/some_uuid/"
        #          result_img_save_path = "temp_files/some_uuid/frames_for_video/"
        #          output_vid_name      = "temp_files/some_uuid/final_video.mp4"

        result_img_save_path = os.path.join(result_dir, output_basename + "_frames") # Dir for ffmpeg to make video from
        os.makedirs(result_img_save_path, exist_ok=True)

        # Output video will be named and placed by us after ffmpeg
        # final_video_path_temp = os.path.join(result_dir, output_basename + "_temp.mp4") # video without audio
        final_video_path = os.path.join(result_dir, output_basename + ".mp4") # final video with audio

        # --- Frame and Audio Processing (from inference.py) ---
        # Since input is always an image:
        if get_file_type(image_path) != "image":
            raise ValueError(f"Input video_path '{image_path}' is not a valid image file.")
        input_img_list = [image_path]
        fps = args.fps # Use predefined FPS for image input

        print(f"Extracting audio features for: {audio_path}")
        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path, weight_dtype=self.weight_dtype)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        print(f"Audio features extracted, num_chunks: {len(whisper_chunks)}")

        # Landmark and BBox extraction (for a single image)
        # inference.py has a coord_placeholder and save/load logic we can simplify for single image
        print(f"Extracting landmarks for image: {image_path}")
        # bbox_shift is 0 for v15, configurable for v1
        bbox_shift_val = 0 if args.version == "v15" else args.bbox_shift # Simplified
        
        coord_list, frame_list = get_landmark_and_bbox([image_path], bbox_shift_val) # Pass as list

        if not coord_list or coord_list[0] == coord_placeholder:
            # Try to provide more insight if landmark extraction fails
            img_check = cv2.imread(image_path)
            if img_check is None:
                 raise ValueError(f"Source image is invalid or not found for landmark extraction: {image_path}")
            raise ValueError(f"No landmarks detected or placeholder returned for image: {image_path}. Check image clarity and face visibility.")
        print(f"Landmarks extracted. Coords: {coord_list[0]}")

        # Latent processing for the single input image
        input_latent_list = []
        bbox, frame = coord_list[0], frame_list[0] # Single image
        x1, y1, x2, y2 = bbox
        if args.version == "v15": # As per inference.py logic for cropping
            y2 = y2 + args.extra_margin
            y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
        latents = self.vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)
        print("Input image processed to latent.")

        # "Cycle" lists for datagen (since it expects iterable inputs that can be longer than audio)
        # For a single image, these "cycled" lists will just repeat the single image's data.
        frame_list_cycle = frame_list + frame_list[::-1] # Effectively [frame, frame]
        coord_list_cycle = coord_list + coord_list[::-1] # Effectively [bbox, bbox]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1] # Effectively [latent, latent]

        # Batch inference
        print("Starting batch inference...")
        video_num = len(whisper_chunks)
        batch_size = args.batch_size
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list_cycle, # Will cycle the single image's latent
            batch_size=batch_size,
            delay_frame=0, # As in inference.py
            device=self.device,
        )

        res_frame_list = []
        # total_batches = int(np.ceil(float(video_num) / batch_size)) # For tqdm if used

        for i, (whisper_batch, latent_batch) in enumerate(gen): # Removed tqdm for now
            audio_feature_batch = self.pe(whisper_batch) # PE already on device
            # latent_batch is already on device from datagen
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype) # Ensure correct dtype

            pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
            # VAE might need specific dtype too, often float32 for decode
            # pred_latents = pred_latents.to(dtype=self.vae.vae.dtype) # if vae has specific dtype needs
            recon = self.vae.decode_latents(pred_latents)
            for res_frame_pil in recon: # Assuming recon gives list of PIL Images or similar
                # Convert PIL to NumPy array (OpenCV BGR format) if needed by get_image
                # res_frame_np = np.array(res_frame_pil.convert("RGB"))[:, :, ::-1] # RGB to BGR
                # If 'recon' already gives ndarray in BGR:
                res_frame_list.append(res_frame_pil) # Assuming recon gives ndarray in BGR
        print(f"Inference complete. Generated {len(res_frame_list)} frames.")

        # Padding generated images and saving (from inference.py)
        print("Processing and saving generated frames...")
        for i, res_frame_np in enumerate(res_frame_list):
            # Determine which original frame/bbox to use for pasting
            # Since we have one source image, use its bbox and frame repeatedly
            current_coord_idx = i % len(coord_list_cycle) # Should always pick one of the two identical entries
            bbox_to_use = coord_list_cycle[current_coord_idx]
            original_frame_to_use = copy.deepcopy(frame_list_cycle[current_coord_idx])

            x1_orig, y1_orig, x2_orig, y2_orig = bbox_to_use
            # y2_orig adjustment if v15 (already done when creating latents, but ensure consistency for pasting)
            if args.version == "v15":
                 current_y2_crop = y2_orig # This y2_orig should be the one after adding extra_margin
            else:
                 current_y2_crop = y2_orig


            try:
                # res_frame_np is the 256x256 generated talking face
                # Resize it to the original crop dimensions
                resized_res_frame = cv2.resize(res_frame_np.astype(np.uint8), (x2_orig - x1_orig, current_y2_crop - y1_orig))
            except Exception as e_resize:
                print(f"Warning: Could not resize frame {i}. Error: {e_resize}. Skipping frame.")
                continue

            # Merge results (pasting into original_frame_to_use)
            if args.version == "v15":
                combined_frame = get_image(original_frame_to_use, resized_res_frame,
                                           [x1_orig, y1_orig, x2_orig, current_y2_crop], # Use the potentially adjusted y2 for pasting
                                           mode=args.parsing_mode, fp=self.fp)
            else: # v1
                combined_frame = get_image(original_frame_to_use, resized_res_frame,
                                           [x1_orig, y1_orig, x2_orig, y2_orig], # Original y2 for v1
                                           fp=self.fp)
            
            cv2.imwrite(os.path.join(result_img_save_path, f"{str(i).zfill(8)}.png"), combined_frame)
        print(f"All frames processed and saved to {result_img_save_path}")

        # Create video from frames
        temp_video_for_audio_merge = os.path.join(result_dir, "temp_video.mp4")
        vf_options = "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"
        cmd_img2video = (f"ffmpeg -y -v warning -r {fps} -f image2 "
                         f"-i \"{os.path.join(result_img_save_path, '%08d.png')}\" "
                         f"-vf \"{vf_options}\" " # Apply the video filter for scaling and format
                         f"-vcodec libx264 -crf 18 \"{temp_video_for_audio_merge}\"")
        print(f"Executing: {cmd_img2video}")
        try:
            subprocess.run(cmd_img2video, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during video generation from frames: {e}")
            # Check if any frames were generated
            if not os.listdir(result_img_save_path):
                 raise RuntimeError(f"No frames were generated in {result_img_save_path}. Cannot create video.")
            raise RuntimeError(f"ffmpeg failed to create video from frames. Command: {cmd_img2video}. Error: {e}")


        # Combine with audio
        cmd_combine_audio = (f"ffmpeg -y -v warning -i \"{audio_path}\" -i \"{temp_video_for_audio_merge}\" "
                             f"-c:v copy -c:a aac -strict experimental \"{final_video_path}\"")
        print(f"Executing: {cmd_combine_audio}")
        try:
            subprocess.run(cmd_combine_audio, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during audio merge: {e}")
            raise RuntimeError(f"ffmpeg failed to merge audio. Command: {cmd_combine_audio}. Error: {e}")

        # Cleanup
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