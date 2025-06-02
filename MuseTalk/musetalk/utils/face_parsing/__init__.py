import torch
import time
import os # Make sure os is imported
import cv2
import numpy as np
from PIL import Image
from .model import BiSeNet # Assuming BiSeNet is in model.py in the same directory
import torchvision.transforms as transforms

# --- Calculate paths relative to this script's location ---
FACE_PARSING_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# e.g., L:\...\MuseTalk\musetalk\utils\face_parsing\
MUSE_TALK_UTILS_DIR = os.path.dirname(FACE_PARSING_SCRIPT_DIR)
# e.g., L:\...\MuseTalk\musetalk\utils\
MUSE_TALK_PACKAGE_ROOT = os.path.dirname(MUSE_TALK_UTILS_DIR)
# e.g., L:\...\MuseTalk\musetalk\
MUSE_TALK_PROJECT_ROOT_FROM_FACE_PARSING = os.path.dirname(MUSE_TALK_PACKAGE_ROOT)
# e.g., L:\...\MuseTalk\  (This is the root of the MuseTalk project)
# --- End path calculation ---

class FaceParsing():
    def __init__(self, left_cheek_width=80, right_cheek_width=80):
        # Construct absolute paths for the models used by model_init
        abs_resnet_path = os.path.normpath(os.path.join(
            MUSE_TALK_PROJECT_ROOT_FROM_FACE_PARSING,
            'models', 'face-parse-bisent', 'resnet18-5c106cde.pth'
        ))
        abs_model_pth = os.path.normpath(os.path.join(
            MUSE_TALK_PROJECT_ROOT_FROM_FACE_PARSING,
            'models', 'face-parse-bisent', '79999_iter.pth'
        ))

        print(f"[FaceParsing] Using ResNet path: {abs_resnet_path}")
        print(f"[FaceParsing] Using Model path: {abs_model_pth}")

        if not os.path.isfile(abs_resnet_path):
            raise FileNotFoundError(f"FaceParsing ResNet model not found at: {abs_resnet_path}")
        if not os.path.isfile(abs_model_pth):
            raise FileNotFoundError(f"FaceParsing main model not found at: {abs_model_pth}")

        self.net = self.model_init(resnet_path=abs_resnet_path, model_pth=abs_model_pth)
        self.preprocess = self.image_preprocess()
        # Ensure all size parameters are integers
        cone_height = 21
        tail_height = 12
        total_size = cone_height + tail_height
        
        # Create kernel with explicit integer dimensions
        kernel = np.zeros((total_size, total_size), dtype=np.uint8)
        center_x = total_size // 2  # Ensure center coordinates are integers
        
        # Cone part
        for row in range(cone_height):
            if row < cone_height//2:
                continue
            width = int(2 * (row - cone_height//2) + 1)
            start = int(center_x - (width // 2))
            end = int(center_x + (width // 2) + 1)
            kernel[row, start:end] = 1

        # Vertical extension part
        if cone_height > 0:
            base_width = int(kernel[cone_height-1].sum())
        else:
            base_width = 1
        
        for row in range(cone_height, total_size):
            start = max(0, int(center_x - (base_width//2)))
            end = min(total_size, int(center_x + (base_width//2) + 1))
            kernel[row, start:end] = 1
        self.kernel = kernel
        
        # Modify cheek erosion kernel to be flatter ellipse
        self.cheek_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (35, 3))
        
        # Add cheek area mask (protect chin area)
        self.cheek_mask = self._create_cheek_mask(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)
        
    def _create_cheek_mask(self, left_cheek_width=80, right_cheek_width=80):
        """Create cheek area mask (1/4 area on both sides)"""
        mask = np.zeros((512, 512), dtype=np.uint8)
        center = 512 // 2
        cv2.rectangle(mask, (0, 0), (center - left_cheek_width, 512), 255, -1)    # Left cheek
        cv2.rectangle(mask, (center + right_cheek_width, 0), (512, 512), 255, -1)  # Right cheek
        return mask

    def model_init(self, 
                   resnet_path='./models/face-parse-bisent/resnet18-5c106cde.pth', 
                   model_pth='./models/face-parse-bisent/79999_iter.pth'):
        net = BiSeNet(resnet_path)
        if torch.cuda.is_available():
            net.cuda()
            net.load_state_dict(torch.load(model_pth)) 
        else:
            net.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))
        net.eval()
        return net

    def image_preprocess(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image, size=(512, 512), mode="raw"):
        if isinstance(image, str):
            image = Image.open(image)

        width, height = image.size
        with torch.no_grad():
            image = image.resize(size, Image.BILINEAR)
            img = self.preprocess(image)
            if torch.cuda.is_available():
                img = torch.unsqueeze(img, 0).cuda()
            else:
                img = torch.unsqueeze(img, 0)
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
            # Add 14:neck, remove 10:nose and 7:8:9
            if mode == "neck":
                parsing[np.isin(parsing, [1, 11, 12, 13, 14])] = 255
                parsing[np.where(parsing!=255)] = 0
            elif mode == "jaw":
                face_region = np.isin(parsing, [1])*255
                face_region = face_region.astype(np.uint8)
                original_dilated = cv2.dilate(face_region, self.kernel, iterations=1)
                eroded = cv2.erode(original_dilated, self.cheek_kernel, iterations=2)
                face_region = cv2.bitwise_and(eroded, self.cheek_mask)
                face_region = cv2.bitwise_or(face_region, cv2.bitwise_and(original_dilated, ~self.cheek_mask))
                parsing[(face_region==255) & (~np.isin(parsing, [10]))] = 255         
                parsing[np.isin(parsing, [11, 12, 13])] = 255
                parsing[np.where(parsing!=255)] = 0
            else:
                parsing[np.isin(parsing, [1, 10, 11, 12, 13])] = 255
                parsing[np.where(parsing!=255)] = 0

        parsing = Image.fromarray(parsing.astype(np.uint8))
        return parsing

if __name__ == "__main__":
    fp = FaceParsing()
    segmap = fp('154_small.png')
    segmap.save('res.png')
    
