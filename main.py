import cv2
import os
import torch
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'MiDaS'))

from MiDaS.midas.dpt_depth import DPTDepthModel
from torchvision.transforms import Compose
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet

# Create folders if not exist
for folder in ["frames", "depth_maps", "stereo", "output"]:
    os.makedirs(folder, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/frame_{i:05d}.png", frame)
        i += 1
    cap.release()

def load_midas_model():
    model_path = "MiDaS/weights/dpt_large-midas-2f21e586.pt"
    model = DPTDepthModel(
        path=model_path,
        backbone="vitl16_384",
        non_negative=True
    )
    model.eval()
    model.to(DEVICE)
    return model

def get_transform():
    return Compose([
        Resize(
            384, 384, resize_target=None, keep_aspect_ratio=True,
            ensure_multiple_of=32, resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])

def estimate_depth(model, transform, frame_path, save_path):
    img = cv2.imread(frame_path)
    img_input = transform({"image": img})["image"]
    sample = torch.from_numpy(img_input).to(DEVICE).unsqueeze(0)
    prediction = model.forward(sample)
    depth = prediction.squeeze().detach().cpu().numpy()
    norm_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(save_path, norm_depth.astype(np.uint8))
    return norm_depth

def create_stereo_pair(image, depth_map, disparity_scale=0.08):
    h, w = depth_map.shape
    disparity = (1.0 / (depth_map + 1e-6)) * disparity_scale

    left = np.zeros_like(image)
    right = np.zeros_like(image)

    for y in range(h):
        for x in range(w):
            d = int(disparity[y, x])
            if x - d >= 0:
                left[y, x] = image[y, x - d]
            if x + d < w:
                right[y, x] = image[y, x + d]
    return left, right

def create_anaglyph(left, right):
    anaglyph = np.zeros_like(left)
    anaglyph[:, :, 0] = left[:, :, 0]  # R
    anaglyph[:, :, 1] = right[:, :, 1]  # G
    anaglyph[:, :, 2] = right[:, :, 2]  # B
    return anaglyph

def frames_to_video(input_folder, output_path, fps=30):
    images = sorted(os.listdir(input_folder))
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, _ = frame.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for img in images:
        frame = cv2.imread(os.path.join(input_folder, img))
        out.write(frame)
    out.release()

def main():
    video_path = "video/input.mp4"
    print("[1] Extracting frames...")
    extract_frames(video_path, "frames")

    print("[2] Loading MiDaS model...")
    model = load_midas_model()
    transform = get_transform()

    print("[3] Estimating depth...")
    for frame_file in sorted(os.listdir("frames")):
        frame_path = os.path.join("frames", frame_file)
        depth_path = os.path.join("depth_maps", frame_file)
        estimate_depth(model, transform, frame_path, depth_path)

    print("[4] Generating 3D frames...")
    for frame_file in sorted(os.listdir("frames")):
        img = cv2.imread(f"frames/{frame_file}")
        depth = cv2.imread(f"depth_maps/{frame_file}", 0)
        left, right = create_stereo_pair(img, depth)
        anaglyph = create_anaglyph(left, right)
        cv2.imwrite(f"stereo/{frame_file}", anaglyph)

    print("[5] Rebuilding video...")
    frames_to_video("stereo", "output/3d_output.mp4")
    print("✅ Done! Check 'output/3d_output.mp4'")

if __name__ == "__main__":
    main()