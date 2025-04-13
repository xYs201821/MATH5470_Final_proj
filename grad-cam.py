import torch
import torch.nn.functional as F
import numpy as np
import cv2  # For heatmap visualization; install using `pip install opencv-python`
import matplotlib.pyplot as plt
from model import CNN20
import yaml
import os

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
    
def load_model(model_path, model_name, device, config_path='./config.yaml'):
    config = yaml.safe_load(open(config_path, 'r'))
    model = CNN20(**config)
    model.load_state_dict(torch.load(os.path.join(model_path, f"{model_name}.pth"), map_location=device))
    model.to(device)
    model.eval()
    return model

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model', help='Path to model file')
    parser.add_argument('--model_name', type=str, default='baseline_I20R20', help='Model name')
    parser.add_argument('--image_path', type=str, default='./monthly_20d/20d_month_has_vb_[20]_ma_1995_images.dat', help='Path to image file')
    parser.add_argument('--config_path', type=str, default='./config.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    args = parser.parse_args()
    model = load_model(args.model_path, args.model_name, args.device, args.config_path)
    model.eval()
    IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
    IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96} 
    images = np.memmap(args.image_path, dtype=np.uint8, mode='r')
    images = images.reshape((-1,1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20]))
    for i in range(3):
        for j in range(8):
            target_block = model.blocks[-i]
            target_layer = target_block[0]
            grad_cam = GradCAM(model=model, target_layers=[target_layer])
            input_tensor = torch.tensor(images[j].copy(), dtype=torch.float).unsqueeze(0).to(args.device)
            print(F.softmax(model(input_tensor), dim=1))
            # Get the GradCAM output
            grayscale_cam = grad_cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]
            
            # Prepare the original image
            image_single_channel = images[j].copy()
            image_gray = np.squeeze(image_single_channel, axis=0)
            
            # Convert to uint8 and ensure proper range [0, 255]
            image_gray = ((image_gray - image_gray.min()) / (image_gray.max() - image_gray.min()) * 255).astype(np.uint8)
            
            # Convert to BGR format
            image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
            
            # Ensure grayscale_cam is in float [0, 1] range
            grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
            
            # Create the CAM visualization
            cam_image = show_cam_on_image(image_rgb / 255.0, grayscale_cam, use_rgb=True)
            height, width = cam_image.shape[:2]
            cam_image_resized = cv2.resize(
            cam_image, 
            (width * 5, height * 5), 
            interpolation=cv2.INTER_CUBIC
        )
            if not os.path.exists(f'output/grad-cam'):
                os.makedirs(f'output/grad-cam', exist_ok=True)
            # Save the image
            cv2.imwrite(
                f'output/grad-cam/{args.model_name}_grad_cam_2019_20d_conv-layer{i}_date_{j}.png', 
                cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            )
