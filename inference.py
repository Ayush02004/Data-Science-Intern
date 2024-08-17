import torch
from torchvision import transforms
from PIL import Image
import argparse





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script for image classification')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model weights')
    
    args = parser.parse_args()
    
    prediction = predict(args.image_path, args.model_path)
    print(f'Predicted class: {prediction}')