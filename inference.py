import warnings
warnings.filterwarnings("ignore")

import torch
from torchvision.transforms import v2 as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from u2net import U2NET
import numpy as np
import cv2
import os 
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

# Define transformations for the input image
img_height, img_width = 320, 320
transform_image = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((img_height, img_width), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToDtype(torch.float32, scale=True),
])

# Load the trained model
def load_model(model_path):
    model = U2NET()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def infer(image_path, model, transform):
    # Load and preprocess the input image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image)

    if isinstance(output, tuple):
        pred_mask = output[0]
    pred_mask_binary = pred_mask > 0.5
    pred_mask_binary = pred_mask_binary.cpu().numpy()
    pred_mask_binary = pred_mask_binary.squeeze()
    return pred_mask_binary

# get orientation of the image
def get_orientation(mask):
    def max_ones_in_line(line):
        max_count = 0
        current_count = 0
        for value in line:
            if value == 1:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count

    # Convert mask to numpy array if it's not already
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    
    mask[mask == 255] = 1

    max_row_ones = max(max_ones_in_line(row) for row in mask)
    max_col_ones = max(max_ones_in_line(col) for col in mask.T)  # Transpose to iterate over columns

    return "Horizontal" if max_row_ones > max_col_ones else "Vertical"

# get ground mask using kmeans clustering
def get_ground_mask(img, k=3):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Reshape the image into a 2D array of pixels
    pixel_values = hsv_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria, number of clusters(K), and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 3  # Number of clusters, adjust as necessary
    _, labels, (centers) = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8 bit values
    centers = np.uint8(centers)
    labels = labels.flatten()

    # Convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image dimension
    segmented_image = segmented_image.reshape(hsv_image.shape)

    # Convert the image back to BGR to display
    segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR)

    # Identify the cluster with the maximum distribution at the bottom
    # Initialize an array to store the sum of pixels for each cluster
    cluster_distribution = np.zeros(K)

    # Calculate the distribution of clusters along the y-axis
    height, width = hsv_image.shape[:2]
    for i in range(K):
        cluster_mask = (labels == i).reshape(height, width)  # Mask for the current cluster
        cluster_distribution[i] = np.sum(cluster_mask[-height//4:])  # Sum pixels in the bottom quarter of the image

    # Identify the cluster with the maximum distribution at the bottom
    ground_cluster = np.argmax(cluster_distribution)

    ground_mask = (labels == ground_cluster).reshape(height, width).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    ground_mask = cv2.dilate(ground_mask, kernel, iterations=2)

    return ground_mask


# add the car to the background
def blend_car_with_background(car_image, car_mask, background_image):

    if car_mask.dtype != 'uint8':
        car_mask = car_mask.astype('uint8')
    # Find contours of the car mask
    if get_orientation(car_mask) == "Vertical":
        car_mask = cv2.rotate(car_mask, cv2.ROTATE_90_CLOCKWISE)
        car_image = cv2.rotate(car_image, cv2.ROTATE_90_CLOCKWISE)
    
    car_contours, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box for the car
    x, y, w, h = cv2.boundingRect(car_contours[0])

    # Resize the background image to match the car image dimensions
    background = cv2.resize(background_image, car_image.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)

    # Get the ground mask and find the y-coordinate of the lowest non-zero pixel
    ground_mask = get_ground_mask(background)
    ground_y = np.max(np.where(ground_mask > 0)[0]) - 20

    # Calculate the new y-coordinate for the car
    new_y = ground_y - h

    # Center the car horizontally
    new_x = (ground_mask.shape[1] - w) // 2

    # Create a translation matrix
    translation_matrix = np.float32([[1, 0, new_x - x], [0, 1, new_y - y]])

    # Translate the car mask
    translated_car_mask = cv2.warpAffine(car_mask, translation_matrix, (car_mask.shape[1], car_mask.shape[0]))

    # Translate the car image
    translated_car_image = cv2.warpAffine(car_image, translation_matrix, (car_image.shape[1], car_image.shape[0]))

    # Apply the mask to the car image and background
    translated_car_image[translated_car_mask == 0] = 0
    background[translated_car_mask == 1] = 0

    # Blend the images
    result_image = cv2.add(background, translated_car_image)

    return result_image

def visualize_result(original_image, mask, result):
    
    # Visualize the original image and the segmentation mask    
    plt.figure(figsize=(18, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.axis('off')
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(original_image)
    plt.imshow(mask, alpha=0.5)
    plt.axis('off')
    plt.title('Segmentation Mask')
    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.axis('off')
    plt.title('Result')
    plt.tight_layout()
    plt.savefig('result_plot.png',  dpi=600)

# Add the main function to parse arguments and call the necessary functions
def main():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--background', type=str, required=True, help='Path to the background image')
    args = parser.parse_args()

    model = load_model(args.model)
    mask = infer(args.image, model, transform_image)

    # Read the image using OpenCV
    original_image = cv2.imread(args.image)

    # Resize the image
    original_image = cv2.resize(original_image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Rotate the image if the orientation is vertical
    if get_orientation(mask) == "Vertical":
        if mask.dtype != 'uint8':
            mask = mask.astype('uint8')
        original_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
        mask = np.rot90(mask, k=3)  # Rotate counter-clockwise 90 degrees

    background = cv2.imread(args.background)
    background = cv2.resize(background, (img_height, img_width), interpolation=cv2.INTER_CUBIC)
    result = blend_car_with_background(original_image, mask, background)
    visualize_result(original_image, mask, result)

if __name__ == "__main__":
    main()