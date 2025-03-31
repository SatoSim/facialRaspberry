import os
import time

# Define the folder to save images
save_folder = "captured_images"
os.makedirs(save_folder, exist_ok=True)

# Number of images to capture
num_images = 10  # Change this as needed

# Capture multiple images with a delay
for i in range(num_images):
    image_path = os.path.join(save_folder, f"image_{i+1}.jpg")
    os.system(f"raspistill -o {image_path}")
    print(f"Captured: {image_path}")
    time.sleep(0.01)  # 0.2-second delay
