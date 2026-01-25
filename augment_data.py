import os
import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# 1. Configuration - Points to your existing dataset folder
DATASET_PATH = "dataset/faceskin" 
TARGET_COUNT = 1000
MINORITY_CLASSES = ['normal', 'dry', 'oily']

# 2. Define Augmentation Rules (Preserving skin texture is critical)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    fill_mode='nearest'
)

def run_augmentation():
    for category in MINORITY_CLASSES:
        folder_path = os.path.join(DATASET_PATH, category)
        if not os.path.exists(folder_path):
            continue
            
        # Get list of existing genuine images
        current_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        num_to_generate = TARGET_COUNT - len(current_images)
        
        if num_to_generate <= 0:
            print(f"Skipping {category}: Already balanced.")
            continue
            
        print(f"Adding {num_to_generate} augmented images to {category}...")
        
        generated_count = 0
        while generated_count < num_to_generate:
            # Pick a random genuine image you already verified
            img_name = random.choice(current_images)
            img_path = os.path.join(folder_path, img_name)
            
            try:
                img = load_img(img_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                
                # Save augmented versions directly into the class folder
                for batch in datagen.flow(x, batch_size=1, save_to_dir=folder_path, 
                                          save_prefix='aug_', save_format='jpg'):
                    generated_count += 1
                    break 
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    run_augmentation()
    