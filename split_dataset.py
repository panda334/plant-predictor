import os
import shutil
import random

random.seed(42)

SOURCE_DIR = r"C:\Users\prime\Desktop\workspace\PlantVillage"
DEST_DIR = "dataset_split"
SPLIT_RATIO = 0.8

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_index]
    val_images = images[split_index:]

    for split_type, split_images in [("train", train_images), ("val", val_images)]:
        split_class_dir = os.path.join(DEST_DIR, split_type, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy2(src, dst)

print("Dataset split completed successfully!")
