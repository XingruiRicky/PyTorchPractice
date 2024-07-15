import os
import shutil
import random

def split_dataset(source_dir, dest_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # create dir
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'test'), exist_ok=True)

    # get all the dir name in source dir
    classes = []
    for d in os.listdir(source_dir) :
        # adjust the exclude file accordingly
        if d not in['train', 'val', 'test','__pycache__']:
            path = os.path.join(source_dir, d)
            if os.path.isdir(path):
                classes.append(d)


    for cls in classes:
        os.makedirs(os.path.join(dest_dir, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'val', cls), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'test', cls), exist_ok=True)
        
        src_folder = os.path.join(source_dir, cls)
        images = os.listdir(src_folder)
        random.shuffle(images)
        
        train_count = int(len(images) * train_ratio)
        val_count = int(len(images) * val_ratio)
        
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]
        
        for img in train_images:
            shutil.copy(os.path.join(src_folder, img), os.path.join(dest_dir, 'train', cls, img))
        for img in val_images:
            shutil.copy(os.path.join(src_folder, img), os.path.join(dest_dir, 'val', cls, img))
        for img in test_images:
            shutil.copy(os.path.join(src_folder, img), os.path.join(dest_dir, 'test', cls, img))

current_dir = os.path.dirname(os.path.realpath(__file__))

source_dir = current_dir
dest_dir = current_dir

split_dataset(source_dir, dest_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

print("Done")
