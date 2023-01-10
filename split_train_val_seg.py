import os
import shutil
from tqdm import tqdm

root_dir_image = "/home/hank/Desktop/Dataset/seg_images"
root_dir_mask = "/home/hank/Desktop/Dataset/seg_label"

target_dir_train_image = "/home/hank/Desktop/Dataset/seg/train/images"
target_dir_train_mask = "/home/hank/Desktop/Dataset/seg/train/mask"
target_dir_val_image = "/home/hank/Desktop/Dataset/seg/val/images"
target_dir_val_mask = "/home/hank/Desktop/Dataset/seg/val/mask"
os.makedirs(target_dir_train_image, exist_ok=True)
os.makedirs(target_dir_train_mask, exist_ok=True)
os.makedirs(target_dir_val_image, exist_ok=True)
os.makedirs(target_dir_val_mask, exist_ok=True)

filenames = os.listdir(root_dir_image)
for filename1 in tqdm(filenames[:-100]):
    filename = os.path.join(root_dir_image, filename1)
    maskname = os.path.join(root_dir_mask, filename1[:-4] + ".png")
    target_image = os.path.join(target_dir_train_image, filename1)
    target_mask = os.path.join(target_dir_train_mask, filename1[:-4] + ".png")
    shutil.copyfile(filename, target_image)
    shutil.copyfile(maskname, target_mask)

for filename1 in tqdm(filenames[-100:]):
    filename = os.path.join(root_dir_image, filename1)
    maskname = os.path.join(root_dir_mask, filename1[:-4] + ".png")
    target_image = os.path.join(target_dir_val_image, filename1)
    target_mask = os.path.join(target_dir_val_mask, filename1[:-4] + ".png")
    shutil.copyfile(filename, target_image)
    shutil.copyfile(maskname, target_mask)
