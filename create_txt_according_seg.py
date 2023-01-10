import os
import cv2

root_dir = "/home/hank/Desktop/Dataset/seg/train/images"
root_dir_mask = "/home/hank/Desktop/Dataset/seg/train/mask"
target_dir = "/home/hank/Desktop/Dataset/seg/train/images_seg"
os.makedirs(target_dir, exist_ok=True)
f = open("/home/hank/Desktop/Dataset/seg_cls_train.txt", "w")

for filename1 in os.listdir(root_dir):
    if filename1[-4:] == ".png":
        continue
    filename = os.path.join(root_dir, filename1)
    image = cv2.imread(filename)
    maskname = os.path.join(root_dir_mask, filename1[:-4] + ".png")
    mask = cv2.imread(maskname, 0)
    image[mask==0] = 0
    cv2.imwrite(os.path.join(target_dir, filename1), image)
    f.write(os.path.join(target_dir, filename1) + " " + filename[-5] + "\n")
f.close()

root_dir = "/home/hank/Desktop/Dataset/seg/val/images"
root_dir_mask = "/home/hank/Desktop/Dataset/seg/val/mask"
target_dir = "/home/hank/Desktop/Dataset/seg/val/images_seg"
os.makedirs(target_dir, exist_ok=True)
f = open("/home/hank/Desktop/Dataset/seg_cls_val.txt", "w")

for filename1 in os.listdir(root_dir):
    if filename1[-4:] == ".png":
        continue
    filename = os.path.join(root_dir, filename1)
    image = cv2.imread(filename)
    maskname = os.path.join(root_dir_mask, filename1[:-4] + ".png")
    mask = cv2.imread(maskname, 0)
    image[mask==0] = 0
    cv2.imwrite(os.path.join(target_dir, filename1), image)
    f.write(os.path.join(target_dir, filename1) + " " + filename[-5] + "\n")
f.close()