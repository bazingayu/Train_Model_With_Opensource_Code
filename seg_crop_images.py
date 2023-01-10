import os
import cv2
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import numpy as np

root_dir_images = "/home/hank/Desktop/Dataset/images"
root_dir_bbs = "/home/hank/Desktop/Dataset/bounding_box"
root_dir_seg = "/home/hank/Desktop/Dataset/instance_segmentation"
target_dir = "/home/hank/Desktop/Dataset/seg_images"
target_dir_seg = "/home/hank/Desktop/Dataset/seg_label"
os.makedirs(target_dir, exist_ok=True)
os.makedirs(target_dir_seg, exist_ok=True)

img_id = 1
ann_id = 1
img_list = []
ann_list = []
for filename1 in tqdm(os.listdir(root_dir_images)):
    filename = os.path.join(root_dir_images, filename1)

    filename_bbs = os.path.join(root_dir_bbs, filename1[:-4] + ".txt")
    image = cv2.imread(filename)
    seg = cv2.imread(os.path.join(root_dir_seg, filename1[:-4] + ".png"))
    for i in np.unique(seg):
        if i == 0:
            continue
        seg[seg==i] = 255
    # cv2.imshow("win", seg)
    # cv2.waitKey()

    height, width, _ = image.shape
    img_dic = {"id": img_id, "width": width, "height": height, "file_name": filename1}

    lines = open(filename_bbs, "r").readlines()
    for index, line in enumerate(lines):
        ann_dic = {}
        s = line[:-1].split(" ")
        cls = int(s[0])
        center_x = float(s[1]) * width
        center_y = float(s[2]) * height
        w = float(s[3]) * width
        h = float(s[4]) * height
        x1 = int(max(center_x - w / 2 - w * 0.05, 0))
        x2 = int(min(center_x + w / 2 + w * 0.05, width))
        y1 = int(max(center_y - h / 2 - h * 0.05, 0))
        y2 = int(min(center_y + h / 2 + h * 0.05, height))
        crop_image = image[y1:y2, x1:x2]
        crop_seg = seg[y1:y2, x1:x2]
        crop_seg = cv2.cvtColor(crop_seg, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(os.path.join(target_dir, filename1[:-4] + "_" + str(index) + "_" + str(cls) + ".jpg"), crop_image)
        # cv2.imwrite(os.path.join(target_dir_seg, filename1[:-4] + "_" + str(index) + "_" + str(cls) + ".png"), crop_seg)
        cv2.imshow("win", crop_image)
        cv2.waitKey()
        cv2.imshow("win1", crop_seg)
        cv2.waitKey()
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # img_list.append(img_dic)
    # img_id += 1
        # cv2.circle(image, [x, y], 1, (0, 0, 255))
    # cv2.imshow("win", image)
    # cv2.waitKey(0)
    # break
# category_list = [{"id": 0, "name": "strawberry", "supercategory": "strawberry"}]

# dic = {"images": img_list, "annotations": ann_list, "categories": category_list}
# b = json.dumps(dic)
# f = open("strawberry_train.json", "w")
# f.write(b)
# f.close()
