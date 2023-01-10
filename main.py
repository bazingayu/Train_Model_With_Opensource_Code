import os
import cv2
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

root_dir_images = "/home/hank/Desktop/Dataset/images_train"
root_dir_bbs = "/home/hank/Desktop/Dataset/bounding_box"

img_id = 1
ann_id = 1
img_list = []
ann_list = []
for filename1 in tqdm(os.listdir(root_dir_images)):
    filename = os.path.join(root_dir_images, filename1)

    filename_bbs = os.path.join(root_dir_bbs, filename1[:-4] + ".txt")
    image = cv2.imread(filename)

    height, width, _ = image.shape
    img_dic = {"id": img_id, "width": width, "height": height, "file_name": filename1}

    lines = open(filename_bbs, "r").readlines()
    for line in lines:
        ann_dic = {}
        s = line[:-1].split(" ")

        center_x = float(s[1]) * width
        center_y = float(s[2]) * height
        w = float(s[3]) * width
        h = float(s[4]) * height
        x1 = int(center_x - w / 2)
        x2 = int(center_x + w / 2)
        y1 = int(center_y - h / 2)
        y2 = int(center_y + h / 2)
        ann_dic = {
            "id" : ann_id,
            "image_id" : img_id,
            "category_id": 0,
            "segmentation": [x1, y1, x2, y1, x2, y2, x1, y2],
            "area" : w * h,
            "bbox": [x1, y1, w, h],
            "iscrowd": 0
        }
        ann_list.append(ann_dic)
        ann_id += 1
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    img_list.append(img_dic)
    img_id += 1
        # cv2.circle(image, [x, y], 1, (0, 0, 255))
    # cv2.imshow("win", image)
    # cv2.waitKey(0)
    # break
category_list = [{"id": 0, "name": "strawberry", "supercategory": "strawberry"}]

dic = {"images": img_list, "annotations": ann_list, "categories": category_list}
b = json.dumps(dic)
f = open("strawberry_train.json", "w")
f.write(b)
f.close()
