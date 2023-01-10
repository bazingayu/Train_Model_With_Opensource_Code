import os

root_dir = "/home/hank/Desktop/Dataset/cropped_images"
files = os.listdir(root_dir)

f = open("/home/hank/Desktop/Dataset/cls_train.txt", "w")
for filename1 in files[:-100]:
    filename = os.path.join(root_dir, filename1)
    cls = str(filename1[:-4].split("_")[-1])
    f.write(filename + " " + cls + "\n")
f.close()

f = open("/home/hank/Desktop/Dataset/cls_val.txt", "w")
for filename1 in files[-100:]:
    filename = os.path.join(root_dir, filename1)
    cls = str(filename1[:-4].split("_")[-1])
    f.write(filename + " " + cls + "\n")
f.close()