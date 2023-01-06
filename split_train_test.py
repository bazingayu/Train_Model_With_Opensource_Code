import os
import shutil
from tqdm import tqdm

root_dir = "/home/hank/Desktop/Dataset/images"
target_dir_train = "/home/hank/Desktop/Dataset/images_train"
target_dir_test = "/home/hank/Desktop/Dataset/images_valid"
os.makedirs(target_dir_train, exist_ok=True)
os.makedirs(target_dir_test, exist_ok=True)

filenames = os.listdir(root_dir)

trainnames = filenames[:-100]
testnames = filenames[-100:]

for filename1 in tqdm(trainnames):
    filename = os.path.join(root_dir, filename1)
    target = os.path.join(target_dir_train, filename1)
    shutil.copyfile(filename, target)

for filename1 in tqdm(testnames):
    filename = os.path.join(root_dir, filename1)
    target = os.path.join(target_dir_test, filename1)
    shutil.copyfile(filename, target)


