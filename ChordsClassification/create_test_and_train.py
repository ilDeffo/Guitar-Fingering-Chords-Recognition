import os
import random
from shutil import copyfile

if __name__ == "__main__":
    all_images = [img for img in os.listdir("../cropped_images")]
    random.shuffle(all_images)

    train_val_list, test_list = all_images[:int(0.8 * len(all_images))], all_images[int(0.8 * len(all_images)):]

    os.mkdir("../cropped_images/test")
    os.mkdir("../cropped_images/train")
    for im in test_list:
        if os.path.isdir(im):
            continue
        copyfile(os.path.join("../cropped_images", im), os.path.join("../cropped_images/test", im))
    for im in train_val_list:
        if os.path.isdir(im):
            continue
        copyfile(os.path.join("../cropped_images", im), os.path.join("../cropped_images/train", im))
