import os
import random
from shutil import copyfile

DATA_DIR = "cropped_rotated_processed_images_2"

if __name__ == "__main__":
    all_images = [img for img in os.listdir(f"../chords_data/{DATA_DIR}")]
    random.shuffle(all_images)

    train_val_list, test_list = all_images[:int(0.8 * len(all_images))], all_images[int(0.8 * len(all_images)):]

    os.mkdir(f"../chords_data/{DATA_DIR}/test")
    os.mkdir(f"../chords_data/{DATA_DIR}/train")
    for im in test_list:
        if os.path.isdir(im):
            continue
        copyfile(os.path.join(f"../chords_data/{DATA_DIR}", im), os.path.join(f"../chords_data/{DATA_DIR}/test", im))
    for im in train_val_list:
        if os.path.isdir(im):
            continue
        copyfile(os.path.join(f"../chords_data/{DATA_DIR}", im), os.path.join(f"../chords_data/{DATA_DIR}/train", im))
