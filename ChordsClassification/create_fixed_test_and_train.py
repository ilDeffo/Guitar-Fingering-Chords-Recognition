import os
import random
import numpy as np
from shutil import copyfile, copy2


DATASET_DIRS = ["cropped_images", "cropped_processed_images", "cropped_processed_rotated_images",
                "cropped_rotated_images", "cropped_rotated_processed_images_0",
                "cropped_rotated_processed_images_1", "cropped_rotated_processed_images_2",
                "cropped_rotated_processed_images_3", "cropped_rotated_processed_images_4",
                "cropped_rotated_processed_images_5", "cropped_rotated_processed_images_6"
                ]
#DATASET_DIRS = ["cropped_images", "cropped_rotated_images",
#                "cropped_rotated_processed_images_5"]

INDEXES_FILENAME = "chords_data_random_indexes.txt"

if __name__ == "__main__":
    for DATA_DIR in DATASET_DIRS:
        all_images = [img for img in os.listdir(f"../chords_data/{DATA_DIR}")]

        indexes = None
        if not os.path.exists(INDEXES_FILENAME):
            indexes = [i for i in range(len(all_images))]
            np.random.shuffle(indexes)
            with open(INDEXES_FILENAME, 'w') as f:
                for i in indexes:
                    if i == len(indexes):
                        f.write(str(i))
                    else:
                        f.write(str(i) + '\n')
        else:
            with open(INDEXES_FILENAME, 'r') as f:
                indexes = [int(line.rstrip('\n')) for line in f]

        # Reordering all_images basing on always the same first random order
        all_images = [all_images[i] for i in indexes]
        # random.shuffle(all_images)

        train_val_list, test_list = all_images[:int(0.8 * len(all_images))], all_images[int(0.8 * len(all_images)):]

        if not os.path.exists(f"../chords_data/{DATA_DIR}/test"):
            os.mkdir(f"../chords_data/{DATA_DIR}/test")
        if not os.path.exists(f"../chords_data/{DATA_DIR}/train"):
            os.mkdir(f"../chords_data/{DATA_DIR}/train")

        for im in test_list:
            if os.path.isdir(im):
                continue
            copyfile(os.path.join(f"../chords_data/{DATA_DIR}", im), os.path.join(f"../chords_data/{DATA_DIR}/test", im))
        for im in train_val_list:
            if os.path.isdir(im):
                continue
            copyfile(os.path.join(f"../chords_data/{DATA_DIR}", im), os.path.join(f"../chords_data/{DATA_DIR}/train", im))
