import pandas as pd
import os


if __name__ == "__main__":
    train_df = pd.DataFrame(columns=["img_name", "label"])
    train_df["img_name"] = os.listdir("../cropped_images/train/")
    for idx, i in enumerate(os.listdir("../cropped_images/train/")):
        if "A" in i:
            train_df["label"][idx] = 0
        if "B" in i:
            train_df["label"][idx] = 1
        if "C" in i:
            train_df["label"][idx] = 2
        if "D" in i:
            train_df["label"][idx] = 3
        if "E" in i:
            train_df["label"][idx] = 4
        if "F" in i:
            train_df["label"][idx] = 5
        if "G" in i:
            train_df["label"][idx] = 6

    train_df.to_csv(r'../cropped_images/train/train_labels.csv', index=False, header=True)
