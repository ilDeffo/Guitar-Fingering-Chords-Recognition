import os

import pandas as pd

def create_final_csv(root_dir, save_path):
    final_df = None
    for dir in os.listdir(root_dir):
        if final_df is None:
            final_df = pd.read_csv(os.path.join(root_dir, dir, 'bounding_boxes.csv'))
        else:
            df = pd.read_csv(os.path.join(root_dir, dir, 'bounding_boxes.csv'))
            final_df = pd.concat([final_df, df])

    final_df.to_csv(save_path, index=False)


if __name__ == '__main__':
    root_dir = os.path.join('egohands_data', '_LABELLED_SAMPLES')
    save_path = os.path.join('egohands_data', 'all_images_bounding_boxes.csv')
    create_final_csv(root_dir, save_path)


