import pandas as pd
import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import os

dir = os.path.join('egohands_data', 'all_images')

df = pd.read_csv(os.path.join('egohands_data', 'all_images_bounding_boxes.csv'))

# consideriamo un'immagine
x = df.iloc[500]
image_name = x['image_name']
box_1 = x.loc[['box2_x1', 'box2_y1', 'box2_x2', 'box2_y2']].astype(int).tolist()
if box_1[0] is None:
    import sys
    sys.exit(0)

im = cv.imread(os.path.join(dir, image_name))
cv.rectangle(im, (box_1[0], box_1[1]), (box_1[2], box_1[3]), (0, 0, 255), 2)
plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
plt.show()

