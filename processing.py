"""
Image processing & geometric based main script
for "Guitar Fingering & Chords Recognition" project.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from Utils.processing_operators import *
from Utils.geometric_operators import *
from detect_hands_specific_image import get_hand_image_cropped

TMP_DIR = "Temp" + os.sep

# Image file to test for processing
FILE_NAME = 'Dataset/F/F (135).jpeg'
# Processing mode to test
PROCESS_MODE = 4


def process_image(img=None, filename=None, crop=True, process=True, process_mode=0, rotate=True,
                  rotate_first=True, verbose=False, save_images=False):
    """
    Method to crop, process and rotate the image for chords classification.

    :param img: BGR PyTorch image tensor of shape (h, w, c).
    :param filename: If not None, we try to read image from file.
    :param crop: If true, the method crops the image around the hand playing the chord.
    :param process: If true, the method applies the best processing operators found experimentally.
    :param process_mode: Selects the type of processing to apply.
    :param rotate: If true, the method applies the angular correction respect the strings.
    :param rotate_first: If true, the rotation is applied before processing. It can be better or not
                         depending on type of processing applied. For instance, with default process_mode=0
                         it's suggested to rotate after to get better results
    :param verbose:
    :param save_images: If true, the algorithm saves image results in TMP_DIR.
                        Turn false to increment performances.
    :return: BGR PyTorch image tensor of shape (h, w, c).
    """
    if filename:
        img = cv.imread(filename)
        # Convert image to Torch tensor (h, w, c)
        img = torch.from_numpy(img)

    out = img

    if crop:
        # 1. Detecting hand playing the chord to crop region of interest.
        #    From experiments detection actually works better on original image,
        #    even if the difference is around 0.001-0.003.
        # cropped_image = get_hand_image_cropped(img, threshold=0.799, padding=100, verbose=True)
        out = get_hand_image_cropped(out, threshold=0.79, padding=100, verbose=verbose, save_img=save_images)

    if rotate and rotate_first:
        # 2. Calling geometric based operators to do the angle correction based on strings.
        #    From experiments, finding the strings is actually easier on processed image if processed_mode=0.
        out = correct_angle(out, threshold=270, verbose=verbose, save_images=save_images)

    if process:
        # 3. Calling processing operators

        # Intervals for contrast stretching
        bgr_mins = [0, 0, 0]
        bgr_maxs = [180, 180, 180]
        bgr_mins_1 = [0, 0, 0]
        bgr_maxs_1 = [255, 255, 255]

        if process_mode == 0:
            # Processing n.10
            edges = frei_and_chen_edges(out)
            out = saturation(blending(out, edges, a=0.5))
        if process_mode == 1:
            # Processing n.11
            out = out.type(torch.uint8)
            c_edges = torch.from_numpy(cv.Canny(out.numpy(), threshold1=100, threshold2=200))
            out = saturation(blending(out, c_edges, a=0.5))
        if process_mode == 2:
            # Processing n.3
            out = sharpening(out)
            out = contrast_stretching(out, bgr_mins, bgr_maxs, bgr_mins_1, bgr_maxs_1)
        if process_mode == 3:
            # Processing n.5
            out = sharpening(out)
            out = contrast_stretching(out, bgr_mins, bgr_maxs, bgr_mins_1, bgr_maxs_1)
            out = frei_and_chen_edges(out)
        if process_mode == 4:
            # Processing n. 6
            out = out.type(torch.uint8)
            out = torch.from_numpy(cv.Canny(out.numpy(), threshold1=100, threshold2=200))

    if rotate and not rotate_first:
        out = correct_angle(out, threshold=270, verbose=verbose, save_images=save_images)

    # 4. Returning final image
    return out


if __name__ == '__main__':
    # Creating temporary directory
    if not os.path.exists(TMP_DIR):
        print(f"WARNING! Temp directory not found, creating at {TMP_DIR} ...")
        os.mkdir(TMP_DIR)
    # Importing an image file from dataset
    file = FILE_NAME
    img = cv.imread(file)
    if img is None:
        print(f"ERROR! Impossible to read image from file {file}.")
        input("Press any key to exit.")
        exit(1)

    f, ax = plt.subplots(6, 2, figsize=(12, 20))
    f.tight_layout()

    # 0 - ORIGINAL IMAGE
    ax[0][0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    ax[0][0].set_title('0. Original image')

    # Convert to Torch tensor (c, h, w)
    # img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    img = torch.from_numpy(img)

    # Intervals for contrast stretching
    bgr_mins = [0, 0, 0]
    bgr_maxs = [180, 180, 180]
    bgr_mins_1 = [0, 0, 0]
    bgr_maxs_1 = [255, 255, 255]

    # 1 - CONTRAST STRETCHING
    out_1 = contrast_stretching(img, bgr_mins, bgr_maxs, bgr_mins_1, bgr_maxs_1)
    ax[0][1].imshow(cv.cvtColor(out_1.numpy(), cv.COLOR_BGR2RGB))
    ax[0][1].set_title('1. Contrast stretching')

    # 2 - SHARPENING
    out_2 = sharpening(img)
    ax[1][0].imshow(cv.cvtColor(out_2.numpy(), cv.COLOR_BGR2RGB))
    ax[1][0].set_title('2. Sharpening')

    # 3 - SHARPENING + CONTRAST STRETCHING
    out_3 = contrast_stretching(out_2, bgr_mins, bgr_maxs, bgr_mins_1, bgr_maxs_1)
    ax[1][1].imshow(cv.cvtColor(out_3.numpy(), cv.COLOR_BGR2RGB))
    ax[1][1].set_title('3. Sharpening + Contrast stretching')

    # 4 - FREI & CHEN EDGE DETECTION
    out_4 = frei_and_chen_edges(img)
    ax[2][0].imshow(cv.cvtColor(out_4.numpy(), cv.COLOR_BGR2RGB))
    ax[2][0].set_title('4. Frei & Chen edges')

    # 5 - SHARPENING + CONTRAST STRETCHING + FREI & CHEN EDGES
    out_5 = frei_and_chen_edges(out_3)
    ax[2][1].imshow(cv.cvtColor(out_5.numpy(), cv.COLOR_BGR2RGB))
    ax[2][1].set_title('5. Sharpening + Contrast stretching + Frei & Chen edges')

    # 6 - CANNY EDGE DETECTION
    out_6 = cv.Canny(img.numpy(), threshold1=100, threshold2=200)
    out_6 = torch.from_numpy(out_6)
    ax[3][0].imshow(out_6.numpy(), cmap='gray')
    ax[3][0].set_title('6. Canny edges')

    # 7 - NEGATIVE CANNY EDGES
    out_7 = negative(out_6)
    ax[3][1].imshow(cv.cvtColor(out_7.numpy(), cv.COLOR_BGR2RGB))
    ax[3][1].set_title('7. Negative Canny edges')

    # 8 - SATURATED IMAGE BLENDING with out 6
    out_8 = saturation(blending(img, negative(out_6), a=0.9))
    ax[4][0].imshow(cv.cvtColor(out_8.numpy(), cv.COLOR_BGR2RGB))
    ax[4][0].set_title('8. Saturated blending: Original + 7 (a=0.9)')

    # 9 - SATURATED IMAGE BLENDING with out 5
    out_9 = saturation(blending(img, out_5, a=0.5))
    ax[4][1].imshow(cv.cvtColor(out_9.numpy(), cv.COLOR_BGR2RGB))
    ax[4][1].set_title('9. Saturated blending: Original + 5 (a=0.5)')

    # 10 - SATURATED IMAGE BLENDING with Frei & Chen edges
    fc_edges = frei_and_chen_edges(img)
    out_10 = saturation(blending(img, fc_edges, a=0.5))
    ax[5][0].imshow(cv.cvtColor(out_10.numpy(), cv.COLOR_BGR2RGB))
    ax[5][0].set_title('10. Saturated blending: Original + Frei & Chen edges (a=0.5)')

    # 11 - SATURATED IMAGE BLENDING with Canny edges
    c_edges = torch.from_numpy(cv.Canny(img.numpy(), threshold1=100, threshold2=200))
    out_11 = saturation(blending(img, c_edges, a=0.5))
    ax[5][1].imshow(cv.cvtColor(out_11.numpy(), cv.COLOR_BGR2RGB))
    ax[5][1].set_title('11. Saturated blending: Original + Canny edges (a=0.5)')

    # Saving image processing synthetic table
    plt.savefig(TMP_DIR + 'image_processing_table.jpg')
    plt.show()

    # FINAL IMAGE PROCESSING TEST
    # cv.imshow("Input image", img.numpy())
    processed_img = process_image(filename=file, process_mode=PROCESS_MODE, save_images=True)
    # cv.imshow("Processed image", processed_img.numpy())
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Saving image input and image output
    cv.imwrite(TMP_DIR+'original.jpg', img.numpy())
    cv.imwrite(TMP_DIR+'processed.jpg', processed_img.numpy())
