"""
Classical image processing operators and edge detectors used
for "Guitar Fingering & Chords Recognition" project.

ATTENTION! Each of the following methods accepts img as BGR PyTorch image tensor
           of shape (h, w, c).
"""

import cv2 as cv
import numpy as np
import torch


def blending(img1, img2, a):
    if a < 0 or a > 1:
        print("ATTENTION! The parameter 'a' must be in [0, 1] interval. Skipping blending ...")
        return img1
    img1 = img1.type(torch.float32)
    img2 = img2.type(torch.float32)
    if len(img1.shape) != 3:
        img1 = img1.unsqueeze(2)
    if len(img2.shape) != 3:
        img2 = img2.unsqueeze(2)
    out = a * img1 + (1 - a) * img2
    return out.type(torch.uint8)


def negative(img):
    out = -1 * img + 255
    return out


def saturation(img):
    img = img.type(torch.float32)
    out = img * 1.8 - 10
    out = torch.round(out).clip(0, 255)
    return out.type(torch.uint8)


def contrast_stretching(img):
    img = img.type(torch.float32)
    if len(img.shape) != 3:
        img = img.unsqueeze(2)
    out = torch.clone(img)

    # Loop over the image channels and apply Min-Max contrast stretching
    bgr_mins = [0, 0, 0]
    bgr_maxs = [180, 180, 180]

    for c in range(img.shape[2]):
        out[:, :, c][out[:, :, c] <= bgr_mins[c]] = bgr_mins[c]
        out[:, :, c][out[:, :, c] >= bgr_maxs[c]] = bgr_maxs[c]

        mask = (out[:, :, c] > bgr_mins[c]) & (out[:, :, c] < bgr_maxs[c])
        scale_factor = (bgr_maxs[c] - bgr_mins[c]) / (255 - 0)

        out[:, :, c][mask] = (out[:, :, c][mask] - 0) * scale_factor + bgr_mins[c]

    return out.type(torch.uint8)


def sharpening(img):
    img = img.numpy().astype(np.float32)
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    out = cv.filter2D(img, -1, kernel)
    out = np.around(out)
    out = np.clip(out, 0, 255)

    out = torch.from_numpy(out).type(torch.uint8)
    return out


def get_gradient(im, kernel_x, kernel_y):
    # Applying kernel gradient masks
    Gx = cv.filter2D(im, -1, kernel_x)
    Gy = cv.filter2D(im, -1, kernel_y)
    # Getting magnitude and direction
    M = np.sqrt(Gx ** 2 + Gy ** 2)
    D = np.arctan2(Gy, Gx)
    return M, D


def get_grayscale_image_from_gradient(M, max_magnitude_value, use_maximum_array_value=False, color_image=True):
    # Even if we could divide by the maximum possible value, it's better to divide by the actual maximum.
    # In this way, the contours of image are more evident.
    if use_maximum_array_value:
        if color_image:
            for i in range(3):
                M[:, :, i] = M[:, :, i] / np.max(M[:, :, i]) * 255
        else:
            M = M / np.max(M) * 255
    else:
        M = M / max_magnitude_value * 255
    M = np.clip(np.around(M), 0, 255).astype(np.uint8)

    return M


def frei_and_chen_edges(img, threshold1=40, threshold2=255):
    img = img.numpy()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    frei_and_chen_x = np.array([[-1, 0, 1], [-np.sqrt(2), 0, np.sqrt(2)], [-1, 0, 1]], dtype=np.float32)
    frei_and_chen_y = np.array([[-1, -np.sqrt(2), -1], [0, 0, 0], [1, np.sqrt(2), 1]], dtype=np.float32)
    frei_and_chen_max_magnitude_value = 1232

    M, D = get_gradient(img_gray.astype(np.float32), frei_and_chen_x, frei_and_chen_y)
    out = get_grayscale_image_from_gradient(M, frei_and_chen_max_magnitude_value,
                                            use_maximum_array_value=True, color_image=False)

    # Final edge thresholding
    out[threshold1 <= out.all() <= threshold2] = 1
    out[out < threshold1] = 0
    out[out > threshold2] = 0

    out = torch.from_numpy(out).type(torch.uint8)
    return out
