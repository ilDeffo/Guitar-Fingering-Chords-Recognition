"""
Geometric based operators for perspective and angular correction
for "Guitar Fingering & Chords Recognition" project.

ATTENTION! Each of the following methods accepts img as BGR PyTorch image tensor
           of shape (h, w, c).
"""

import cv2 as cv
import numpy as np
import torch

from Utils.processing_operators import frei_and_chen_edges


def get_projective_matrix(img1, img2):
    # TODO: Use OpenCV method to get the projective transformation matrix
    pass


def correct_perspective(img, matrix):
    # TODO: Use OpenCV method to warp the image with the transformation matrix
    pass


def correct_angle(img):
    # TODO: Use HoughLines to detect strings and rotate image in order to have them parallel to x axis)
    img = img.numpy()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # edges = cv.Canny(gray, 100, 200)
    edges = frei_and_chen_edges(torch.from_numpy(img)).numpy()
    lines = cv.HoughLines(edges, 1, np.pi / 180, 550)
    if lines is None:
        return torch.from_numpy(img)

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imwrite('houghlines.jpg', img)

    # TODO: Take the first 6 lines and do a mean of the params to adjust image rotation

    out = torch.from_numpy(img)
    return out
