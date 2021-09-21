"""
Geometric based operators for perspective and angular correction
for "Guitar Fingering & Chords Recognition" project.

ATTENTION! Each of the following methods accepts img as BGR PyTorch image tensor
           of shape (h, w, c).
"""
import math

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

    # edges = cv.Canny(img, 100, 200)
    # 250 value is suggested for Canny edges
    # lines = cv.HoughLines(edges, 1, np.pi / 180, 250)
    edges = frei_and_chen_edges(torch.from_numpy(img)).numpy()
    # 750 value is suggested for Frei & Chen edges
    # -> Frei & Chen edges with this value is most robust choice!
    lines = cv.HoughLines(edges, 1, np.pi / 180, 700)

    if lines is None:
        return torch.from_numpy(img)

    drawed_img = np.copy(img)
    # Taking diagonal of img as max length of edge
    max_l = math.sqrt(img.shape[0]**2 + img.shape[1]**2)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + max_l * (-b))
        y1 = int(y0 + max_l * (a))
        x2 = int(x0 - max_l * (-b))
        y2 = int(y0 - max_l * (a))
        cv.line(drawed_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imwrite('houghlines.jpg', drawed_img)
    # cv.imwrite('houghlines.jpg', edges)

    # TODO: Take the first 6 lines and do a mean of the params to adjust image rotation

    out = torch.from_numpy(img)
    return out
