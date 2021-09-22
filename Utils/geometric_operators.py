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
    # 700 value is suggested for Frei & Chen edges
    # -> Frei & Chen edges with this value is most robust choice!
    lines = cv.HoughLines(edges, 1, np.pi / 180, 700)

    if lines is None:
        print(f"WARNING! No lines found in the image {img}. Skipping angular correction...")
        return torch.from_numpy(img)

    drawed_img = np.copy(img)
    # Taking diagonal of img as max length of edge
    max_l = math.sqrt(img.shape[0]**2 + img.shape[1]**2)
    # Drawing lines on image
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
    # TODO: Take the lines and do a mean of the theta param to adjust image rotation
    mean_theta = np.mean(lines[:, :, 1])
    # Convert to degrees
    mean_theta = mean_theta * 180 / np.pi
    angle = mean_theta - 90     # Angle of rotation

    h = img.shape[0]
    w = img.shape[1]
    (cX, cY) = (w / 2, h / 2)
    # Rotate image around the center of the image
    R = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv.warpAffine(img, R, (w, h))
    # TODO: Crop black borders
    cv.imwrite('rotated.jpg', rotated)

    out = torch.from_numpy(img)
    return out
