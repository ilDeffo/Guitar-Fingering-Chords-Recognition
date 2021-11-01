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
import os

from Utils.processing_operators import frei_and_chen_edges

TMP_DIR = "Temp" + os.sep


def line_points(line, offset):
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + offset * -b)
    y1 = int(y0 + offset * a)
    x2 = int(x0 - offset * -b)
    y2 = int(y0 - offset * a)
    return (x1, y1), (x2, y2)


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def angle_axis_x(line):
    rho, theta = line[0]
    # Converting to degrees
    theta = theta * 180 / np.pi
    angle = theta - 90  # Real angle respect to x axis
    return angle


def filter_lines(lines):
    if lines is None:
        return lines
    # Let's filter the perpendicular lines with offsetof 40 degrees
    offset = 30
    '''
    exclude_interval_1 = [90 - offset, 90 + offset]
    exclude_interval_2 = [270 - offset, 270 + offset]
    '''
    exclude_interval_1 = [-90 - offset, -90 + offset]
    exclude_interval_2 = [90 - offset, 90 + offset]
    filtered_lines = None
    for l in lines:
        p1, p2 = line_points(l, offset=50)
        # angle = angle_between(p1, p2)
        angle = angle_axis_x(l)

        if exclude_interval_1[0] <= angle <= exclude_interval_1[1]\
                or exclude_interval_2[0] <= angle <= exclude_interval_2[1]:
            continue
        else:
            if filtered_lines is None:
                filtered_lines = np.copy(l)
                filtered_lines = filtered_lines[np.newaxis, :]
            else:
                filtered_lines = np.concatenate(
                    (filtered_lines, l[np.newaxis, :]), axis=0)

    return filtered_lines


def correct_angle(img, threshold=270, verbose=True, save_images=True):
    """
    Method using HoughLines to detect strings and rotate image in order to have them parallel
    to x axis. Black borders caused by rotation are cropped.

    :param img: BGR PyTorch image tensor of shape (h, w, c).
    :param threshold: Threshold to use in lines detection.
    :param verbose:
    :param save_images: If true, the algorithm saves image results in TMP_DIR.
    :return: img with angle correction.
    """
    # Creating temporary directory
    if not os.path.exists(TMP_DIR):
        print(f"WARNING! Temp directory not found, creating at {TMP_DIR} ...")
        os.mkdir(TMP_DIR)

    img = img.numpy()

    #edges = cv.Canny(img, 100, 200)
    # 140 (hand cropped image) or 250 (original) threshold value is suggested for Canny edges
    # lines = cv.HoughLines(edges, 1, np.pi / 180, threshold)
    edges = frei_and_chen_edges(torch.from_numpy(img)).numpy()
    # 200 (hand cropped image) or 700 (original) threshold value is suggested for Frei & Chen edges
    # -> Frei & Chen edges with this value is the most robust choice!
    lines = cv.HoughLines(edges, 1, np.pi / 180, threshold)

    # Removing the lines with a too much perpendicular angular value
    lines = filter_lines(lines)

    # Let's try different thresholds to find lines!
    attempts = 30
    if lines is None:
        for i in range(1, attempts + 1):
            threshold = threshold - 20
            if threshold <= 20:
                print(
                    f"WARNING! No lines found in the image {img.shape} after {attempts-1} attempts "
                    f"with final threshold {threshold+20*i}. Skipping angular correction...")
                return torch.from_numpy(img)

            lines = cv.HoughLines(edges, 1, np.pi / 180, threshold)
            # Removing the lines with a too much perpendicular angular value
            lines = filter_lines(lines)

            if lines is not None:
                break
            if lines is None and i == attempts:
                print(
                    f"WARNING! No lines found in the image {img.shape} after {attempts} attempts "
                    f"with final threshold {threshold}. Skipping angular correction...")
                return torch.from_numpy(img)

    # Let's optimize the lines: if they are too many respect the strings we have to increment threshold.
    optimized_lines = None
    optimized_threshold = threshold
    if len(lines) > 6:
        for i in range(1, attempts + 1):
            if len(lines) > 6:
                optimized_threshold = optimized_threshold + 20
                optimized_lines = cv.HoughLines(edges, 1, np.pi / 180, optimized_threshold)

                # Removing the lines with a too much perpendicular agnular value
                optimized_lines = filter_lines(optimized_lines)

            if optimized_lines is not None:
                if len(optimized_lines) <= 6:
                    lines = optimized_lines
                    threshold = optimized_threshold
                    break

    if verbose:
        print(f"{len(lines)} lines found in the image {img.shape} with threshold {threshold}!")

    drawed_img = np.copy(img)
    # Taking diagonal of img as max length of edge
    max_l = math.sqrt(img.shape[0]**2 + img.shape[1]**2)
    angles = []
    for line in lines:
        p1, p2 = line_points(line, offset=max_l)
        angles.append(angle_between(p1, p2))
        # Drawing line on image
        cv.line(drawed_img, p1, p2, (0, 0, 255), 2)

    if save_images:
        cv.imwrite(TMP_DIR+'houghlines.jpg', drawed_img)

    # Considering angle of rotation as the median of lines' thetas
    median_theta = np.median(lines[:, :, 1])
    # Converting to degrees
    median_theta = median_theta * 180 / np.pi
    angle = median_theta - 90     # Real angle of rotation

    h = img.shape[0]
    w = img.shape[1]
    (cX, cY) = (w / 2, h / 2)
    # Rotating image around the center of the image
    R = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv.warpAffine(img, R, (w, h))

    if save_images:
        cv.imwrite(TMP_DIR+'rotated_with_black_borders.jpg', rotated)

    # Cropping image to remove black borders generated by the rotation
    rotated_cropped = crop_around_center(
        rotated,
        *largest_rotated_rect(
            w,
            h,
            math.radians(angle)
        )
    )
    if save_images:
        cv.imwrite(TMP_DIR+'rotated.jpg', rotated_cropped)

    out = torch.from_numpy(rotated_cropped)
    return out


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    The full proof of the algorithm is available at
    https://newbedev.com/rotate-image-and-crop-out-black-bordersOriginal
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around its centre point.
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if width > image_size[0]:
        width = image_size[0]

    if height > image_size[1]:
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]
