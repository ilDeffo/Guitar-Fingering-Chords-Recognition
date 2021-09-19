"""
    Image processing & geometric based main script
    for "Guitar Fingering & Chords Recognition" project.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch


def process(filename):
    """
        * FUNCTION TO CALL FOR IMAGE PROCESSING OF A SINGLE FILE *
        The experiments showed that the actual better image processing is:
        Saturated blending betweeen the original image and the Frei & Chen edges.
    """
    img = cv.imread(filename)
    # Convert image to Torch tensor (h, w, c)
    img = torch.from_numpy(img)
    # Calling processing methods
    edges = frei_and_chen_edges(img)
    out = saturation(blending(img, edges, a=0.5))
    # Returning final image
    return out


# ATTENTION! Each of the following methods accepts img as BGR PyTorch image tensor
#            of shape (h, w, c).


def blending(img1, img2, a):
    if a < 0 or a > 1:
        print("ATTENTION! The parameter 'a' must be in [0, 1] interval. Skipping blending ...")
        return img
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


def frei_and_chen_edges(img):
    img = img.numpy()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    frei_and_chen_x = np.array([[-1, 0, 1], [-np.sqrt(2), 0, np.sqrt(2)], [-1, 0, 1]], dtype=np.float32)
    frei_and_chen_y = np.array([[-1, -np.sqrt(2), -1], [0, 0, 0], [1, np.sqrt(2), 1]], dtype=np.float32)
    frei_and_chen_max_magnitude_value = 1232

    M, D = get_gradient(img_gray.astype(np.float32), frei_and_chen_x, frei_and_chen_y)
    out = get_grayscale_image_from_gradient(M, frei_and_chen_max_magnitude_value,
                                            use_maximum_array_value=True, color_image=False)

    out = torch.from_numpy(out).type(torch.uint8)
    return out


if __name__ == '__main__':
    # Import an image file from dataset
    file = 'Dataset/A/A (18).jpeg'
    img = cv.imread(file)
    f, ax = plt.subplots(6, 2, figsize=(12, 20))
    f.tight_layout()
    ax[0][0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    ax[0][0].set_title('0. Original image')

    # Convert to Torch tensor (c, h, w)
    # img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    img = torch.from_numpy(img)

    # 1 - CONTRAST STRETCHING
    out_1 = contrast_stretching(img)
    ax[0][1].imshow(cv.cvtColor(out_1.numpy(), cv.COLOR_BGR2RGB))
    ax[0][1].set_title('1. Contrast stretching')

    # 2 - SHARPENING
    out_2 = sharpening(img)
    ax[1][0].imshow(cv.cvtColor(out_2.numpy(), cv.COLOR_BGR2RGB))
    ax[1][0].set_title('2. Sharpening')

    # 3 - SHARPENING + CONTRAST STRETCHING
    out_3 = contrast_stretching(out_2)
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

    plt.show()

    # FINAL IMAGE PROCESSING TEST
    cv.imshow("Input image", img.numpy())
    processed_img = process(file)
    cv.imshow("Processed image", processed_img.numpy())
    cv.waitKey(0)
    cv.destroyAllWindows()
