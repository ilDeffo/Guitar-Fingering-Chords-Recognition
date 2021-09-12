# Color histogram

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


# ATTENTION! Each of the following methods except img as RGB PyTorch image tensor.

def contrast_stretching(img, new_min, new_max):
    img = img.type(torch.float32)
    out = torch.clone(img)
    # Contrast stretching on each RGB channel
    for c in range(img.shape[0]):
        old_min = torch.min(img[c])
        old_max = torch.max(img[c])
        scale_factor = (new_max - new_min) / (old_max - old_min)

        out[c] = (img[c] - old_min)*scale_factor + new_min

    out = out.type(torch.uint8)
    return out


def sharpening(img):
    # RGB -> BGR
    img = img.permute(1, 2, 0).numpy()

    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    out = cv2.filter2D(img, -1, kernel)

    out = torch.from_numpy(out)
    print(out.shape)
    # BGR -> RGB
    out = out.permute(2, 0, 1)
    return out


if __name__ == '__main__':
    # Import an image from dataset
    img = cv2.imread('../Dataset/A/A (3).jpeg')
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    img = torch.from_numpy(img)

    # plt.imshow(img.permute(1, 2, 0).numpy())
    # plt.show()

    print("Input image shape: ", img.shape)
    print("Input image dtype: ", img.dtype)
    print(img)

    out = contrast_stretching(img, new_max=255, new_min=0)
    out = sharpening(img)

    print("Ouput image shape: ", out.shape)
    print("Output image dtype: ", out.dtype)
    print(out)

    # Displaying the images
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Input & Ouput images')
    ax[0].set_title('Input image'), ax[0].imshow(img.permute(1, 2, 0))
    ax[1].set_title('Output image'), ax[1].imshow(out.permute(1, 2, 0))
    ax[0].plot(), ax[1].plot()
    plt.show()

    cv2.imshow('Input image', img.permute(1, 2, 0).numpy())
    cv2.waitKey(0)
    cv2.imshow('Ouput image', out.permute(1, 2, 0).numpy())
    cv2.waitKey(0)
