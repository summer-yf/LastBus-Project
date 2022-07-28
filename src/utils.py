import cv2
import torch
import numpy as np
# prepross image from 512x218 to 84x84 Grayscale
# everything greater than 1 is set to 255
# for easier use in training


def pre_processing(image):
    weight = 90
    height = 90
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dim = (weight, height)
    image_resize = cv2.resize(image, dim)
    _, thresh1 = cv2.threshold(image_resize, 1, 255, cv2.THRESH_BINARY)
    return thresh1.astype(np.float32)

# 84x84x4 image(probably can change 4 for further tests)
def net_input(image):
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        image = image.cuda()
    return torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

# def pre_processing(image, width, height):
#     image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
#     _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
#     return image[None, :, :].astype(np.float32)
