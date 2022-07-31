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
    
    thresh1 = np.reshape(thresh1, (weight, height, 1))
    output = thresh1.astype(np.float32)
    output = output.transpose(2, 0, 1)
    output = torch.from_numpy(output.astype(np.float32))
    
    
    if torch.cuda.is_available():
        output = output.cuda()
    
    return output



