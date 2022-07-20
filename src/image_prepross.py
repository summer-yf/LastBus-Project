import cv2
import torch
# prepross image from 512x218 to 84x84 Grayscale
# everything greater than 1 is set to 255
# for easier use in training


def preprocess(image):
    weight = 84
    height = 84
    image_color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dim = (weight, height)
    image_1 = cv2.resize(image, dim)
    ret, thresh1 = cv2.threshold(image_1, 1, 255, cv2.THRESH_BINARY)

    return image[None, :, :].astype(np.float32)

# 84x84x4 image(probably can change 4 for further tests)
def net_input(image):
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        image = image.cuda()
    return torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
