import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
def gauss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

#默认10%的椒盐噪声
def salt_and_pepper_noise(noise_img, proportion=0.1):
    height, width, _ = noise_img.shape
    num = int(height * width * proportion)  # 多少个像素点添加椒盐噪声
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img

def main():
  img1='results/plain1.png'
  im=cv2.imread(img1)
  gauss_img=gauss_noise(im,mean=0,var=0.01)
  salt_img=salt_and_pepper_noise(im,proportion=0.05)
  cv2.imwrite('./gauss_img.png',gauss_img)
  cv2.imwrite('./salt_img.png',salt_img)


if __name__== '__main__':
  main()