import numpy as np
import matplotlib.pyplot as plt
import cv2
result = np.loadtxt('result.txt')
img = result.astype('int')
cv2.imshow('scrambled_img', np.uint8(img))
cv2.waitKey(0)
cv2.imwrite('results/comparison2.png', np.uint8(img))