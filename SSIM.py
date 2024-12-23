from skimage.measure import compare_ssim
import cv2
import numpy as np

path1 = 'results/SSIM-B.png'
img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
img1 = np.expand_dims(img1, -1)

path2 = 'results/SSIM-C.png'
img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
img2 = np.expand_dims(img2, -1)

psnr = compare_ssim(img1, img2, 255, multichannel=True)
print(psnr)


# img3 = cv2.imread('results/plain5.png', cv2.IMREAD_GRAYSCALE)
# img3 = np.expand_dims(img3, -1)
# img3[128,128] = 32
# cv2.imwrite('results/plain5changeOneBit.png', (img3))