import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(5)
'''
对加密图像进行处理，返回处理后的结果
'''
def occlusion(img):
  img=cv2.imread(img)
  h,w,_=img.shape
  # B,G,R=cv2.split(img)
  #随机移除R通道80*80大小的像素
  #产生随机整数，裁剪大小为80
  # pos_w=np.random.randint(0,w-80)
  # pos_h=np.random.randint(0,h-80)
  # for i in range(80):
  #   for j in range(80):
  #     img[pos_h+i][pos_w+j]=0
  #随机移除G通道50*80的像素
  pos_w=np.random.randint(0,w-50)
  pos_h=np.random.randint(0,h-80)
  for i in range(80):
    for j in range(50):
      img[pos_h+i][pos_w+j]=0
  #三通道合并
  # im=cv2.merge([R,G,B])
  #随机移除All通道60*50的像素
  # pos_w=np.random.randint(0,w-60)
  # pos_h=np.random.randint(0,h-50)
  # for i in range(50):
  #   for j in range(60):
  #     img[pos_h+i][pos_w+j]=np.array([0,0,0])

  return img


def main():
  img1='results/imgXOR.png'
  lossy_encrypt=occlusion(img1)
  plt.imshow(lossy_encrypt)
  plt.show()
  #保存处理后的有损加密图像,opencv的颜色通道顺序为[B,G,R]
  # cv2.imwrite('./block2.png',lossy_encrypt[:,:,(2,1,0)])
 
if __name__== '__main__':
  main()