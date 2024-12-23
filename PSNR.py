import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def PSNR(img1,img2):
  img1=cv2.imread(img1)
  img2=cv2.imread(img2)
  w,h,_=img1.shape
  B1,G1,R1=cv2.split(img1)
  B2,G2,R2=cv2.split(img2)

  #强制转换元素类型，为了运算
  R1=R1.astype(np.int32)
  R2=R2.astype(np.int32)
  G1=G1.astype(np.int32)
  G2=G2.astype(np.int32)
  B1=B1.astype(np.int32)
  B2=B2.astype(np.int32)

  #计算均方误差,初始化64位无符号整型，防止累加中溢出
  R_mse=np.uint64(0)
  G_mse=np.uint64(0)
  B_mse=np.uint64(0)
  for i in range(w):
    for j in range(h):
      R_mse+=(R1[i][j]-R2[i][j])**2
      G_mse+=(G1[i][j]-G2[i][j])**2
      B_mse+=(B1[i][j]-B2[i][j])**2
  R_mse/=(w*h)
  G_mse/=(w*h)
  B_mse/=(w*h)
  R_psnr=10*math.log((255**2)/R_mse,10)
  G_psnr=10*math.log((255**2)/G_mse,10)
  B_psnr=10*math.log((255**2)/B_mse,10)

  return R_psnr,G_psnr,B_psnr


def main():
  img='results/plain1.png'
  img1='results/comparison1.png'
  # img2='./Lena_encrypt2.png'
  #计算PSNR
  R_psnr,G_psnr,B_psnr=PSNR(img,img1)
  print('********峰值信噪比*********')
  print(R_psnr,G_psnr,B_psnr)
  print('通道R:{:.4}'.format(R_psnr))
  print('通道G:{:.4}'.format(G_psnr))
  print('通道B:{:.4}'.format(B_psnr))


if __name__== '__main__':
  main()