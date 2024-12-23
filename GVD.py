import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

#计算图像分量的GVD,参数为加密前后的2个分量二维矩阵
def GVD_components(channel0,channel1):
  AN0=0
  AN1=0
  w,h=channel0.shape
  for i in range(1,w-1):
    for j in range(1,h-1):
      #加密前
      GN0=0
      GN0+=(channel0[i][j]-channel0[i-1][j])**2
      GN0+=(channel0[i][j]-channel0[i+1][j])**2
      GN0+=(channel0[i][j]-channel0[i][j-1])**2
      GN0+=(channel0[i][j]-channel0[i][j+1])**2
      GN0/=4
      #加密后
      GN1=0
      GN1+=(channel1[i][j]-channel1[i-1][j])**2
      GN1+=(channel1[i][j]-channel1[i+1][j])**2
      GN1+=(channel1[i][j]-channel1[i][j-1])**2
      GN1+=(channel1[i][j]-channel1[i][j+1])**2
      GN1/=4
      AN0+=GN0
      AN1+=GN1
  AN0/=((w-2)*(h-2))
  AN1/=((w-2)*(h-2))
  gvd=(AN1-AN0)/(AN1+AN0)
  return gvd

def GVD(img1,img2):
  img1=cv2.imread(img1)
  img2=cv2.imread(img2)
  w,h,_=img1.shape
  B1,G1,R1=cv2.split(img1)
  B2,G2,R2=cv2.split(img2)
  R1=R1.astype(np.int16)
  R2=R2.astype(np.int16)
  G1=G1.astype(np.int16)
  G2=G2.astype(np.int16)
  B1=B1.astype(np.int16)
  B2=B2.astype(np.int16)
  R_gvd=GVD_components(R1,R2)
  G_gvd=GVD_components(G1,G2)
  B_gvd=GVD_components(B1,B2)
  return R_gvd,G_gvd,B_gvd

def main():
  img='results/plain1.png'
  img1='results/comparison1.png'
  # img2='./Lena_encrypt2.png'
  #计算GVD
  R_gvd,G_gvd,B_gvd=GVD(img,img1)
  print('***********GVD*********')
  #print(R_gvd,G_gvd,B_gvd)
  print('通道R:{:.5}'.format(R_gvd))
  print('通道G:{:.4}'.format(G_gvd))
  print('通道B:{:.4}'.format(B_gvd))

if __name__== '__main__':
  main()