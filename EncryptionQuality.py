import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

'''
计算加密质量，img1是原图，img2是加密图像
'''
def EQ(img1,img2):
  img1=cv2.imread(img1)
  img2=cv2.imread(img2)
  w,h,_=img1.shape
  B1,G1,R1=cv2.split(img1)
  B2,G2,R2=cv2.split(img2)
  R1_H={}
  R2_H={}
  G1_H={}
  G2_H={}
  B1_H={}
  B2_H={}
  R_EQ=0
  G_EQ=0
  B_EQ=0
  for i in range(256):
    R1_H[i]=0
    R2_H[i]=0
    G1_H[i]=0
    G2_H[i]=0
    B1_H[i]=0
    B2_H[i]=0

  for i in range(w):
    for j in range(h):
      R1_H[R1[i][j]]+=1;
      R2_H[R2[i][j]]+=1;
      G1_H[G1[i][j]]+=1;
      G2_H[G2[i][j]]+=1;
      B1_H[B1[i][j]]+=1;
      B2_H[B2[i][j]]+=1;

  for i in range(256):
    #公式里是平方，待考虑
    R_EQ+=abs(R1_H[i]-R2_H[i])
    G_EQ+=abs(G1_H[i]-G2_H[i])
    B_EQ+=abs(B1_H[i]-B2_H[i])
  # print(R_EQ)
  R_EQ/=256
  G_EQ/=256
  B_EQ/=256
  return R_EQ,G_EQ,B_EQ


def main():
  img='results/plain1.png'
  img1='results/comparison1.png'
  #计算加密质量
  R_EQ,G_EQ,B_EQ=EQ(img,img1)
  print('***********EQ*********')
  print(R_EQ,G_EQ,B_EQ)
  print('通道R:{:.0f}'.format(R_EQ))
  print('通道G:{:.0f}'.format(G_EQ))
  print('通道B:{:.0f}'.format(B_EQ))

if __name__== '__main__':
  main()