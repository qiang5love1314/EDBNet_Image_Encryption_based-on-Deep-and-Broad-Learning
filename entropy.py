import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


'''
计算图像的信息熵
'''
def entropy(img):
  img=cv2.imread(img)
  w,h,_=img.shape
  B,G,R=cv2.split(img)
  gray,num1=np.unique(R,return_counts=True)
  gray,num2=np.unique(G,return_counts=True)
  gray,num3=np.unique(B,return_counts=True)
  R_entropy=0
  G_entropy=0
  B_entropy=0

  for i in range(len(gray)):
    p1=num1[i]/(w*h)
    p2=num2[i]/(w*h)
    p3=num3[i]/(w*h)
    R_entropy-=p1*(math.log(p1,2))
    G_entropy-=p2*(math.log(p2,2))
    B_entropy-=p3*(math.log(p3,2))
  return R_entropy,G_entropy,B_entropy

def main():
  img='results/plain1.png'
  img1='results/comparison1.png'
  # img2='./Lena_encrypt2.png'
  #图像lena的熵
  R_entropy,G_entropy,B_entropy=entropy(img)
  cipherEntropy,G_entropy,B_entropy = entropy(img1)
  print('***********信息熵*********')
  print('通道R:{:.5}'.format(R_entropy))
  print('通道G:{:.5}'.format(cipherEntropy))
  # print('通道B:{:.5}'.format(B_entropy))

if __name__== '__main__':
  main()