import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
绘制灰度直方图
'''
def hist(img):
  img=cv2.imread(img)
  B,G,R=cv2.split(img)
  #转成一维
  R=R.flatten(order='C')
  G=G.flatten(order='C')
  B=B.flatten(order='C')


  #结果展示
  plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
  plt.subplot(232)
  # plt.imshow(img[:,:,(2,1,0)])
  plt.hist(img.flatten(order='C'),bins=range(257),color='gray')
  plt.title('原图像')
  #子图2，通道R
  plt.subplot(234)
  #imshow()对图像进行处理，画出图像，show()进行图像显示
  plt.hist(R,bins=range(257),color='red')
  plt.title('通道R')
  # plt.show()
  #不显示坐标轴
  # plt.axis('off')

  #子图3，通道G
  plt.subplot(235)
  plt.hist(G,bins=range(257),color='green')
  plt.title('通道G')
  # plt.show()
  # plt.axis('off')

  #子图4，通道B
  plt.subplot(236)
  plt.hist(B,bins=range(257),color='blue')
  plt.title('通道B')
  # plt.axis('off')
  # #设置子图默认的间距
  plt.tight_layout()
  plt.show()


def main():
  img='results/SSIM-C.png'
  
  hist(img)

if __name__== '__main__':
  img='results/en3.png'
  img=cv2.imread(img)
  plt.hist(img.flatten(),bins=range(257),color='blue', rwidth=0.8)
  plt.savefig("hist-cipher3.png",dpi=1000,bbox_inches = 'tight')
  plt.show()
  # main()