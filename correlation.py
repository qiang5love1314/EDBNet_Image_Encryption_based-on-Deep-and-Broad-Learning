import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
分别计算图像通道相邻像素的水平、垂直和对角线的相关系数并返回
'''
np.random.seed(0)

def RGB_correlation(channel,N):
  #计算channel通道
  h,w=channel.shape
  #随机产生pixels个[0,w-1)范围内的整数序列
  row=np.random.randint(0,h-1,N)
  col=np.random.randint(0,w-1,N)
  #绘制相邻像素相关性图,统计x,y坐标
  x=[]
  h_y=[]
  v_y=[]
  d_y=[]
  for i in range(N):
    #选择当前一个像素
    x.append(channel[row[i]][col[i]])
    #水平相邻像素是它的右侧也就是同行下一列的像素
    h_y.append(channel[row[i]][col[i]+1])
    #垂直相邻像素是它的下方也就是同列下一行的像素
    v_y.append(channel[row[i]+1][col[i]])
    #对角线相邻像素是它的右下即下一行下一列的那个像素
    d_y.append(channel[row[i]+1][col[i]+1])
  #三个方向的合到一起
  x=x*3
  y=h_y+v_y+d_y

  #结果展示
  # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
  # plt.scatter(x,y)
  # plt.show()

  #计算E(x)，计算三个方向相关性时，x没有重新选择也可以更改
  ex=0
  for i in range(N):
    ex+=channel[row[i]][col[i]]
  ex=ex/N
  #计算D(x)
  dx=0
  for i in range(N):
    dx+=(channel[row[i]][col[i]]-ex)**2
  dx/=N

  #水平相邻像素h_y
  #计算E(y)
  h_ey=0
  for i in range(N):
    h_ey+=channel[row[i]][col[i]+1]
  h_ey/=N
  #计算D(y)
  h_dy=0
  for i in range(N):
    h_dy+=(channel[row[i]][col[i]+1]-h_ey)**2
  h_dy/=N
  #计算协方差
  h_cov=0
  for i in range(N):
    h_cov+=(channel[row[i]][col[i]]-ex)*(channel[row[i]][col[i]+1]-h_ey)
  h_cov/=N
  h_Rxy=h_cov/(np.sqrt(dx)*np.sqrt(h_dy))

  #垂直相邻像素v_y
  #计算E(y)
  v_ey=0
  for i in range(N):
    v_ey+=channel[row[i]+1][col[i]]
  v_ey/=N
  #计算D(y)
  v_dy=0
  for i in range(N):
    v_dy+=(channel[row[i]+1][col[i]]-v_ey)**2
  v_dy/=N
  #计算协方差
  v_cov=0
  for i in range(N):
    v_cov+=(channel[row[i]][col[i]]-ex)*(channel[row[i]+1][col[i]]-v_ey)
  v_cov/=N
  v_Rxy=v_cov/(np.sqrt(dx)*np.sqrt(v_dy))

  #对角线相邻像素d_y
  #计算E(y)
  d_ey=0
  for i in range(N):
    d_ey+=channel[row[i]+1][col[i]+1]
  d_ey/=N
  #计算D(y)
  d_dy=0
  for i in range(N):
    d_dy+=(channel[row[i]+1][col[i]+1]-d_ey)**2
  d_dy/=N
  #计算协方差
  d_cov=0
  for i in range(N):
    d_cov+=(channel[row[i]][col[i]]-ex)*(channel[row[i]+1][col[i]+1]-d_ey)
  d_cov/=N
  d_Rxy=d_cov/(np.sqrt(dx)*np.sqrt(d_dy))

  return h_Rxy,v_Rxy,d_Rxy,x,y

'''
分别计算图像img的各通道相邻像素的相关系数，默认随机选取3000对相邻像素
'''
def correlation(img,N=3000):
  img=cv2.imread(img, cv2.IMREAD_GRAYSCALE)
  h,w=img.shape
  
  # B,G,R=cv2.split(img)
  R_Rxy=RGB_correlation(img,N)
  G_Rxy=RGB_correlation(img,N)
  B_Rxy=RGB_correlation(img,N)

  #结果展示
  # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
  # plt.subplot(221)
  # plt.imshow(img[:,(2,1,0)])
  # plt.title('原图像')
  #子图2
  # plt.subplot(222)
  # plt.scatter(R_Rxy[3],R_Rxy[4],s=1, c='blue')
  # plt.xlabel('Gary value of pixel location(x,y)')
  # plt.ylabel('Gary value of pixel location(x+1,y)')
  # plt.savefig("cipher-horizontal.png",dpi=1000,bbox_inches = 'tight')
  # # plt.show()

  #子图3
  # plt.subplot(223)
  # plt.scatter(G_Rxy[3],G_Rxy[4],s=1,c='blue')
  # plt.xlabel('Gary value of pixel location(x,y)')
  # plt.ylabel('Gary value of pixel location(x,y+1)')
  # plt.savefig("cipher-vertical.png",dpi=1000,bbox_inches = 'tight')
  
  #子图4
  # plt.subplot(224)
  # plt.scatter(B_Rxy[3],B_Rxy[4],s=1,c='blue')
  # plt.xlabel('Gary value of pixel location(x,y)')
  # plt.ylabel('Gary value of pixel location(x+1,y+1)')
  # plt.savefig("cipher-diagonal.png",dpi=1000,bbox_inches = 'tight')


  return R_Rxy[0:3],G_Rxy[0:3],B_Rxy[0:3]


def main():
  img='results/plain1.png'
  img1='results/comparison1.png'
  R_Rxy,G_Rxy,B_Rxy=correlation(img1)
  #输出结果保留四位有效数字
  print("******该图像的各通道各方向的相关系数为*****")
  print('通道\tHorizontal\tVertical\tDiagonal')
  print(' R    \t{:.4f}    {:.4f}    {:.4f}'.format(R_Rxy[0],R_Rxy[1],R_Rxy[2]))
  print(' G    \t{:.4f}    {:.4f}    {:.4f}'.format(G_Rxy[0],G_Rxy[1],G_Rxy[2]))
  print(' B    \t{:.4f}    {:.4f}    {:.4f}'.format(B_Rxy[0],B_Rxy[1],B_Rxy[2]))

if __name__== '__main__':
  main()
