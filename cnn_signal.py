import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def conv(img,fil):
    fil_x,fil_y=fil.shape
    x,y=img.shape
    res_x=x-fil_x
    res_y=y-fil_y
    res=np.zeros((res_x,res_y))
    for i in range(res_x):
        for j in range(res_y):
            seg=img[i:(i+fil_x),j:(j+fil_y)]
            res[i][j]=np.sum(np.multiply(fil,seg))
    return res

def thresh(img,val):
    m,n=img.shape
    new=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if img[i][j]>val:
                new[i][j]=255
            else:
                new[i][j]=0
    return new

img=cv2.imread('flower.jpg',0)

#sobel_x=1.2*np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
sobel_x=(1/9)*(np.array([[0.5,1,0,-1,-0.5],
                               [1,2,0,-2,-1],
                               [1.5,3,0,-3,-1.5],
                               [1,2,0,-2,-1],
                               [0.5,1,0,-1,-0.5]]))
resx=conv(img,sobel_x)
resy=conv(img,sobel_x.T)
res=np.zeros_like(resx)

res=np.sqrt(np.add(np.square(resx),np.square(resy)))

res_thresh=thresh(res,40)
res_thresh2=thresh(res,80)
plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(img,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(res,cmap='gray')
plt.subplot(2,2,3)
plt.imshow(res_thresh,cmap='gray')
plt.subplot(2,2,4)
plt.imshow(res_thresh2,cmap='gray')
plt.show()
cv2.imwrite('edge.jpg',res)

