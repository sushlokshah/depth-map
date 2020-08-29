import numpy as np
import cv2 as cv 
from matplotlib import pyplot as plt

# calibration matrix

pl = np.zeros([3,4])
pl[0][0] =640
pl[0][1] = 0
pl[0][2] =640
pl[0][3] =2176
pl[1][0] =0
pl[1][1] =480
pl[1][2] =480
pl[1][3] =552
pl[2][0] =0
pl[2][1] = 0
pl[2][2] = 1
pl[2][3] = 1.4

kl,rl,tl,a,d,f,g = cv.decomposeProjectionMatrix(pl)
print(kl)
print(rl)
for i in range(4):
	tl[i][0] = tl[i][0]/tl[3][0]  
print(tl)

pr = np.zeros([3,4])
pr[0][0] =640
pr[0][1] = 0
pr[0][2] =640
pr[0][3] =2176
pr[1][0] =0
pr[1][1] =480
pr[1][2] =480
pr[1][3] =792
pr[2][0] =0
pr[2][1] = 0
pr[2][2] = 1
pr[2][3] = 1.4

kr,rr,tr,a,d,f,g = cv.decomposeProjectionMatrix(pr)
print(kr)
print(rr)
for i in range(4):
	tr[i][0] = tr[i][0]/tr[3][0]  
print(tr)
translation = tl-tr
print(translation)

#loading img
imgL = cv.imread('left.png',0)

imgR = cv.imread('right.png',0)

#loading template
temp = cv.imread("SAVE_20200826_120954.jpg",0)
n,m = temp.shape
#functiopn for finding location of template
def correlation_function(img,temp):
	img2 = img.astype(np.float32)
	img = img.astype(np.float32)
	imgmean = np.mean(img)
	x,y = img.shape
	img = img - imgmean*np.ones([x,y])
	temp = temp.astype(np.float32)
	n,m = temp.shape
	temp = temp - np.ones([n,m])*(np.mean(temp))
	sumt = np.sum(np.square(temp))
	position = np.zeros([(x-n+1),(y-m+1)],np.float32)

	for i in range(x-n+1):
		for j  in range(y-m+1):
			patch = img2[i:i+n,j:j+m]  - np.mean(img2[i:i+n,j:j+m])
			sump = np.sum(np.square(patch))
			mat = temp*img[i:i+n,j:j+m]
			sum = np.sum(mat)
			#print(sum)
			position[i][j] =sum/((sump*sumt)**0.5)
		print(i)
	min_val, max_val, min_loc, max_loc = cv.minMaxLoc(position)
	return position ,max_val,max_loc
	
#finding the coordinate of the point of match
location1,val2,point1 = correlation_function(imgR,temp)
plt.imshow(location1,'gray')
plt.show()	
print(point1)
x1,y1=point1
cox =int(x1+(n-1)/2)
coy =int(y1+(m-1)/2)
#finding disparity
stereo = cv.StereoSGBM_create(minDisparity = 0,numDisparities= 100, blockSize = 9,P1= 24*9*9,P2=32*9*9)

disparity = stereo.compute(imgL,imgR).astype(np.float32)/60
plt.imshow(disparity,'jet')
plt.show()
x,y = disparity.shape

#finding depth
depth = np.zeros([x,y]).astype(np.float32)
for i in range(x):
	for j in range(y):
		if disparity[i][j] != 0:
			depth[i][j] = (translation[1][0]*kr[1][1])/disparity[i][j]
			
		
		else:
			depth[i][j] = (translation[1][0]*kr[1][1])/0.5
depth = depth.astype(np.float32)
plt.imshow(depth,'gray')
print(depth[cox][coy])
plt.show()
print(disparity[cox][coy])
print(depth[cox][coy])





















