import numpy as np
import cv2
import star as a


cap = cv2.imread("asset/pic1.png")
gray = cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)
grayRes = cv2.resize(gray,(0,0),fx=0.3 , fy=0.3)
cv2.imwrite("GrayResize.png",grayRes,)
obsticle = []
sx = int(504)
sy = int(346)
gx = int(436)
gy = int(106)

start = (sx,sy)
goal = (gx,gy)
path = a.astar(gray.shape,start,goal,obsticle)
if path:
    a.cvDraw(cap,path)
else:
    print("No path found!")

cv2.imshow("picture",cap)
cv2.imshow("picture2",grayRes)
cv2.waitKey(0)
