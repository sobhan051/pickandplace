import numpy as np
import cv2
import star as a
import alist

startList = np.array([[0,0]])
goalList = np.array([[0,0]])
startList = np.delete(startList,[0],0)
goalList = np.delete(goalList,[0],0)

cap = cv2.imread("asset/pic1.png")
gray = cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)
hsv =  cv2.cvtColor(cap,cv2.COLOR_BGR2HSV)
grayRes = cv2.resize(gray,(0,0),fx=0.5 , fy=0.5)



red_lower = np.array([136, 87, 111], np.uint8) 
red_upper = np.array([180, 255, 255], np.uint8) 
red_mask = cv2.inRange(hsv, red_lower, red_upper)
kernel = np.ones((5, 5), "uint8") 
red_mask = cv2.dilate(red_mask, kernel) 
res_red = cv2.bitwise_and(cap, cap, 
                        mask = red_mask)  
contours, hierarchy = cv2.findContours(red_mask, 
                                        cv2.RETR_TREE, 
                                        cv2.CHAIN_APPROX_SIMPLE) 



for pic, contour in enumerate(contours): 
    area = cv2.contourArea(contour) 

    if(area > 300): 
        x, y, w, h = cv2.boundingRect(contour) 
        xCenter = int(x + w/2)
        yCenter = int(y + h/2)
        if area < 8000:
            # GO TO THIS OBJ
            cv2.circle(cap, (xCenter,yCenter), 2, (0,0,0), 2)
            xCenter = int(xCenter*0.5)
            yCenter = int(yCenter*0.5)
            startList = np.append(startList,[[xCenter,yCenter]],0)
            cv2.putText(cap, "Cube", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255))


        if area > 8000:
            # THEN GO TO THIS SPOT
            cv2.circle(cap, (xCenter,yCenter), 2, (0,0,0), 2)
            xCenter = int(xCenter*0.5)
            yCenter = int(yCenter*0.5)
            goalList = np.append(goalList,[[xCenter,yCenter]],0)
            cv2.putText(cap, "Spot", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))


for i in range(len(startList)):
    startPoint , endPoint, startWithchange = alist.listing(startList,goalList)
    startList = startWithchange
    startPoint = tuple(startPoint)    
    endPoint = tuple(endPoint) 
    
    obsticle = []
    #print(f"{startingPoint} {endingPoint} \n")
    path = a.astar(grayRes.shape,startPoint,endPoint,obsticle)
    if path:
        print(f"this is new path: {path} \n")
        a.cvDraw(cap,path)
    else:
        print("No path found!")


    cv2.imshow("picture",cap)
    if cv2.waitKey(0) & 0xFF == ord('q'): 
        cv2.destroyAllWindows() 

