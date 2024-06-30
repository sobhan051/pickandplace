import cv2
import numpy as np
import star

startList = np.array([[0,0]])
goalList = np.array([[0,0]])
startList = np.delete(startList,[0],0)
goalList = np.delete(goalList,[0],0)



cap = cv2.VideoCapture("video2.mp4")

while (1):
    _ , frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hsv =  cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    red_lower = np.array([136, 87, 111], np.uint8) 
    red_upper = np.array([180, 255, 255], np.uint8) 
    red_mask = cv2.inRange(hsv, red_lower, red_upper)

    kernel = np.ones((5, 5), "uint8") 

    red_mask = cv2.dilate(red_mask, kernel) 
    res_red = cv2.bitwise_and(frame, frame, 
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
                startList = np.append(startList,[[xCenter,yCenter]],0)
                cv2.putText(frame, "Cube", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255))
                cv2.circle(frame, (xCenter,yCenter), 2, (0,0,0), 2)

            if area > 8000:
                # THEN GO TO THIS SPOT
                goalList = np.append(goalList,[[xCenter,yCenter]],0)
                cv2.circle(frame, (xCenter,yCenter), 2, (0,0,0), 2)
                cv2.putText(frame, "Spot", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))



    cv2.imshow("Multiple Color Detection in Real-TIme", frame) 
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        cv2.destroyAllWindows() 
        break




