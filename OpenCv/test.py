# CREATE 6 LIST FOR 3 COLORS (GREEN, BLUE, RED) FIRST LIST STORE XCENTER AND Y CENTER OF START AND SECOND LIST KEEPS THE X,Y CENTER OF GOAL.
# make hsv bar to find the colors in the webcam and replace the lower and upper bounds. 
# find cube write obj and find spot and write spot for each color. <DONE>
# FIND THE CENTER POINT OF THE OBJ AND SPOT. <DONE>
# know the direction of car go from start to goal. 
# give the other color objs as a obsticle.

import numpy as np 
import cv2 
from heapq import heappop, heappush



def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(image_shape, start, goal, obstacles):
    rows, cols = image_shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        for neighbor in get_neighbors(current, rows, cols):
            if neighbor in obstacles:
                continue  # skip obstacles
            
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if neighbor not in [i[1] for i in open_set]:
                    heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # no path found

def get_neighbors(position, rows, cols):
    neighbors = [
        (position[0] - 1, position[1]),
        (position[0] + 1, position[1]),
        (position[0], position[1] - 1),
        (position[0], position[1] + 1)
    ]
    valid_neighbors = [(r, c) for r, c in neighbors if 0 <= r < rows and 0 <= c < cols]
    return valid_neighbors


def cvDraw(image, path):
    for r, c in path:
        cv2.circle(image,(r,c), 2, (0,255,0), 2)












imageFrame = cv2.imread("asset/pic1.png")


# Reading the video from the 
# webcam in image frames 


# Convert the imageFrame in 
# BGR(RGB color space) to 
# HSV(hue-saturation-value) 
# color space 
hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 
grayFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)


# Set range for red color and 
# define mask 
red_lower = np.array([136, 87, 111], np.uint8) 
red_upper = np.array([180, 255, 255], np.uint8) 
red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

# Set range for green color and 
# define mask 
green_lower = np.array([25, 52, 72], np.uint8) 
green_upper = np.array([102, 255, 255], np.uint8) 
green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

# Set range for blue color and 
# define mask 
blue_lower = np.array([94, 80, 2], np.uint8) 
blue_upper = np.array([120, 255, 255], np.uint8) 
blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 

# Morphological Transform, Dilation 
# for each color and bitwise_and operator 
# between imageFrame and mask determines 
# to detect only that particular color 
kernel = np.ones((5, 5), "uint8") 

# For red color 
red_mask = cv2.dilate(red_mask, kernel) 
res_red = cv2.bitwise_and(imageFrame, imageFrame, 
                        mask = red_mask) 

# For green color 
green_mask = cv2.dilate(green_mask, kernel) 
res_green = cv2.bitwise_and(imageFrame, imageFrame, 
                            mask = green_mask) 

# For blue color 
blue_mask = cv2.dilate(blue_mask, kernel) 
res_blue = cv2.bitwise_and(imageFrame, imageFrame, 
                        mask = blue_mask) 

# Creating contour to track red color 
contours, hierarchy = cv2.findContours(red_mask, 
                                    cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE) 

for pic, contour in enumerate(contours): 
    area = cv2.contourArea(contour) 
    if(area > 300): 
        x, y, w, h = cv2.boundingRect(contour) 
        imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                (x + w, y + h), 
                                (0, 0, 255), 2) 
        xCenter = int(x + w/2)
        yCenter = int(y + h/2)
        if area < 8000:
            # GO TO THIS OBJ
            print(xCenter,yCenter)
            cv2.putText(imageFrame, "Cube", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255))
            cv2.circle(imageFrame, (xCenter,yCenter), 2, (0,0,0), 2)
        if area > 8000:
            # THEN GO TO THIS SPOT
            print(xCenter,yCenter)
            cv2.circle(imageFrame, (xCenter,yCenter), 2, (0,0,0), 2)
            cv2.putText(imageFrame, "Spot", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))
    
# Creating contour to track green color 
contours, hierarchy = cv2.findContours(green_mask, 
                                    cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE) 

for pic, contour in enumerate(contours): 
    area = cv2.contourArea(contour) 

    if(area > 300): 
        x, y, w, h = cv2.boundingRect(contour) 
        imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                (x + w, y + h), 
                                (0, 255, 0), 2) 
        
        xCenter = int(x + w/2)
        yCenter = int(y + h/2)
        if area < 8000:
            # GO TO THIS OBJ
            
            cv2.putText(imageFrame, "Cube", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255))
            cv2.circle(imageFrame, (xCenter,yCenter), 2, (0,0,0), 2)
        if area > 8000:
            # THEN GO TO THIS SPOT
            
            cv2.circle(imageFrame, (xCenter,yCenter), 2, (0,0,0), 2)
            cv2.putText(imageFrame, "Spot", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))


# Creating contour to track blue color 
contours, hierarchy = cv2.findContours(blue_mask, 
                                    cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE) 
for pic, contour in enumerate(contours): 
    area = cv2.contourArea(contour) 

    if(area > 300): 
        x, y, w, h = cv2.boundingRect(contour) 
        imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                (x + w, y + h), 
                                (255, 0, 0), 2) 
        
        xCenter = int(x + w/2)
        yCenter = int(y + h/2)
        if area < 8000:
            # GO TO THIS OBJ
            
            cv2.putText(imageFrame, "Cube", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255))
            cv2.circle(imageFrame, (xCenter,yCenter), 2, (0,0,0), 2)
        if area > 8000:
            # THEN GO TO THIS SPOT
            
            cv2.circle(imageFrame, (xCenter,yCenter), 2, (0,0,0), 2)
            cv2.putText(imageFrame, "Spot", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))

        
# Program Termination 
cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame) 
if cv2.waitKey(0) & 0xFF == ord('q'): 
    cv2.destroyAllWindows() 
