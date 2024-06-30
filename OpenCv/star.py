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
        #r = int(r/0.5)
        #c = int(c/0.5)
        cv2.circle(image,(r,c), 2, (255,0,127), 2)



img = cv2.imread("asset/pic2.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
start= (576,375)
end = (140,106)
obsitcl = []

path = astar(gray.shape,start,end,obsitcl)
if path:
    cvDraw(img, path)
else:
    print("NO PATH")