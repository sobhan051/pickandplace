import numpy as np
import cv2
import heapq

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(image, start_set, end_set):
    open_list = []
    closed_list = set()
    end_positions = set(end_set)
    
    for start in start_set:
        start_node = Node(start)
        heapq.heappush(open_list, start_node)
    
    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)
        
        if current_node.position in end_positions:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]
        
        (x, y) = current_node.position
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        
        for next_position in neighbors:
            (nx, ny) = next_position
            if nx < 0 or nx >= image.shape[0] or ny < 0 or ny >= image.shape[1]:
                continue
            if image[nx, ny] == 0:
                continue
            neighbor_node = Node(next_position, current_node)
            if neighbor_node.position in closed_list:
                continue
            
            neighbor_node.g = current_node.g + 1
            neighbor_node.h = min(heuristic(neighbor_node.position, end) for end in end_set)
            neighbor_node.f = neighbor_node.g + neighbor_node.h
            
            if add_to_open(open_list, neighbor_node):
                heapq.heappush(open_list, neighbor_node)
    
    return None

def add_to_open(open_list, neighbor_node):
    for node in open_list:
        if neighbor_node == node and neighbor_node.g > node.g:
            return False
    return True

# Example usage
if __name__ == "__main__":
    # Load image
    image = cv2.imread('asset/pic1.png', cv2.IMREAD_GRAYSCALE)
    grayRes = cv2.resize(image,(0,0),fx=0.3 , fy=0.3)
    # Convert to binary
    _, binary_image = cv2.threshold(grayRes, 127, 1, cv2.THRESH_BINARY)
    
    start_set = [(173,112)]  # Set of starting points (row, col)
    end_set = [(42,32)]  # Set of goal points (row, col)
    
    path = astar(binary_image, start_set, end_set)
    if path:
        print("Path found:", path)
    else:
        print("No path found")



