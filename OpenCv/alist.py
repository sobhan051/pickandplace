import numpy as np
def listing(start,goal):
    #print(f"start without change: {start}\n")
    s1 = start[0]
    #print(f"this is s1: {s1}\n")
    start = np.delete(start,[0],0)
    #print(f"start with change: {start}\n")
    g1 = goal[0]
    #print(f"this is the goal:{goal}\n")
    return s1,g1,start
