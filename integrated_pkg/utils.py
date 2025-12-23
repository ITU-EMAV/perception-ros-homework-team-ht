import math
import csv
import os

def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    
    return roll, pitch, yaw

def load_waypoints(filepath):
    waypoints = []
    if not os.path.exists(filepath):
        print(f"Error: Waypoint file not found at {filepath}")
        return []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                # Format: x, y, z (z ignored)
                x = float(row[0].strip())
                y = float(row[1].strip())
                waypoints.append([x, y])
            except ValueError:
                continue
    return waypoints