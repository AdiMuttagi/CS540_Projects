"""
Question 1 (Part 1): output just the sensor readings (s1..sk) for all
300 test items, one row per line, comma separated. No action label, no
x/y/vx/vy -- those are only needed to compute the score, not for upload.
"""
import os
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(os.path.join(root, "data", "test.txt"), delimiter=",", dtype=np.float64)

sensors = data[:, 5:]  # columns: action, x, y, vx, vy, s1, ..., sk -> sensors start at index 5

out_path = os.path.join(root, "answers", "q1_sensors.txt")
with open(out_path, "w") as f:
    for row in sensors:
        f.write(",".join(f"{v:.4f}" for v in row) + "\n")

print(f"Wrote {len(sensors)} rows x {sensors.shape[1]} sensors to {out_path}")
