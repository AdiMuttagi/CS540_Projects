"""
The "simple behavior policy" from Part 2.

Rule (5 sensors s1..s5, index 0..4):
  - if the middle sensor (s3, index 2) is tied for the highest value -> speed up
  - else if either left sensor (s1 or s2, index 0 or 1) is tied for the highest -> turn left
  - else (one of the right sensors, s4 or s5, index 3 or 4, is highest) -> turn right

Tie-break order when multiple sensors share the max value: speed up > left > right.

Action labels used everywhere in this project:
  0 = turn left
  1 = turn right
  2 = speed up
  3 = no action   (this simple policy never outputs "no action")
"""
import numpy as np

LEFT, RIGHT, SPEED_UP, NO_ACTION = 0, 1, 2, 3


def behavior_action(sensors):
    """sensors: array of 5 values (s1..s5). Returns the action label (int)."""
    best = np.max(sensors)
    is_max = sensors == best  # boolean array, True where a sensor ties for the max

    if is_max[2]:                       # middle sensor is highest
        return SPEED_UP
    if is_max[0] or is_max[1]:          # one of the two left sensors is highest
        return LEFT
    return RIGHT                        # otherwise one of the two right sensors is highest


def behavior_actions(sensor_matrix):
    """sensor_matrix: (n, 5) array. Returns an (n,) array of action labels."""
    return np.array([behavior_action(row) for row in sensor_matrix])
