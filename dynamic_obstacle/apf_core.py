# apf_core.py
import numpy as np

# Parametreler
k_att = 1.0
k_rep = 100.0
rep_range = 2.5

def attractive_force(pos, goal):
    return k_att * (goal - pos)

def repulsive_force(pos, obstacles):
    force = np.zeros(2)
    for obs in obstacles:
        diff = pos - obs
        dist = np.linalg.norm(diff)
        if dist < rep_range and dist != 0:
            force += k_rep * (1.0 / dist - 1.0 / rep_range) / (dist ** 3) * diff
    return force
