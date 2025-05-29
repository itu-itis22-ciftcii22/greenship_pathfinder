# apf_core.py
import numpy as np

# Parametreler
k_att = 1.0
k_rep = 10000.0
rep_range = 20

def attractive_force(pos, goal):
    return k_att * (goal - pos)

def repulsive_force(pos, obs):
    force = np.zeros(2)
    print(pos, obs)
    diff = pos - obs
    print(diff)
    dist = np.linalg.norm(diff)
    print(dist, rep_range)
    if dist < rep_range and dist != 0:
        force += k_rep * (1.0 / dist - 1.0 / rep_range) / (dist ** 3) * diff
    return force
