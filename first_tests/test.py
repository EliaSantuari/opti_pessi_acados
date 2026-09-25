
import numpy as np

gait_pattern = [["FR", "RL"], ["FL", "RR"]]*6


hip_pos = {'FL': np.array([0.2, 0.2]), 
           'FR': np.array([ 0.2, -0.2]), 
           'RL': np.array([-0.2,  0.2]), 
           'RR': np.array([-0.2, -0.2])}


def hip_pos01(hip_pos, c, gait_pattern, i):
    hip_pos0 = c + hip_pos[gait_pattern[i][0]]
    hip_pos1 = c + hip_pos[gait_pattern[i][1]]
    return hip_pos0, hip_pos1

hip_pos0, hip_pos1 = hip_pos01(hip_pos, np.array([1, 1]), gait_pattern, 1)
print(hip_pos0)
print(hip_pos1)