import numpy as np
 
class GaitPlanner:
    def __init__(self, off_x, off_y, steps_per_phase=4):
        self.steps_per_phase = steps_per_phase
        self.off_x = off_x
        self.off_y = off_y
        self.hip_offsets = {
            'FL': np.array([ self.off_x,  self.off_y]),
            'FR': np.array([ self.off_x, -self.off_y]),
            'RL': np.array([-self.off_x,  self.off_y]),
            'RR': np.array([-self.off_x, -self.off_y])
        }
        self.trot_pairs = [
            ["FR", "RL"],
            ["FL", "RR"]
        ]

    def get_gait_horizon(self, step_index: int, N: int):
        """
        Returns the same couple for the whole horizon, in this way avoids the solver to move the right foot underneath the prediction of the left shoulder
        inputs: step_index (int), N (int)
        outputs: ['FR', 'RL', 'FR', 'RL', 'FR', 'RL',...] (N+1) times in base on the selected phase, if even phase is 0 otherwise it is 1
        """
        phase = (step_index // self.steps_per_phase) % 2
        return [self.trot_pairs[phase]] * (N + 1)

    def compute_hip_positions(self, c_state: np.ndarray, gait_pair: list):
        """
        Returns position of the hips given the robot position and the selected gait pair
        inputs: c_state (array), gait_pair (list: ['FR', 'RL', 'FR', 'RL', 'FR', 'RL',...])
        outputs: np.hstack([hip0, hip1]) hip position for that specific gait pair
        """
        c_x, c_y, theta = c_state[0], c_state[1], c_state[2]

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        hip0 = np.array([c_x, c_y]) + R @ self.hip_offsets[gait_pair[0]]
        hip1 = np.array([c_x, c_y]) + R @ self.hip_offsets[gait_pair[1]]

        return np.hstack([hip0, hip1])