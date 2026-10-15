from dataclasses import dataclass
import numpy as np


@dataclass
class RobotConfig:
    # Geometrical dimensions
    h_com: float = 0.38     # m
    g: float = 9.81         # m/s^2
    m: float = 24.24        # kg
    Iz: float = 1.048       # kg/m^2
    off_x: float = 0.2407   # m
    off_y: float = 0.134    # m
    mu: float = 0.8         # 
    max_ext_sq: float = 0.3**2 # m: Leg extension limit squared
    # Initial state
    x_init: np.ndarray = np.array([-0.8, -0.5]) # m
    theta_init: float = np.deg2rad(0) # deg


@dataclass
class Limits:
    dt_min: float = 0.1         # sec
    dt_max: float = 0.25        # sec
    theta_dot: float = 0.6      # rad/sec 
    v_max_x: float = 1.5        # m/s
    v_max_y: float = 0.45       # m/s
    alpha_min: float = 0.1      #
    alpha_max: float = 0.9      #
    f_diff_max:float = 100      # NOT USED


@dataclass
class MPCWeights:
    # Weights for the optimistic branch (W_diag_op)
    tracking_xy: float = 5.0        # 0-1: Tracking x, y
    vel_alignment: float = 10     # 2: Velocity alignment - penalize lateral walk
    vel_xy: float = 5.0             # 3-4: Velocity x, y
    yaw_rate: float = 1             # 5: Yaw rate (penalization on fast rotations)
    time_weight: float = 1e-6       # 6: Time
    anti_skating: float = 10      # 7-10: Anti-Skating foot 0 and 1
    posture: float = 1e4              # 11-14: Posture (keep hips above feet)
    alpha_weight: float = 1            # 15: Alpha
    dt_weight: float = 1e-3           # 16: dt
    

@dataclass
class SimulationConfig:
    N_horizon: int = 7
    sim_steps: int = 200
    steps_per_phase: int = 1
    c_target: np.ndarray = np.array([10, 2]) # Target position
    theta_target: float = np.deg2rad(0)         # Target orientation
    solver_type: str = 'SQP' # SQP, SQP_RTI
    max_iter: int = 4


def generate_random_obs():
    start_pos_x = np.random.uniform(-1, 10)
    start_pos_y = np.random.uniform(-2, 2)

    start_pos = [start_pos_x, start_pos_y]

    radius = np.random.uniform(0.1, 0.4)

    speed = np.random.uniform(0.1, 0.5)

    return start_pos, radius, speed




@dataclass
class ObstacleConfig:
    # General
    N_OBS_MAX: int = 6 # max number of obstacles
    pos_init: np.ndarray = np.array([0.1, 1.0]) #np.array([0.1, 1.0])
    r_obs: float = 0.2
    speed: float = 0.5              # actual velocity of the obstacle
    y_dot_max: float = 1         # velocity against which the pessimistic is robust
    obs_type: str = "static" # "static", "dynamic", "adversarial", "circular"
    # Dynamic
    top_pos_dyn: np.ndarray = np.array([0.1, 1])
    bot_pos_dyn: np.ndarray = np.array([0.1, -1])
    # Circular - make a circle around center starting from pos_init
    center: np.ndarray = np.array([1.0, 0.2])

    # RANDOM OBSTACLES
    corridor_bounds: np.ndarray = np.array([1, 9, -1.0, 1.0])
