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
    mu: float = 0.6         # 
    max_ext_sq: float = 0.1 # m: Leg extension limit squared
    # Initial state
    x_init: np.ndarray = np.array([-0.8, -0.5]) # m
    theta_init: float = np.deg2rad(0) # deg


@dataclass
class Limits:
    dt_min: float = 0.1         # sec
    dt_max: float = 0.25        # sec
    theta_dot: float = 0.6      # rad/sec 
    v_max_x: float = 1.0        # m/s
    v_max_y: float = 0.45       # m/s
    alpha_min: float = 0.1      #
    alpha_max: float = 0.9      #
    f_diff_max:float = 100      # NOT USED


@dataclass
class MPCWeights:
    # Weights for the optimistic branch (W_diag_op)
    tracking_xy: float = 5.0        # 0-1: Tracking x, y
    vel_alignment: float = 60.0     # 2: Velocity alignment - penalize lateral walk
    vel_xy: float = 5.0             # 3-4: Velocity x, y
    yaw_rate: float = 1             # 5: Yaw rate (penalization on fast rotations)
    time_weight: float = 1e-6       # 6: Time
    anti_skating: float = 10.0      # 7-10: Anti-Skating foot 0 and 1
    posture: float = 1000.0         # 11-14: Posture (keep hips above feet)
    alpha_weight: float = 1         # 15: Alpha
    dt_weight: float = 1e-3         # 16: dt
    f_diff_weight: float = 10       # f_diff NOT USED
    theta_dyn: float = 10.0         # Theta dynamic NOT USED


@dataclass
class SimulationConfig:
    N_horizon: int = 7
    sim_steps: int = 400
    steps_per_phase: int = 1
    c_target: np.ndarray = np.array([1.0, 0.2]) # Target position
    theta_target: float = np.deg2rad(0)         # Target orientation
    solver_type: str = 'SQP_RTI' # SQP, SQP_RTI
    max_iter: int = None




@dataclass
class ObstacleConfig:
    # General
    pos_init: np.ndarray = np.array([0.5, 2.0]) #np.array([0.1, 1.0])
    r_obs: float = 0.2
    speed: float = 1
    y_dot_max: float = 1
    obs_type: str = "dynamic" # "static", "dynamic", "adversarial", "circular"
    # Dynamic
    top_pos_dyn: np.ndarray = np.array([0.1, 0.5])
    bot_pos_dyn: np.ndarray = np.array([0.1, -1])
    # Circular - make a circle around center starting from pos_init
    center: np.ndarray = np.array([1.0, 0.2])