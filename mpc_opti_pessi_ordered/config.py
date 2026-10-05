from dataclasses import dataclass
import numpy as np


@dataclass
class RobotConfig:
    # Geometrical dimensions
    h_com: float = 0.38
    g: float = 9.81
    m: float = 24.24 # 12.0
    Iz: float = 1.048 # 0.15
    off_x: float = 0.2407 # 0.2
    off_y: float = 0.134
    mu: float = 0.6 # 0.8
    max_ext_sq: float = 0.1 # Leg extension limit squared
    # Initial state
    x_init: np.ndarray = np.array([-0.8, -0.5])
    theta_init: float = np.deg2rad(0)


@dataclass
class Limits:
    dt_min: float = 0.01     # / steps_per_phase 
    dt_max: float = 0.1   # / steps_per_phase 
    theta_dot: float = 0.6 # 0.8  
    v_max_x: float = 1.0 # 0.6
    v_max_y: float = 0.45 # 0.3
    alpha_min: float = 0.1
    alpha_max: float = 0.9
    f_diff_max:float = 100


@dataclass
class MPCWeights:
    # Weights for the optimistic branch (W_diag_op)
    tracking_xy: float = 5.0       # 0-1: Tracking x, y
    theta_dyn: float = 10.0         # 2: Theta dynamic NOT USED
    vel_xy: float = 5.0             # 3-4: Velocity x, y
    yaw_rate: float = 1             # 5: Yaw rate
    time_weight: float = 1e-6       # 6: Time
    anti_skating: float = 10.0      # 7-10: Anti-Skating
    posture: float = 300.0          # 11-14: Posture
    alpha_weight: float = 1         # 15: Alpha
    f_diff_weight: float = 10       # 16-17: f_diff NOT USED
    dt_weight: float = 1e-3         # 18: dt
    vel_alignment: float = 60.0     # 19: Velocity alignment


@dataclass
class SimulationConfig:
    N_horizon: int = 7
    sim_steps: int = 400
    steps_per_phase: int = 1
    c_target: np.ndarray = np.array([1.0, 0.2])
    theta_target: float = np.deg2rad(0)
    solver_type: str = 'SQP_RTI' # SQP, SQP_RTI
    max_iter: int = None



@dataclass
class ObstacleConfig:
    # General
    pos_init: np.ndarray = np.array([0.5, 0.5]) #np.array([0.1, 1.0])
    r_obs: float = 0.21
    speed: float = 0.35
    y_dot_max: float = 0.35
    obs_type: str = "dynamic" # "static", "dynamic", "adversarial", "circular"
    # Dynamic
    top_pos_dyn: np.ndarray = np.array([0.1, 0.5])
    bot_pos_dyn: np.ndarray = np.array([0.1, -1])
    # Circular
    center: np.ndarray = np.array([1.0, 0.2])