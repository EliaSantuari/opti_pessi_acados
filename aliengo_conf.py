import numpy as np
from casadi import SX, vertcat, sin
from acados_template import AcadosModel


from robot_descriptions.loaders.pinocchio import load_robot_description


robot = load_robot_description("aliengo_description")

foot_names = ["FL", "FR", "RL", "RR"]

hip_joint_names = {"FL": "FL_hip_joint", "FR": "FR_hip_joint", "RL": "RL_hip_joint", "RR": "RR_hip_joint"}

hip_joint_ids, hip_pos = {}, {}


for k in hip_joint_names.keys():
	# Indeces of the hip joints
	hip_joint_ids[k] = robot.model.getJointId(hip_joint_names[k]) 

	# Position of the hip in the robot frame
	hip_pos[k] = robot.placement(robot.q0, hip_joint_ids[k]).translation[:2]

# Hip positions defined manually because closer to reality
hip_pos = {'FL': np.array([0.2407, 0.134]),
		   'FR': np.array([0.2407, -0.134]),
		   'RL': np.array([-0.2407, 0.134]),
    	   'RR': np.array([-0.2407, -0.134]),}
		  
print(robot)





### ------ Configuration fo LIPM Trajectory optimization ------

# COST WEIGHTS

wc = 1			# CoM Position error cost weight

wdc = 2 		# CoM Velocity error cost weight

wtheta = 0		# Orientation cost weight

wdtheta = 2		# Angulat velocity cost weight

wp = 5e-1		# Footstep distance to hip cost weight

wa = 5e-1 		# CoP error cost weight

wdt = 1e-2 		# Contact phase duration cost weight


# PHISICAL PARAMETERS

h = 0.38		# Fixed CoM height [m]

g = 9.81		# Gravity 

m = 24.24		# Robot mass [kg]: 21.5+-1 kg + battery(2.740 kg)

i = 1.048		# 24.24*(0.31**2 + 0.65**2)/12 

mu = 0.8 		# Feet-ground friction coefficient



### ------ Controller configuration ------

plot_each_time_step = 0		# Plot figures during simulation

save_results = 0			# save times array and results figures as pdf

N = 6						# Horizon size

sim_T = 20					# Max simulation time in [s]

x_threshold = 0.1			# Stop simulation id |x - x_goal| <= thr

alpha_reduction = 0.1		# Shrink CoP bounds

dt_min = 0.2				# Minimum time step [s]

dt_max = 0.35				# Max time step [s]

dt_cost0 = 0.35				# Time step with zero cost [s]

foot_hip_max = 0.1			# Foot-hip max distance [m]

dcx_max = 1.5				# Robot max longitudinal velocity [m/s]

dcy_max = 0.45				# Robot max lateral velocity [m/s]

dtheta_max = 0.8			# Robot max rotation velocity [rad/s]


# Robot configuration

x_0 = np.array([-0.8, -0.5, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0])
# CoM initial state (x, y, theta, dx, dy, dtheta) 

x_goal = np.array([1.0, 0.2])
# Final goal position

x_obstacle = np.array([[0.1, 1.0]])
# Obstacle initial state - we can add more obstacle

v_obstacle = 1.0
# Max obstacle velocity [m/s]

r_obstacle = 0.2
# Obstacle radius size [m]


### ------ Obstacle Dynamics ------

obstacle_movement = np.array(["straight", "straight"])
# straight, patrol, circle, antagonist

v_mag = np.array([0.0, 0.0])
# Velocity magnitud of the obstacle

v_dir = np.array([[0, -1], [0, -1]])
# Velocity direction 


