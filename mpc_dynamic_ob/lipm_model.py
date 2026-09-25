import casadi as cs
import numpy as np
from numpy import cosh, sinh
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver

def create_lipm_ocp(
        N=20, h_com=0.38, m=12.0, Iz=0.15,
        c_target=np.array([0.0, 0.0]), theta_target=0.001,
        x_init=np.zeros(11)
    ) -> AcadosOcpSolver:
    
    g = 9.81
    w = np.sqrt(g / h_com) # Frequency constant

    # Variables and parameters
    x = cs.SX.sym('x', 11) # cx, cy, theta, cdotx, cdoty, thetadot, p0x, p0y, p1x, p1y, time
    # Add the line parametrs a and b in the control variables and the parameters of the obstacle in the parameters
    u = cs.SX.sym('u', 11) # p0next_x, p0next_y, p1next_x, p1next_y, alpha, f_diff, dt, ax, ay, b
    p_sym = cs.SX.sym('p', 8) # local offset of hips with respect to CoM, y0_x, y0_y, r_obs, y_dot_max

    c, theta = x[0:2], x[2]
    c_dot, theta_dot = x[3:5], x[5]
    p0, p1 = x[6:8], x[8:10] 
    t = x[10]

    p0_next, p1_next = u[0:2], u[2:4]
    alpha = u[4]
    f_diff = u[5:7] 
    dt_var = u[7] 

    # Separating line (a_i, b_i)
    a = u[8:10]
    b = u[10]

    # Hip offsets parameters
    hip_offset0, hip_offset1 = p_sym[0:2], p_sym[2:4]

    # y_dot_max
    y_dot_max = p_sym[7]

    # Obstacle parameters
    y0_obs = p_sym[4:6]
    r_obs = p_sym[6] 


    # Dynamic function of variable dt
    # (CasADi will compute the gradient also with respect to cosh and sinh)
    ch = cs.cosh(w * dt_var)
    sh = cs.sinh(w * dt_var)

    cop = p0 + alpha * (p1 - p0) # Convex combination of the feet position
    c_ddot = (w**2) * (c - cop) # CoM acceleration

    # Divide the force into tangential (F_tot) and rotationsl (f_diff)
    # beta and gamma could only make the robot rotate only if it was already going ahead, otherwise it couldn't
    F_tot = m * c_ddot
    f0 = F_tot / 2.0 + f_diff
    f1 = F_tot / 2.0 - f_diff

    tau_z0 = (p0[0] - c[0]) * f0[1] - (p0[1] - c[1]) * f0[0]
    tau_z1 = (p1[0] - c[0]) * f1[1] - (p1[1] - c[1]) * f1[0]
    tau_z = tau_z0 + tau_z1

    # Numerical simulation (integration)
    c_next = ch * c + (sh / w) * c_dot + (1 - ch) * cop
    c_dot_next = w * sh * c + ch * c_dot - w * sh * cop
    theta_next = theta + theta_dot * dt_var
    theta_dot_next = theta_dot + (dt_var / Iz) * tau_z

    # Comulative time
    t_next = t + dt_var        

    x_next = cs.vertcat(c_next[0], c_next[1], theta_next, 
                        c_dot_next[0], c_dot_next[1], theta_dot_next, 
                        p0_next, p1_next, t_next)

    # Create the Acados Model
    model = AcadosModel()
    model.x, model.u, model.p, model.disc_dyn_expr = x, u, p_sym, x_next
    model.name = "lipm_step_ca" # Collision Avoidanca

    # Create the OCP
    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * 0.05 # fictitious dt for acados

    # Rotation matrix
    R_theta = cs.vertcat(
        cs.horzcat(cs.cos(theta_next), -cs.sin(theta_next)),
        cs.horzcat(cs.sin(theta_next), cs.cos(theta_next))
    )
    # Hip next position in the global frame
    hip0_expr_next = c_next + R_theta @ hip_offset0
    hip1_expr_next = c_next + R_theta @ hip_offset1

    # COSTS
    delta_p0_move = p0_next - p0  # anti-skating
    delta_p1_move = p1_next - p1
    hip_err_0 = p0_next - hip0_expr_next # Penalize distance from the shoulder
    hip_err_1 = p1_next - hip1_expr_next

    dt_nominal = 0.275 / 4.0 # We must fix a nominal value, otherwise the solver would sent it to zero to avoid errors
    # Define cost function: (y-y_ref).T W (y-y_ref)
    model.cost_y_expr = cs.vertcat(
        c, theta, c_dot, theta_dot, t,
        delta_p0_move, delta_p1_move, 
        hip_err_0, hip_err_1, 
        alpha - 0.5, f_diff,
        dt_var - dt_nominal
    )
    # Final cost: comprises only state variables and not controls
    model.cost_y_expr_e = cs.vertcat(c, theta, c_dot, theta_dot, t)

    W_diag = [
        2.0, 2.0,   # Tracking CoM
        2.0,          # Tracking Theta
        5.0, 5.0,     # Vel
        0.5,          # Ang Vel
        1e-8,          # t
        100.0, 100.0, # Anti-Skating P0
        100.0, 100.0, # Anti-Skating P1
        1.0, 1.0,     # Keep posture P0
        1.0, 1.0,     # Keep posture P1
        0.5,          # Alpha
        0.05, 0.05,   # f_diff (beta, gamma)
        10             # keep close to dt_nominal
    ]
    W_end = W_diag[0:7]
    

    # Type of cost function
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W = np.diag(W_diag)
    ocp.cost.W_e = np.diag(W_end)

    # Reference for the cost function
    y_ref = np.zeros(19)
    y_ref[:2] = c_target
    y_ref[2] = theta_target
    y_ref_e = np.zeros(7)
    y_ref_e[:2] = c_target
    y_ref_e[2] = theta_target 

    ocp.cost.yref, ocp.cost.yref_e = y_ref, y_ref_e

    # CONSTRAINTS
    mu = 0.8 # Friction coefficient
    # Hip distance from current step
    hip_dist0_curr_sq = cs.sumsqr(p0 - hip0_expr_next)
    hip_dist1_curr_sq = cs.sumsqr(p1 - hip1_expr_next)
    # Hip distance from next step
    hip_dist0_next_sq = cs.sumsqr(p0_next - hip0_expr_next)
    hip_dist1_next_sq = cs.sumsqr(p1_next - hip1_expr_next)

    # Friction constraints
    f0_sq = f0[0]**2 + f0[1]**2
    f1_sq = f1[0]**2 + f1[1]**2
    friction_cone_0_sq = f0_sq - (mu * alpha * m * g)**2 + 1e-3
    friction_cone_1_sq = f1_sq - (mu * (1.0 - alpha) * m * g)**2 + 1e-3



    # Separating plane Constraints
    # 1) Compute the norm of a == 1
    eq_norm_a = a[0]**2 + a[1]**2 # == 1

    # 2) 4 angles of the robot must stay in the negative semiplane (we don't want the robot to be inside the obstacle)
    # Assuming the shoulders +/- 0.2 with respect to CoM
    corners_local = [
        cs.vertcat(0.2, 0.2), cs.vertcat(-0.2, 0.2),
        cs.vertcat(-0.2, -0.2), cs.vertcat(0.2, -0.2)
    ]
    eq_s = []
    for c_loc in corners_local:
        s_v = c_next + R_theta @ c_loc
        # aT.sv+b<=0
        eq_s.append(cs.dot(a, s_v) + b) # <= 0

    # 3) 2 feet in the negative semiplane
    eq_p0_obs = cs.dot(a, p0_next) + b # <= 0
    eq_p1_obs = cs.dot(a, p1_next) + b # <= 0

    # 4) Dynamic obstacle should be in the positive semiplane and distant at least r_obs (+ 5cm for safety!)
    r_dynamic = r_obs + y_dot_max * t_next
    eq_obs = cs.dot(a, y0_obs) + b - (r_dynamic+0.0) # > 0

    # Define the constraints expression
    model.con_h_expr = cs.vertcat(hip_dist0_curr_sq, hip_dist1_curr_sq, hip_dist0_next_sq, hip_dist1_next_sq, friction_cone_0_sq, friction_cone_1_sq, eq_norm_a, eq_s[0], eq_s[1], eq_s[2], eq_s[3], eq_p0_obs, eq_p1_obs, eq_obs)
    # Max extension squared
    max_ext_sq = 0.03
    # Lower limits
    lh_robot = [0.0, 0.0, 0.0, 0.0, -1e6, -1e6]
    lh_obs = [0.0, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6, 0.0] 
    ocp.constraints.lh = np.array(lh_robot + lh_obs)
    # Upper limits
    uh_robot = [max_ext_sq, max_ext_sq, max_ext_sq, max_ext_sq, 0.0, 0.0]
    uh_obs  = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e6]
    ocp.constraints.uh = np.array(uh_robot + uh_obs)

    # Constraint on states
    ocp.constraints.x0 = x_init
    ocp.constraints.lbx = np.array([-0.6, -0.3, -0.8, 0.0]) # cdotx, cdoty, theta_dot
    ocp.constraints.ubx = np.array([0.6, 0.3, 0.8, 100.0])
    ocp.constraints.idxbx = np.array([3, 4, 5, 10])

    # SOFT CONSTRAINTS
    # acados manage the soft constraints as SLACK VARIABLES
    # Slack Variables
    # We have the soft constraint on the velocities (3) and on the distance from current and next step to hip
    # Velocities
    ocp.constraints.idxsbx = np.array([0, 1, 2])
    # Make soft the legs and the obstacle contraints except norm of a
    ocp.constraints.idxsh = np.array([0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13])

    # Number of soft constraints 3+11

    # These are the weights for the limits (upper and lower) of soft constraints
    Zu_list = [1e3]*3 + [1e4]*4 + [1e6]*7 # High weights for the collision
    zu_list = [1e2]*3 + [1e3]*4 + [1e5]*7
    ocp.cost.Zu = np.array(Zu_list)
    ocp.cost.zu = np.array(zu_list)

    Zl_list = [1e3]*3 + [0.0]*4 + [0.0]*6 + [1e6] # High weight if obstacle break the line
    zl_list = [1e2]*3 + [0.0]*4 + [0.0]*6 + [1e5]
    ocp.cost.Zl = np.array(Zl_list)
    ocp.cost.zl = np.array(zl_list)

    # Constraint on the control of alpha and time step
    dt_min = 0.2 / 4.0 # Because the robot will do a step every 4 dt_var
    dt_max = 0.35 / 4.0
    ocp.constraints.lbu = np.array([0.1, -100, -100, dt_min, -1.0, -1.0, -10.0])
    ocp.constraints.ubu = np.array([0.9, 100, 100, dt_max, 1.0, 1.0, 10.0])
    ocp.constraints.idxbu = np.array([4, 5, 6, 7, 8, 9, 10])

    ocp.parameter_values = np.zeros(8)

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    ocp.solver_options.nlp_solver_max_iter = 100

    return AcadosOcpSolver(ocp)