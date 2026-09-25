import casadi as cs
import numpy as np
from numpy import cosh, sinh
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver

def create_lipm_ocp(
        N=20, h_com=0.38, m=12.0, Iz=0.15,
        c_target=np.array([0.0, 0.0]), theta_target=0.001,
        x_init=np.zeros(10)
    ) -> AcadosOcpSolver:
    
    g = 9.81
    w = np.sqrt(g / h_com) # Frequency constant

    # Variables and parameters
    x = cs.SX.sym('x', 10) # cx, cy, theta, cdotx, cdoty, thetadot, p0x, p0y, p1x, p1y
    u = cs.SX.sym('u', 8) # p0next_x, p0next_y, p1next_x, p1next_y, alpha, f_diff, dt
    p_sym = cs.SX.sym('p', 4) # local offset of hips with respect to CoM

    c, theta = x[0:2], x[2]
    c_dot, theta_dot = x[3:5], x[5]
    p0, p1 = x[6:8], x[8:10] 

    p0_next, p1_next = u[0:2], u[2:4]
    alpha = u[4]
    f_diff = u[5:7] 
    dt_var = u[7] 

    hip_offset0, hip_offset1 = p_sym[0:2], p_sym[2:4]

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

    x_next = cs.vertcat(c_next[0], c_next[1], theta_next, 
                        c_dot_next[0], c_dot_next[1], theta_dot_next, 
                        p0_next, p1_next)

    # Create the Acados Model
    model = AcadosModel()
    model.x, model.u, model.p, model.disc_dyn_expr = x, u, p_sym, x_next
    model.name = "lipm_step"

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
        c, theta, c_dot, theta_dot, 
        delta_p0_move, delta_p1_move, 
        hip_err_0, hip_err_1, 
        alpha - 0.5, f_diff,
        dt_var - dt_nominal
    )
    # Final cost: comprises only state variables and not controls
    model.cost_y_expr_e = cs.vertcat(c, theta, c_dot, theta_dot)

    W_diag = [
        10.0, 10.0,   # Tracking CoM
        5.0,          # Tracking Theta
        1.0, 1.0,     # Vel
        0.5,          # Ang Vel
        100.0, 100.0, # Anti-Skating P0
        100.0, 100.0, # Anti-Skating P1
        1.0, 1.0,     # Keep posture P0
        1.0, 1.0,     # Keep posture P1
        0.5,          # Alpha
        0.05, 0.05,   # f_diff (beta, gamma)
        10             # keep close to dt_nominal
    ]
    W_end = W_diag[0:6]

    # Type of cost function
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W = np.diag(W_diag)
    ocp.cost.W_e = np.diag(W_end)

    # Reference for the cost function
    y_ref = np.zeros(18)
    y_ref[:2] = c_target
    y_ref[2] = theta_target
    y_ref_e = np.zeros(6)
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

    # Define the constraints expression
    model.con_h_expr = cs.vertcat(hip_dist0_curr_sq, hip_dist1_curr_sq, hip_dist0_next_sq, hip_dist1_next_sq, friction_cone_0_sq, friction_cone_1_sq)
    # Max extension squared
    max_ext_sq = 0.02
    ocp.constraints.lh = np.array([0.0, 0.0, 0.0, 0.0, -1e6, -1e6])
    ocp.constraints.uh = np.array([max_ext_sq, max_ext_sq, max_ext_sq, max_ext_sq, 0, 0])

    # Constraint on states
    ocp.constraints.x0 = x_init
    ocp.constraints.lbx = np.array([-1.5, -0.45, -0.8]) # cdotx, cdoty, thetadot
    ocp.constraints.ubx = np.array([1.5, 0.45, 0.8])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # acados manage the soft constraints as SLACK VARIABLES
    # Slack Variables
    # We have the soft constraint on the velocities (3) and on the distance from current and next step to hip
    # Velocities
    ocp.constraints.idxsbx = np.array([0, 1, 2])
    # Distances
    ocp.constraints.idxsh = np.array([0, 1, 2, 3])

    # These are the weights for the limits (upper and lower) of soft constraints
    ocp.cost.Zu = np.array([1e3, 1e3, 1e3,   1e4, 1e4, 1e4, 1e4])
    ocp.cost.zu = np.array([1e2, 1e2, 1e2,   1e3, 1e3, 1e3, 1e3])
    ocp.cost.Zl = np.array([1e3, 1e3, 1e3,   0.0, 0.0, 0.0, 0.0])
    ocp.cost.zl = np.array([1e2, 1e2, 1e2,   0.0, 0.0, 0.0, 0.0])

    # Constraint on the control of alpha and time step
    dt_min = 0.2 / 4.0 # Because the robot will do a step every 4 dt_var
    dt_max = 0.35 / 4.0
    ocp.constraints.lbu = np.array([0.1, -100, -100, dt_min])
    ocp.constraints.ubu = np.array([0.9, 100, 100, dt_max])
    ocp.constraints.idxbu = np.array([4, 5, 6, 7])

    ocp.parameter_values = np.zeros(4)

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    return AcadosOcpSolver(ocp)