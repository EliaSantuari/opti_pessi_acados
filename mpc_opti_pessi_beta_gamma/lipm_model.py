import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver

def create_lipm_ocp(
        N=20, h_com=0.38, m=12.0, Iz=0.15,
        c_target=np.array([0.0, 0.0]), theta_target=0.001,
        x_init=np.zeros(22)
    ) -> AcadosOcpSolver:
    
    g = 9.81
    w = np.sqrt(g / h_com) # Frequency constant

    # Variables and parameters
    x_op = cs.SX.sym('x_op', 11) # cx, cy, theta, cdotx, cdoty, thetadot, p0x, p0y, p1x, p1y, time
    x_pe = cs.SX.sym('x_pe', 11)
    x = cs.vertcat(x_op, x_pe)
    
    u_op = cs.SX.sym('u_op', 11) # p0next_x, p0next_y, p1next_x, p1next_y, alpha, f_diff_x, f_diff_y, dt, ax, ay, b
    u_pe = cs.SX.sym('u_pe', 11)
    u = cs.vertcat(u_op, u_pe)
    
    p_sym = cs.SX.sym('p', 8) # hip0_x, hip0_y, hip1_x, hip1_y, obs_x, obs_y, r_obs, y_dot_max

    hip_offset0, hip_offset1 = p_sym[0:2], p_sym[2:4]
    y0_obs = p_sym[4:6]
    r_obs = p_sym[6] 
    y_dot_max = p_sym[7]

    # --- HELPER FUNCTION FOR DYNAMIC ---
    def compute_dynamic(x_curr, u_curr):
        c, theta = x_curr[0:2], x_curr[2]
        c_dot, theta_dot = x_curr[3:5], x_curr[5]
        p0, p1 = x_curr[6:8], x_curr[8:10]
        t = x_curr[10]

        p0_next, p1_next = u_curr[0:2], u_curr[2:4]
        alpha = u_curr[4]
        beta = u_curr[5]
        gamma = u_curr[6]
        dt_var = u_curr[7]

        ch, sh = cs.cosh(w * dt_var), cs.sinh(w * dt_var)
        cop = p0 + alpha * (p1 - p0) # Convex combination of the feet position
        c_ddot = (w**2) * (c - cop) # CoM acceleration

        # Divide the force into tangential (F_tot) and rotationsl (f_diff)
                    # beta and gamma could only make the robot rotate only if it was already going ahead, otherwise it couldn't
        F_tot = m * c_ddot
        f0 = cs.vertcat(beta * F_tot[0], gamma * F_tot[1])
        f1 = cs.vertcat((1-beta) * F_tot[0], (1-gamma) * F_tot[1])

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

        return x_next, c_next, theta_next, f0, f1

    # Compute the evolution of both worlds
    x_next_op, c_next_op, theta_next_op, f0_op, f1_op = compute_dynamic(x_op, u_op)
    x_next_pe, c_next_pe, theta_next_pe, f0_pe, f1_pe = compute_dynamic(x_pe, u_pe)

    x_next = cs.vertcat(x_next_op, x_next_pe)

    # Create the Acados Model
    model = AcadosModel()
    model.x, model.u, model.p, model.disc_dyn_expr = x, u, p_sym, x_next
    model.name = "lipm_opti_pessi" 

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * 0.05 

    # --- COST FUNCTION FOR OPTIMISTIC BRANCH (Main Objective) ---
    c_op, theta_op = x_op[0:2], x_op[2]
    c_dot_op, theta_dot_op = x_op[3:5], x_op[5]
    p0_op, p1_op = x_op[6:8], x_op[8:10]
    p0_next_op, p1_next_op = u_op[0:2], u_op[2:4]

    # Rotation matrix
    R_theta_op = cs.vertcat(
        cs.horzcat(cs.cos(theta_next_op), -cs.sin(theta_next_op)),
        cs.horzcat(cs.sin(theta_next_op), cs.cos(theta_next_op))
    )

    # Hip next position in the global frame
    hip0_expr_next_op = c_next_op + R_theta_op @ hip_offset0
    hip1_expr_next_op = c_next_op + R_theta_op @ hip_offset1

    # --- Dynamic weights and alignment with velocity vector ---
    k_dist = 1.0
    dist_sq_op = (c_op[0] - c_target[0])**2 + (c_op[1] - c_target[1])**2
    # epsilon will be 1 on the target and 0 when far from it
    epsilon_op = 1.0 / (1.0 + k_dist * dist_sq_op)

    # Use the weighted error instead of theta_err
    theta_err_weighted_op = epsilon_op * (theta_op - theta_target)

    # Weight the velocity (1-alpha)
    v_loc_y_op = -c_dot_op[0] * cs.sin(theta_op) + c_dot_op[1] * cs.cos(theta_op)
    v_loc_weighted_op = (1.0 - epsilon_op) * v_loc_y_op


    # COSTS
    delta_p0_move_op = p0_next_op - p0_op  # anti-skating
    delta_p1_move_op = p1_next_op - p1_op
    hip_err_0_op = p0_next_op - hip0_expr_next_op # Penalize distance from the shoulder
    hip_err_1_op = p1_next_op - hip1_expr_next_op

    dt_nominal = 0.275 / 4.0 # We must fix a nominal value, otherwise the solver would sent it to zero to avoid errors
    cost_y_expr_op = cs.vertcat(
        c_op, theta_err_weighted_op, c_dot_op, theta_dot_op, x_op[10],
        delta_p0_move_op, delta_p1_move_op, 
        hip_err_0_op, hip_err_1_op, 
        u_op[4] - 0.5, u_op[5:7] - 0.5, u_op[7] - dt_nominal,
        v_loc_weighted_op
    )
    cost_y_expr_e_op = cs.vertcat(c_op, theta_op, c_dot_op, theta_dot_op, x_op[10])

    # --- COST FUNCTION FOR PESSIMISTIC BRANCH (Regularization) ---
    c_pe, theta_pe = x_pe[0:2], x_pe[2]
    c_dot_pe, theta_dot_pe = x_pe[3:5], x_pe[5]
    p0_pe, p1_pe = x_pe[6:8], x_pe[8:10]
    p0_next_pe, p1_next_pe = u_pe[0:2], u_pe[2:4]

    R_theta_pe = cs.vertcat(
        cs.horzcat(cs.cos(theta_next_pe), -cs.sin(theta_next_pe)),
        cs.horzcat(cs.sin(theta_next_pe), cs.cos(theta_next_pe))
    )
    hip0_expr_next_pe = c_next_pe + R_theta_pe @ hip_offset0
    hip1_expr_next_pe = c_next_pe + R_theta_pe @ hip_offset1

    dist_sq_pe = (c_pe[0] - c_target[0])**2 + (c_pe[1] - c_target[1])**2
    epsilon_pe = 1.0 / (1.0 + k_dist * dist_sq_pe)
    theta_err_weighted_pe = epsilon_pe * (theta_pe - theta_target)
    v_loc_y_pe = -c_dot_pe[0] * cs.sin(theta_pe) + c_dot_pe[1] * cs.cos(theta_pe)
    v_loc_weighted_pe = (1.0 - epsilon_pe) * v_loc_y_pe

    delta_p0_move_pe = p0_next_pe - p0_pe  
    delta_p1_move_pe = p1_next_pe - p1_pe
    hip_err_0_pe = p0_next_pe - hip0_expr_next_pe 
    hip_err_1_pe = p1_next_pe - hip1_expr_next_pe

    cost_y_expr_pe = cs.vertcat(
        c_pe, theta_err_weighted_pe, c_dot_pe, theta_dot_pe, x_pe[10],
        delta_p0_move_pe, delta_p1_move_pe, 
        hip_err_0_pe, hip_err_1_pe, 
        u_pe[4] - 0.5, u_pe[5:7] - 0.5, u_pe[7] - dt_nominal,
        v_loc_weighted_pe
    )
    cost_y_expr_e_pe = cs.vertcat(c_pe, theta_pe, c_dot_pe, theta_dot_pe, x_pe[10])

    # Combine Expressions (40 elements for intermediate, 14 for terminal)
    # Define cost function: (y-y_ref).T W (y-y_ref)
    model.cost_y_expr = cs.vertcat(cost_y_expr_op, cost_y_expr_pe)
    model.cost_y_expr_e = cs.vertcat(cost_y_expr_e_op, cost_y_expr_e_pe)

    # Weights
    W_diag_op = [
        20.0, 20.0,                 # 0,1: Tracking x, y
        10.0,                       # 2: theta dynamic
        5.0, 5.0,                   # 3,4: velocity x, y
        0.5,                        # 5: Yaw rate
        1e-6,                       # 6: Time
        10.0, 10.0, 10.0, 10.0,     # 7-10: Anti-Skating
        200.0, 200.0, 200.0, 200.0, # 11-14: Posture
        0.5, 0.05, 0.05, 10.0,      # 15-18: Controls
        500                         # 19: Velocity alignment
    ]
    # Regularization weight for the pessimistic branch to avoid null-space explosion
    W_diag_pe = [w * 1e-3 for w in W_diag_op] 

    W_end_op = W_diag_op[0:7]
    W_end_pe = W_diag_pe[0:7]
    
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W = np.diag(W_diag_op + W_diag_pe)
    ocp.cost.W_e = np.diag(W_end_op + W_end_pe)

    # Reference (Updated lengths)
    y_ref = np.zeros(40)
    y_ref[:2] = c_target
    # y_ref[2] = theta_target
    y_ref[20:22] = c_target
    # y_ref[21] = theta_target
    
    # In terminal cost I must leave the theta_target elements because for the running I used epsilon which embeds the difference, for the terminal cost I don't
    y_ref_e = np.zeros(14)
    y_ref_e[:2] = c_target
    y_ref_e[2] = theta_target 
    y_ref_e[7:9] = c_target
    y_ref_e[9] = theta_target

    ocp.cost.yref, ocp.cost.yref_e = y_ref, y_ref_e

    # --- HELPER FUNCTION FOR CONSTRAINTS ---
    def compute_constraints(x_curr, u_curr, c_next, theta_next, f0, f1, r_obstacle):
        mu = 0.8 # Friction coefficient

        # Extract current states
        theta = x_curr[2]
        c_dot = x_curr[3:5]

        p0, p1 = x_curr[6:8], x_curr[8:10]
        p0_next, p1_next = u_curr[0:2], u_curr[2:4]
        a, b = u_curr[8:10], u_curr[10]

        R_th = cs.vertcat(cs.horzcat(cs.cos(theta_next), -cs.sin(theta_next)), cs.horzcat(cs.sin(theta_next), cs.cos(theta_next)))
        # Hip position of next step
        h0_next, h1_next = c_next + R_th @ hip_offset0, c_next + R_th @ hip_offset1

        # Hip distance from current step
        h_dist0_curr, h_dist1_curr = cs.sumsqr(p0 - h0_next), cs.sumsqr(p1 - h1_next)
        # Hip distance from future step
        h_dist0_next, h_dist1_next = cs.sumsqr(p0_next - h0_next), cs.sumsqr(p1_next - h1_next)

        # Friction constraints
        fric0 = f0[0]**2 + f0[1]**2 - (mu * (1.0 - u_curr[4]) * m * g)**2 + 1e-3
        fric1 = f1[0]**2 + f1[1]**2 - (mu * u_curr[4] * m * g)**2 + 1e-3

        # Separating plane Constraints
        # 1) Compute the norm of a == 1
        eq_norm_a = a[0]**2 + a[1]**2 

        # 2) 4 angles of the robot must stay in the negative semiplane(we  don't want the robot to be inside the obstacle)
        # Assuming the shoulders +/- 0.2 with respect to CoM
        corners = [cs.vertcat(0.2, 0.2), cs.vertcat(-0.2, 0.2), cs.vertcat(-0.2, -0.2), cs.vertcat(0.2, -0.2)]
        eq_s = [cs.dot(a, c_next + R_th @ loc) + b for loc in corners]
        eq_p = [cs.dot(a, p0_next) + b, cs.dot(a, p1_next) + b]  

        # 4) Dynamic obstacle should be in the positive semiplane and distant at least r_obs
        eq_obs = cs.dot(a, y0_obs) + b - (r_obstacle * (1 + 0.5))

        # 5) Compute the local velocities
        v_loc_x = c_dot[0] * cs.cos(theta) + c_dot[1] * cs.sin(theta)
        v_loc_y = -c_dot[0] * cs.sin(theta) + c_dot[1] * cs.cos(theta)

        return cs.vertcat(h_dist0_curr, h_dist1_curr, h_dist0_next, h_dist1_next, fric0, fric1, eq_norm_a, eq_s[0], eq_s[1], eq_s[2], eq_s[3], eq_p[0], eq_p[1], eq_obs, v_loc_x, v_loc_y)

    # 1. Optimistic uses fixed nominal radius
    con_op = compute_constraints(x_op, u_op, c_next_op, theta_next_op, f0_op, f1_op, r_obs)

    # 2. Pessimistic uses dynamic growing radius
    r_dynamic = r_obs + y_dot_max * x_pe[10]
    con_pe = compute_constraints(x_pe, u_pe, c_next_pe, theta_next_pe, f0_pe, f1_pe, r_dynamic)

    model.con_h_expr = cs.vertcat(con_op, con_pe) # 28 equations
    
    # Decouple separating planes: Only the physical 8 controls are forced equal at node 0
    u_diff = u_op[0:8] - u_pe[0:8] # 8 equations
    model.con_h_expr_0 = cs.vertcat(u_diff, con_op, con_pe)

    # Limits
    max_ext_sq = 0.03
    lh_robot = [0.0, 0.0, 0.0, 0.0, -1e6, -1e6]
    lh_obs = [0.0, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6, 0.0] 
    lh_vel = [-0.6, -0.3] 
    lh = lh_robot + lh_obs + lh_vel
    ocp.constraints.lh = np.array(lh + lh)
    
    uh_robot = [max_ext_sq, max_ext_sq, max_ext_sq, max_ext_sq, 0.0, 0.0]
    uh_obs  = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e6]
    uh_vel = [0.6, 0.3] 
    uh = uh_robot + uh_obs + uh_vel
    ocp.constraints.uh = np.array(uh + uh)

    # Initial constraints (36 equations: 8 for u_diff == 0 + 28)
    ocp.constraints.lh_0 = np.array([0.0]*8 + lh + lh)
    ocp.constraints.uh_0 = np.array([0.0]*8 + uh + uh)

    # Constraints on States
    ocp.constraints.x0 = x_init
    bnd_x = [-0.8, 0.0]
    BND_X = [0.8, 100.0]
    ocp.constraints.lbx = np.array(bnd_x + bnd_x) 
    ocp.constraints.ubx = np.array(BND_X + BND_X)
    # theta_dot (5, 16) and time (10, 21)
    ocp.constraints.idxbx = np.array([5, 10, 16, 21]) 

    # Soft Constraints (Slack Variables)
    # acados manage the soft constraints as SLACK VARIABLES
    # We have the soft constraint on the velocities (3) and on the distance from current and next step to hip
    idxsh_branch = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    idxsh_total = idxsh_branch + [i + 16 for i in idxsh_branch]
    ocp.constraints.idxsh = np.array(idxsh_total)
    # Skip the first 8 decoupling eqns for node 0
    ocp.constraints.idxsh_0 = np.array([i + 8 for i in idxsh_total])
    # theta_dot is now index 0 for op and 2 for pe
    ocp.constraints.idxsbx = np.array([0, 2])

    # These are the weights for the limits (upper and lower) of soft constraints
    # Soft Weights (States first, then Nonlinear constraints)
    Zu_sbx = [1e3]*2
    zu_sbx = [1e2]*2
    Zl_sbx = [1e3]*2
    zl_sbx = [1e2]*2

    Zu_sh_branch = [1e4]*4 + [1e6]*7 + [1e3]*2
    zu_sh_branch = [1e3]*4 + [1e5]*7 + [1e2]*2
    Zl_sh_branch = [0.0]*4 + [0.0]*6 + [1e6] + [1e3]*2
    zl_sh_branch = [0.0]*4 + [0.0]*6 + [1e5] + [1e2]*2

    ocp.cost.Zu = np.array(Zu_sbx + Zu_sh_branch + Zu_sh_branch)
    ocp.cost.zu = np.array(zu_sbx + zu_sh_branch + zu_sh_branch)
    ocp.cost.Zl = np.array(Zl_sbx + Zl_sh_branch + Zl_sh_branch)
    ocp.cost.zl = np.array(zl_sbx + zl_sh_branch + zl_sh_branch)

    # Node 0 soft weights (states are fixed, only nonlinear active)
    ocp.cost.Zu_0 = np.array(Zu_sh_branch + Zu_sh_branch)
    ocp.cost.zu_0 = np.array(zu_sh_branch + zu_sh_branch)
    ocp.cost.Zl_0 = np.array(Zl_sh_branch + Zl_sh_branch)
    ocp.cost.zl_0 = np.array(zl_sh_branch + zl_sh_branch)

    # Constraint on Controls
    dt_min = 0.2 / 4.0 
    dt_max = 0.35 / 4.0
    # [alpha, beta, gamma, dt_var, ax, ay, b]
    bnd_u_min = [0.1, 0.0, 0.0, dt_min, -1.0, -1.0, -10.0]
    bnd_u_max = [0.9, 1, 1, dt_max, 1.0, 1.0, 10.0]
    ocp.constraints.lbu = np.array(bnd_u_min + bnd_u_min)
    ocp.constraints.ubu = np.array(bnd_u_max + bnd_u_max)
    ocp.constraints.idxbu = np.array([4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21])

    ocp.parameter_values = np.zeros(8)

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP' 
    ocp.solver_options.nlp_solver_max_iter = 5

    return AcadosOcpSolver(ocp)