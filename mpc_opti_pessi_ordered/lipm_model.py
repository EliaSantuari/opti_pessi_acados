import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
from config import RobotConfig, MPCWeights, Limits, SimulationConfig

def create_lipm_ocp(
        N=8,
        c_target=np.array([0.0, 0.0]),
        theta_target=0.0,
        x_init=np.zeros(22),
        robot_cfg: RobotConfig = RobotConfig(),
        weights: MPCWeights = MPCWeights(),
        limits: Limits = Limits(),
        sim_conf: SimulationConfig = SimulationConfig()
    ) -> AcadosOcpSolver:
 
    #### CONFIGURATION ####
    # ---- Hip offsets ---
    off_x = robot_cfg.off_x
    off_y = robot_cfg.off_y

    # ---- Weights ----
    W_diag_op = [
        weights.tracking_xy, weights.tracking_xy,       # 0,1: Tracking x, y
        weights.vel_alignment,                          # 2: Velocity alignment - penalize lateral walk
        weights.vel_xy, weights.vel_xy,                 # 3,4: velocity x, y
        weights.yaw_rate,                               # 5: Yaw rate (penalization on fast rotations)
        weights.time_weight,                            # 6: Time
        weights.anti_skating, weights.anti_skating,     # 7-8: Anti-Skating foot 0
        weights.anti_skating, weights.anti_skating,     # 9-10: Anti-Skating foot 1
        weights.posture, weights.posture,               # 11-12: Posture (keep hip above foot 0)
        weights.posture, weights.posture,               # 13-14: Posture (keep hip above foot 1)
        weights.alpha_weight,                           # 15: Alpha
        weights.dt_weight,                              # 16: dt
    ]
    # Regularization weight for the pessimistic branch to avoid null-space explosion
    W_diag_pe = [w * 1e-3 for w in W_diag_op] 
    # cost_y_expr_e_op = cs.vertcat(c_op, c_dot_op, theta_dot_op, x_op[10])
    W_end_op = [
        weights.tracking_xy, weights.tracking_xy,
        weights.vel_xy, weights.vel_xy,
        weights.yaw_rate, 
        weights.time_weight
    ]
    W_end_pe = [w * 1e-3 for w in W_end_op] 

    # dt_nominal for the reference
    dt_nominal = (limits.dt_min + limits.dt_max) / (2*sim_conf.steps_per_phase) # We must fix a nominal value, otherwise the solver would sent it to zero to avoid errors

    # ---- Limits ----
    max_ext_sq = robot_cfg.max_ext_sq
    # (h_dist0_curr, h_dist1_curr, h_dist0_next, h_dist1_next, fric0, fric1) 
    lh_robot = [0.0, 0.0, 0.0, 0.0, -1e6, -1e6]
    uh_robot = [max_ext_sq, max_ext_sq, max_ext_sq, max_ext_sq, 0.0, 0.0]

    # (eq_norm_a, eq_s[0], eq_s[1], eq_s[2], eq_s[3], eq_p[0], eq_p[1], eq_obs) 
    # eq_s and eq_p ensure hips and feet to be in the negative semiplane (<= 0)
    lh_obs = [0.5, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6, 0.0] 
    uh_obs  = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e6]

    # (v_loc_x, v_loc_y) limits local velocities
    lh_vel = [-limits.v_max_x, -limits.v_max_y] 
    uh_vel = [limits.v_max_x, limits.v_max_y] 

    # ---- Constraints on state ----
    # x = [cx, cy, th, cx_dot, cy_dot, th_dot, p0x, p0y, p1x, p1y, time]
    # we only need to limit theta_dot and total time (indeces [5, 10, 16, 21])
    bnd_x = [-limits.theta_dot, 0.0]
    BND_X = [limits.theta_dot, 100.0]

    # ---- Constraints on controls ----
    # u_op = [p0_nx, p0_ny, p1_nx, p1_ny, alpha, beta, gamma, dt_var, ax, ay, b]
    dt_min = limits.dt_min / sim_conf.steps_per_phase
    dt_max = limits.dt_max / sim_conf.steps_per_phase
    # [alpha, beta, gamma, dt_var, ax, ay, b]
    bnd_u_min = [limits.alpha_min, 0.0, 0.0, dt_min, -1.0, -1.0, -100.0]
    bnd_u_max = [limits.alpha_max, 1.0, 1.0, dt_max, 1.0, 1.0, 100.0]

    w = np.sqrt(robot_cfg.g / robot_cfg.h_com) # Frequency constant

    #### MODEL AND OCP ####

    # Variables and parameters
    x_op = cs.SX.sym('x_op', 11) # cx, cy, theta, cdotx, cdoty, thetadot, p0x, p0y, p1x, p1y, time
    x_pe = cs.SX.sym('x_pe', 11)
    x = cs.vertcat(x_op, x_pe)
    
    u_op = cs.SX.sym('u_op', 11) # p0next_x, p0next_y, p1next_x, p1next_y, alpha, beta, gamma, dt, ax, ay, b
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
        F_tot = robot_cfg.m * c_ddot
        f0_x = beta * F_tot[0]
        f1_x = (1.0 - beta) * F_tot[0]
        
        f0_y = gamma * F_tot[1]
        f1_y = (1.0 - gamma) * F_tot[1]
        
        f0 = cs.vertcat(f0_x, f0_y)
        f1 = cs.vertcat(f1_x, f1_y)

        tau_z0 = (p0[0] - c[0]) * f0[1] - (p0[1] - c[1]) * f0[0]
        tau_z1 = (p1[0] - c[0]) * f1[1] - (p1[1] - c[1]) * f1[0]
        tau_z = tau_z0 + tau_z1

        # Numerical simulation (integration)
        c_next = ch * c + (sh / w) * c_dot + (1 - ch) * cop
        c_dot_next = w * sh * c + ch * c_dot - w * sh * cop
        theta_next = theta + theta_dot * dt_var
        theta_dot_next = theta_dot + (dt_var / robot_cfg.Iz) * tau_z

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

    # --- Vel heading ---
    # ((vx^2+vy^2)cos(theta)^2-vx^2)^2 + ((vx^2+vy^2)sin(theta)^2-vy^2)^2
    vx_op, vy_op = c_dot_op[0], c_dot_op[1]
    v_sq_op = vx_op**2 + vy_op**2
    # vel_err_op = (v_sq_op*(cs.cos(theta_op))**2-vx_op**2)**2+(v_sq_op*(cs.sin(theta_op))**2-vy_op**2)**2
    vel_err_op = -vx_op * cs.sin(theta_op) + vy_op * cs.cos(theta_op)


    # COSTS
    delta_p0_move_op = p0_next_op - p0_op  # anti-skating
    delta_p1_move_op = p1_next_op - p1_op
    hip_err_0_op = p0_next_op - hip0_expr_next_op # Penalize distance from the shoulder
    hip_err_1_op = p1_next_op - hip1_expr_next_op

    cost_y_expr_op = cs.vertcat(
        c_op, vel_err_op, c_dot_op, theta_dot_op, x_op[10],
        delta_p0_move_op, delta_p1_move_op, 
        hip_err_0_op, hip_err_1_op, 
        u_op[4] - 0.5, u_op[7] - dt_nominal
    )
    cost_y_expr_e_op = cs.vertcat(c_op, c_dot_op, theta_dot_op, x_op[10])

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

    vx_pe, vy_pe = c_dot_pe[0], c_dot_pe[1]
    v_sq_pe = vx_pe**2 + vy_pe**2
    # vel_err_pe = (v_sq_pe*(cs.cos(theta_pe))**2-vx_pe**2)**2+(v_sq_pe*(cs.sin(theta_pe))**2-vy_pe**2)**2
    vel_err_pe = -vx_pe * cs.sin(theta_pe) + vy_pe * cs.cos(theta_pe)

    delta_p0_move_pe = p0_next_pe - p0_pe  
    delta_p1_move_pe = p1_next_pe - p1_pe
    hip_err_0_pe = p0_next_pe - hip0_expr_next_pe 
    hip_err_1_pe = p1_next_pe - hip1_expr_next_pe

    cost_y_expr_pe = cs.vertcat(
        c_pe, vel_err_pe, c_dot_pe, theta_dot_pe, x_pe[10],
        delta_p0_move_pe, delta_p1_move_pe, 
        hip_err_0_pe, hip_err_1_pe, 
        u_pe[4] - 0.5, u_pe[7] - dt_nominal
    )
    cost_y_expr_e_pe = cs.vertcat(c_pe, c_dot_pe, theta_dot_pe, x_pe[10])

    # Combine Expressions (40 elements for intermediate, 14 for terminal)
    # Define cost function: (y-y_ref).T W (y-y_ref)
    model.cost_y_expr = cs.vertcat(cost_y_expr_op, cost_y_expr_pe)
    model.cost_y_expr_e = cs.vertcat(cost_y_expr_e_op, cost_y_expr_e_pe)

    
    
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W = np.diag(W_diag_op + W_diag_pe)
    ocp.cost.W_e = np.diag(W_end_op + W_end_pe)

    # Reference (Updated lengths)
    y_ref = np.zeros(34)
    y_ref[:2] = c_target
    y_ref[17:19] = c_target
    
    y_ref_e = np.zeros(12)
    y_ref_e[:2] = c_target
    y_ref_e[6:8] = c_target

    ocp.cost.yref, ocp.cost.yref_e = y_ref, y_ref_e

    # --- HELPER FUNCTION FOR CONSTRAINTS ---
    def compute_constraints(x_curr, u_curr, c_next, theta_next, f0, f1, r_obstacle):

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
        fric0 = f0[0]**2 + f0[1]**2 - (robot_cfg.mu * (1.0 - u_curr[4]) * robot_cfg.m * robot_cfg.g)**2 + 1e-3
        fric1 = f1[0]**2 + f1[1]**2 - (robot_cfg.mu * u_curr[4] * robot_cfg.m * robot_cfg.g)**2 + 1e-3

        # Separating plane Constraints
        # 1) Compute the norm of a == 1
        eq_norm_a = a[0]**2 + a[1]**2 

        # 2) 4 angles of the robot must stay in the negative semiplane(we  don't want the robot to be inside the obstacle)
        corners = [cs.vertcat(off_x, off_y), cs.vertcat(-off_x, off_y), cs.vertcat(-off_x, -off_y), cs.vertcat(off_x, -off_y)]
        eq_s = [cs.dot(a, c_next + R_th @ loc) + b for loc in corners]
        eq_p = [cs.dot(a, p0_next) + b, cs.dot(a, p1_next) + b]  

        # 4) Dynamic obstacle should be in the positive semiplane and distant at least r_obs
        eq_obs = cs.dot(a, y0_obs) + b - (r_obstacle * (1 + 0.8))

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


    # Terminal constraint to ensure that CoM is on top of support line
    # (c_x - p0_x)*(p1_y - p0_y) - (c_y - p0_y)*(p1_x - p0_x) == 0
    com_supp_line_op = (x_op[0] - x_op[6]) * (x_op[9] - x_op[7]) - (x_op[1] - x_op[7]) * (x_op[8] - x_op[6])
    com_supp_line_pe = (x_pe[0] - x_pe[6]) * (x_pe[9] - x_pe[7]) - (x_pe[1] - x_pe[7]) * (x_pe[8] - x_pe[6])

    model.con_h_expr_e = cs.vertcat(com_supp_line_op, com_supp_line_pe)




    # Decouple separating planes: Only the physical 8 controls are forced equal at node 0
    u_diff = u_op[0:8] - u_pe[0:8] # 8 equations
    model.con_h_expr_0 = cs.vertcat(u_diff, con_op, con_pe)

    # Limits
    lh = lh_robot + lh_obs + lh_vel
    ocp.constraints.lh = np.array(lh + lh)
    
    uh = uh_robot + uh_obs + uh_vel
    ocp.constraints.uh = np.array(uh + uh)

    # Terminal limits
    ocp.constraints.lh_e = np.array([-1e-2, -1e-2])
    ocp.constraints.uh_e = np.array([1e-2, 1e-2])

    # Initial constraints (36 equations: 8 for u_diff == 0 + 28)
    ocp.constraints.lh_0 = np.array([0.0]*8 + lh + lh)
    ocp.constraints.uh_0 = np.array([0.0]*8 + uh + uh)

    # Constraints on States
    ocp.constraints.x0 = x_init
    ocp.constraints.lbx = np.array(bnd_x + bnd_x) 
    ocp.constraints.ubx = np.array(BND_X + BND_X)
    # theta_dot (5, 16) and time (10, 21)
    ocp.constraints.idxbx = np.array([5, 10, 16, 21]) 

    # ---- Soft Constraints (Slack Variables) -----
    # acados manage the soft constraints as SLACK VARIABLES
    # We have 16 equations to be constrained in h_con, some of them must be softened to simplify the life for the solver
    # We have the soft constraint on the velocities (3) and on the distance from current and next step to hip

    # [0,1,2,3]: distances of feet-hips
    # [4,5,6]: friction cones and a==1 kept as hard
    # [7,8,9,10]: Corners collision
    # [11,12]: Feet collision
    # [13]: Obstacle collision
    # [14,15]: Local velocity limits
    idxsh_branch = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 14, 15]
    idxsh_total = idxsh_branch + [i + 16 for i in idxsh_branch]
    ocp.constraints.idxsh = np.array(idxsh_total)

    # Soft constraints on non linear constraints
    # Z: quadratic penalty - z: linear penalty
    # [4x Kinematic] + [7x Collision] + [2x velocities]
    Zu_sh_branch = [1e4]*4 + [1e6]*6 + [1e3]*2
    zu_sh_branch = [1e3]*4 + [1e5]*6 + [1e2]*2
    Zl_sh_branch = [0.0]*4 + [0.0]*6 + [1e3]*2
    zl_sh_branch = [0.0]*4 + [0.0]*6 + [1e2]*2


    # Soft constraints on state limits
    # Skip the first 8 decoupling eqns for node 0
    ocp.constraints.idxsh_0 = np.array([i + 8 for i in idxsh_total])
    # theta_dot is now index 0 for op and 2 for pe
    ocp.constraints.idxsbx = np.array([0, 2])

    # These are the weights for the limits (upper and lower) of soft constraints
    # softening only the weight on theta_dot
    Zu_sbx = [1e3]*2
    zu_sbx = [1e2]*2
    Zl_sbx = [1e3]*2
    zl_sbx = [1e2]*2

    
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
    ocp.constraints.lbu = np.array(bnd_u_min + bnd_u_min)
    ocp.constraints.ubu = np.array(bnd_u_max + bnd_u_max)
    ocp.constraints.idxbu = np.array([4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21])

    ocp.parameter_values = np.zeros(8)

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 5
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    # regolarization for the hessian diagonal elements
    ocp.solver_options.levenberg_marquardt = 1e-2
    ocp.solver_options.qp_solver_tol_stat = 1e-4
    ocp.solver_options.qp_solver_tol_eq = 1e-4
    ocp.solver_options.qp_solver_tol_ineq = 1e-4

    return AcadosOcpSolver(ocp)