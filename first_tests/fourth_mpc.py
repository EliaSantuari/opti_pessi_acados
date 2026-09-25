import casadi as cs
import numpy as np
from numpy import cosh, sinh
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import scipy.linalg
import matplotlib.pyplot as plt




# ------ DEFINISCO I PARAMETRI ------
N = 20
dt = 0.2
g = 9.81
h = 0.38
w = np.sqrt(g / h) # Frequency constant
nx = 8 # (cx, cy, cdotx, cdoty, p0x, p0y, p1x, p1y)
nu = 5 # (p0x_next, p0y_next, p1x_next, p1y_next, alpha)
ny = 9
ny_e = 4
hip_pos = {'FL': np.array([0.2, 0.2]), 
           'FR': np.array([ 0.2, -0.2]), 
           'RL': np.array([-0.2,  0.2]), 
           'RR': np.array([-0.2, -0.2])}


# gait_pattern = [["FR", "RL"], ["FL", "RR"]]
def hip_pos01(hip_pos, c, gait_pattern, i):
    hip_pos0 = c + hip_pos[gait_pattern[i][0]]
    hip_pos1 = c + hip_pos[gait_pattern[i][1]]
    return hip_pos0, hip_pos1


# ------ DEFINISCO IL MODELLO ------
x = cs.SX.sym('x', nx)
u = cs.SX.sym('u', nu)
p_sym = cs.SX.sym('p', 4) # POSIZIONI DELLE ANCHE


ch, sh = cosh(w*dt), sinh(w*dt)

c = x[0:2]
c_dot = x[2:4]
p0 = x[4:6]
p1 = x[6:8]
p0_next = u[0:2]
p1_next = u[2:4]
alpha = u[4]

hip_pos0_p = p_sym[0:2]
hip_pos1_p = p_sym[2:4]


cop = p0 + alpha * (p1 - p0)


x_next = cs.vertcat(
    ch*c + sh/w*c_dot + (1-ch)*cop,
    w*sh*c + ch*c_dot - w*sh*cop,
    p0_next,
    p1_next
)

model = AcadosModel()
model.x = x
model.u = u
model.p = p_sym

model.disc_dyn_expr = x_next
model.name = "lipm_step"


# ------ DEFINISCO OCP ------
ocp = AcadosOcp()
ocp.model = model


ocp.solver_options.N_horizon = N
ocp.solver_options.tf = N * dt


# ------ COST ------
ocp.cost.cost_type = 'NONLINEAR_LS'
ocp.cost.cost_type_e = 'NONLINEAR_LS'

delta_p0 = p0_next - p0
delta_p1 = p1_next - p1

y_expr = cs.vertcat(
    c,
    c_dot, 
    delta_p0,
    delta_p1,
    alpha
)

model.cost_y_expr = y_expr
model.cost_y_expr_e = cs.vertcat(c, c_dot)

# Target
cx_target = 2
cy_target = 2
c_target_final = np.array([cx_target, cy_target])

y_ref = np.zeros(ny)
y_ref[:2] = c_target_final
y_ref[2:4] = np.zeros(2)
y_ref[4:8] = np.zeros(4)
y_ref[8] = 0.5

y_ref_e = np.zeros(ny_e)
y_ref_e[0:2] = c_target_final
y_ref_e[2:4] = np.zeros(2)

# Weights
W = np.diag([
    1, 1,       # c
    2, 2,       # c_dot
    0.5, 0.5,   # p0
    0.5, 0.5,   # p1
    0.5         # alpha
])
ocp.cost.W = W
ocp.cost.W_e = W[:4, :4] # c, c_dot

ocp.cost.yref = y_ref
ocp.cost.yref_e = y_ref_e



# ------ CONSTRAINTS -------
# Initial conditions
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.2, -0.2, -0.2, 0.2])
ocp.constraints.x0 = x0

# Boundaries
h = cs.vertcat(
    cs.norm_2(p0_next - hip_pos0_p),
    cs.norm_2(p1_next - hip_pos1_p)
)
model.con_h_expr = h
ocp.constraints.lh = np.array([0.0, 0.0])  
ocp.constraints.uh = np.array([0.2, 0.2])

ocp.parameter_values = np.zeros(4) # Inizializzo parametri

print(f"Dimensioni h: {model.con_h_expr.shape}")
print(f"Dimensioni lh: {ocp.constraints.lh.shape}")
print(f"Dimensioni uh: {ocp.constraints.uh.shape}")



# cdot
ocp.constraints.lbx = np.array([-1.5, -0.45])
ocp.constraints.ubx = np.array([1.5, 0.45])
ocp.constraints.idxbx = np.array([2, 3])

# alpha
ocp.constraints.lbu = np.array([0.1])
ocp.constraints.ubu = np.array([0.9])
ocp.constraints.idxbu = np.array([4])




# ------ SOLVER OPTIONS ------
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # Faster in robotics
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
ocp.solver_options.integrator_type = 'DISCRETE' # Explicit Runge Kutta
ocp.solver_options.nlp_solver_type = 'SQP_RTI' # Real Time Iteration


# ------ SOLVE THE PROBLEM ------
solver = AcadosOcpSolver(ocp)




# ------ MPC SIMULATION ------
X = x0


solve_time_hist = []
traj = []
u_traj = []
foot_traj = []
gait_history = []



n = 0
t = 0
sim_steps = 50

PRINT_TIME = 1



# SCOMMENTA PER FERMARE QUANDO RAGGIUNGE TARGET
# while (t < sim_steps and (X[0] - c_target[0])**2 + (X[1] - c_target[1])**2 
#        > 0.01**2):
while n < sim_steps+1:
    gait_pattern = []
    if (n % 2 == 0):
        while (len(gait_pattern) <= N):
            gait_pattern += [["FR", "RL"]]
            gait_pattern += [["FL", "RR"]]
    else:
        while (len(gait_pattern) <= N):
            gait_pattern += [["FL", "RR"]]
            gait_pattern += [["FR", "RL"]]

   
    for k in range(N+1):
        x_pred = solver.get(k, 'x')

        # Calcolo posizione delle anche in quell'istante
        h0, h1 = hip_pos01(hip_pos, x_pred[0:2], gait_pattern, k)
        params = np.array([h0[0], h0[1], h1[0], h1[1]])
        solver.set(k, 'p', params)


    solver.set(0, 'lbx', X)
    solver.set(0, 'ubx', X) 

    # Solve the ocp
    solver.solve()    


    if PRINT_TIME:
         # Computational time
        solve_time = solver.get_stats("time_tot")
        print(f"Computational time at step {t:.2f}: {solve_time*1000:.3f} ms")
        solve_time_hist.append(solve_time)
        if n == sim_steps:
            print(f"AVG. Computational time: {np.mean(solve_time_hist)*1000:.3f}ms")
    
    u_apply = solver.get(0, 'u')
    x_next = solver.get(1, 'x')

    # HISTORY
    traj.append(X.copy())
    u_traj.append(u_apply.copy())
    
    X = x_next

    # print("U:", u_apply[0:4])
    # print()
    # print("X:", X)
    # print()
    
    n += 1
    t += dt



traj = np.array(traj)
u_traj = np.array(u_traj)


# ------ PLOTS ------

time_axis = np.linspace(0, sim_steps * dt, len(traj))


plt.figure()
plt.subplot(2, 1, 1); plt.title("COM POSITIONS")
plt.plot(time_axis, traj[:, 0])
plt.xlabel("time"); plt.ylabel("CoM X")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(time_axis, traj[:, 1])
plt.xlabel("time"); plt.ylabel("CoM Y")
plt.grid()



plt.figure()
plt.subplot(2, 1, 1); plt.title("VELOCITIES")
plt.plot(time_axis, traj[:, 2])
plt.xlabel("time"); plt.ylabel("dX")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(time_axis, traj[:, 3])
plt.xlabel("time"); plt.ylabel("dY")
plt.grid()


p0_traj = traj[:, 4:6]
p1_traj = traj[:, 6:8]
plt.figure(); plt.title("TRAJECTORY")
# CoM
plt.plot(traj[:, 0], traj[:, 1], label="CoM")
# Piede 0
plt.scatter(p0_traj[:, 0], p0_traj[:, 1], 
            marker='o', s=20, label="Foot 0")
# Piede 1
plt.scatter(p1_traj[:, 0], p1_traj[:, 1], 
            marker='x', s=20, label="Foot 1")
# Target
plt.scatter(cx_target, cy_target, marker='*', s=100, label="Target")
plt.xlabel('X'); plt.ylabel('Y')
plt.legend()
plt.grid()



plt.figure()
plt.plot(time_axis, u_traj[:, 4])
plt.title("Alpha"); plt.xlabel("time"); plt.ylabel("alpha")
plt.grid()



plt.figure()
plt.subplot(2,2,1); plt.title("p0x")
plt.scatter(time_axis, u_traj[:, 0])
plt.subplot(2,2,2); plt.title("p0y")
plt.scatter(time_axis, u_traj[:, 1])
plt.subplot(2,2,3); plt.title("p1x")
plt.scatter(time_axis, u_traj[:, 2])
plt.subplot(2,2,4); plt.title("p1y")
plt.scatter(time_axis, u_traj[:, 3])



plt.figure()
plt.scatter(traj[:, 4], traj[:, 5])



plt.show()