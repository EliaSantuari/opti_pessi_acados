import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import scipy.linalg
import matplotlib.pyplot as plt

'''
QUESTO MPC TROVA IL CONTROLLO alpha OTTIMO DATA UNA POSIZIONE COSTANTE  INIZIALE DEI PIEDI, QUINDI IL COM PUO` MUOVERSI SOLTANTO ALL'INTERNO DELLA BASE DI SUPPORTO ( p0 + alpha * (p1 - p0) ) (che è una linea) 

VEDI: 31/03/2026 in work_notes
'''
 

# INTRODUCING FEET POSITIONS IN THE STATE
# x = (cx, cy, cdotx, cdoty, p0x, p0y, p1x, p1y)
# u = alpha



# ------ DEFINISCO I PARAMETRI ------
N = 20
dt = 0.05
g = 9.81
h = 0.38
w = np.sqrt(g / h) # Frequency constant
nx = 8 # (cx, cy, cdotx, cdoty, p0x, p0y, p1x, p1y)
nu = 1 # alpha
ny = nx + nu


# ------ DEFINISCO IL MODELLO ------
x = cs.SX.sym('x', nx)
u = cs.SX.sym('u', nu)

alpha = u[0]

c = x[0:2]
c_dot = x[2:4]
p0 = x[4:6]
p1 = x[6:8]

cop = p0 + alpha * (p1 - p0)

x_dot = cs.vertcat(
    c_dot,
    w ** 2 * (c - cop),
    cs.SX.zeros(4) # Piedi fermi
)

model = AcadosModel()
model.x = x
model.u = u
model.f_expl_expr = x_dot
model.name = "lipm_feet"


# ------ DEFINISCO OCP ------
ocp = AcadosOcp()
ocp.model = model

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = N * dt


# ------ COST ------
# VOGLIO UNA COST FUNCTION LEAST SQUARE LINEARE
ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.cost_type_e = 'LINEAR_LS' # terminal cost


# Weights
Q = np.diag([100, 100, 1, 1, 10, 10, 10, 10])
# cx, cy, cdotx, cdoty, p0x, p0y, p1x, p1y
R = np.diag([0.01])
# alpha

ocp.cost.W = scipy.linalg.block_diag(Q, R)
ocp.cost.W_e = Q

# Mapping states and input in constant vectors
ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx, :nx] = np.eye(nx)
ocp.cost.Vu = np.zeros((ny, nu))
ocp.cost.Vu[nx:, :] = np.eye(nu)
ocp.cost.Vx_e = np.eye(nx)

# Target
cx_target = 0.1
cy_target = 0.15
c_goal = np.array([cx_target, cy_target])

y_ref = np.zeros(ny)
y_ref[0:2] = c_goal

y_ref_e = np.zeros(nx)
y_ref_e[:2] = c_goal

# Posizioni piedi iniziali
y_ref[4:6] = [0.0, 0.2] # p0x, p0y
y_ref[6:8] = [0.0, 0.0] # p1x, p1y

y_ref_e[4:6] = [0.0, 0.2] # p0x, p0y
y_ref_e[6:8] = [0.0, 0.0] # p1x, p1y

ocp.cost.yref = y_ref
ocp.cost.yref_e = y_ref_e


# ------ CONSTRAINTS ------
# Initial conditions
x0 = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0])
ocp.constraints.x0 = x0

# Boundaries
ocp.constraints.lbu = np.array([0.1])
ocp.constraints.ubu = np.array([0.9])
ocp.constraints.idxbu = np.array([0])


# ------ SOLVER OPTIONS ------
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # Faster in robotics
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
ocp.solver_options.integrator_type = 'ERK' # Explicit Runge Kutta
ocp.solver_options.nlp_solver_type = 'SQP_RTI' # Real Time Iteration


# ------ SOLVE THE PROBLEM ------
solver = AcadosOcpSolver(ocp)


# ------ MPC SIMULATION ------
x_current = x0

sim_steps = 50

traj = []
alpha_traj = []
comp_time = []
WARM_START = 0


for i in range(sim_steps):
    solver.set(0, 'lbx', x_current)
    solver.set(0, 'ubx', x_current) 
    # Solve the problem
    status = solver.solve() 
    # Computational time
    solve_time = solver.get_stats("time_tot")
    print(f"Computational time at step {i+1}: {solve_time}")
    comp_time.append(solve_time)
    if i == sim_steps - 1:
        print(f"Average computational time: {np.mean(comp_time) * 1e3:.4f}ms") 
    # Apply optimal control
    u_apply = solver.get(0, 'u')
    x_next = solver.get(1, 'x') 
    traj.append(x_current.copy())
    alpha_traj.append(u_apply.copy())
    x_current = x_next  
    # WARM START
    if WARM_START:
        for j in range(N - 1):
            solver.set(j, 'x', solver.get(j + 1, 'x'))
            solver.set(j, 'u', solver.get(j + 1, 'u'))
        solver.set(N - 1, 'x', solver.get(N, 'x'))
        

traj = np.array(traj)


# ------ PLOTS ------

time_axis = np.linspace(0, sim_steps * dt, len(traj))
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(time_axis, traj[:, 0])
plt.xlabel("time"); plt.ylabel("CoM X")
plt.subplot(2, 1, 2)
plt.plot(time_axis, traj[:, 1])
plt.xlabel("time"); plt.ylabel("CoM Y")

plt.figure()
plt.plot(time_axis, alpha_traj)
plt.title("Alpha"); plt.xlabel("time"); plt.ylabel("alpha")

plt.figure()
plt.scatter(traj[:, 4], traj[:, 5], color = 'blue', label = "p0")
plt.scatter(traj[:, 6], traj[:, 7], color = 'red', label = "p1")
plt.title("Feet Position")
plt.legend()

plt.figure()
plt.plot(traj[:, 0], traj[:, 1])
plt.title("Trajectory"); plt.xlabel("X"); plt.ylabel("Y")

plt.show()