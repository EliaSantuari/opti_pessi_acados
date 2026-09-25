import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import scipy.linalg

# ------ DEFINISCO I PARAMETRI ------
N = 20
dt = 0.05
g = 9.81
h = 0.38
w = np.sqrt(g / h) # Frequency constant
nx = 4 # (cx, cy, cdx, cdy)
nu = 2 # (copx, copy)
ny = nx + nu

# ------ DEFINISCO IL MODELLO ------
x = cs.SX.sym('x', nx) # (cx, cy, cdx, cdy)
u = cs.SX.sym('u', nu) # (copx, copy)

x_dot = cs.vertcat(
    x[2],
    x[3],
    w ** 2 * (x[0] - u[0]),
    w ** 2 * (x[1] - u[1])
)

model = AcadosModel()
model.x = x
model.u = u 
model.f_expl_expr = x_dot
model.name = 'limp_ocp'


# ------ DEFINISCO OCP ------
ocp = AcadosOcp()
ocp.model = model

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = N * dt


# ------ COST ------
ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.cost_type_e = 'LINEAR_LS' # terminal cost

# Weights
w_p = 10
w_v = 1
w_u = 0.1
Q = np.diag([w_p, w_p, w_v, w_v])
R = np.diag([w_u, w_u])

# y = (x, u)
ocp.cost.W = scipy.linalg.block_diag(Q, R)
ocp.cost.W_e = Q

# Mapping states and input in constant vectors
ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx, :nx] = np.eye(nx)
ocp.cost.Vu = np.zeros((ny, nu))
ocp.cost.Vu[nx:, :] = np.eye(nu)
ocp.cost.Vx_e = np.eye(nx)

# Target
cx_target = 1.0
cy_target = 1.0
c_goal = np.array([cx_target, cy_target])

y_ref = np.zeros(ny)
y_ref[:2] = c_goal

y_ref_e = np.zeros(nx)
y_ref_e[:2] = c_goal

ocp.cost.yref = y_ref
ocp.cost.yref_e = y_ref_e


# ------ CONSTRAINTS ------
# Initial conditions
cx_init = 0.0
cy_init = 0.0
cxdot_init = 0.0
cydot_init = 0.0
ocp.constraints.x0 = np.array([cx_init, cy_init, cxdot_init, cydot_init])

# Boundaries
cop_max = 2
ocp.constraints.lbu = np.array([-cop_max, -cop_max])
ocp.constraints.ubu = np.array([cop_max, cop_max])
ocp.constraints.idxbu = np.array([0, 1])


# ------ SOLVER OPTIONS ------
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # Faster in robotics
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
ocp.solver_options.integrator_type = 'ERK' # Explicit Runge Kutta
ocp.solver_options.nlp_solver_type = 'SQP_RTI' # Real Time Iteration


# ----- SOLVE THE PROBLEM -----
solver = AcadosOcpSolver(ocp)



# ------ MPC SIMULATION ------
x_current = np.array([cx_init, cy_init, cxdot_init, cydot_init])


sim_steps = 50

traj = []

for i in range(sim_steps):
    solver.set(0, 'lbx', x_current)
    solver.set(0, 'ubx', x_current)

    status = solver.solve()
    if status != 0:
        raise Exception(f"acados returned status: {status}.")
    
    u_apply = solver.get(0, 'u')
    x_next = solver.get(1, 'x')

    traj.append(x_current.copy())
    x_current = x_next

traj = np.array(traj)

# ------ PLOT ------
import matplotlib.pyplot as plt
import numpy as np

# Genera il vettore tempo corretto per la durata della simulazione
# sim_steps è il numero di iterazioni fatte nel loop
time_axis = np.linspace(0, sim_steps * dt, len(traj))
plt.figure()
plt.subplot(2, 1, 1); plt.plot(time_axis, traj[:, 0])
plt.subplot(2, 1, 2); plt.plot(time_axis, traj[:, 1])


plt.figure(figsize=(8, 6))

# Usiamo scatter per colorare ogni punto in base al tempo
sc = plt.scatter(traj[:,0], traj[:,1], c=time_axis, cmap='viridis', s=10, label="CoM Path")
plt.plot(traj[:,0], traj[:,1], alpha=0.3) # Linea sottile di collegamento

plt.scatter(c_goal[0], c_goal[1], color='red', marker='X', s=100, label="Goal")

# Aggiunge la barra del tempo
cbar = plt.colorbar(sc)
cbar.set_label('Tempo [s]')

plt.xlabel('Posizione X [m]')
plt.ylabel('Posizione Y [m]')
plt.title('Traiettoria CoM nel tempo')
plt.legend()
plt.axis("equal")
plt.grid(True)


plt.show()