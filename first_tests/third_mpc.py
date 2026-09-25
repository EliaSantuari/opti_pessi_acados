import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import scipy.linalg
import matplotlib.pyplot as plt



'''
08/04/2026
'''


# ------ DEFINISCO I PARAMETRI ------
N = 20
dt = 0.05
g = 9.81
h = 0.38
w = np.sqrt(g / h) # Frequency constant
nx = 4 # (cx, cy, cdotx, cdoty)
nu = 1 # alpha
ny = nx + nu
np_param = 4 # (p0x, p0y, p1x, p1y)


# ------ DEFINISCO IL MODELLO ------
x = cs.SX.sym('x', nx)
u = cs.SX.sym('u', nu)
p = cs.SX.sym('p', np_param)

alpha = u[0]
c = x[0:2]
c_dot = x[2:4]
p0 = p[0:2]
p1 = p[2:4]

cop = p0 + alpha * (p1 - p0)

x_dot = cs.vertcat(
    c_dot,
    w**2 * (c - cop)
)

model = AcadosModel()
model.x = x
model.u = u
model.p = p
model.f_expl_expr = x_dot
model.name = "lipm_step"


# ------ DEFINISCO OCP ------
ocp = AcadosOcp()
ocp.model = model
ocp.parameter_values = np.zeros(np_param) # Inizializzo parametri

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = N * dt


# ------ COST ------
ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.cost_type_e = 'LINEAR_LS'

# Weights
Q = np.diag([100, 100, 1, 1])

R = np.diag([0.01])

ocp.cost.W = scipy.linalg.block_diag(Q,R)
ocp.cost.W_e = Q

# Mapping states and input in constant vectors
ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx, :nx] = np.eye(nx)
ocp.cost.Vu = np.zeros((ny, nu))
ocp.cost.Vu[nx:, :] = np.eye(nu)
ocp.cost.Vx_e = np.eye(nx)

# Target
cx_target = 5
cy_target = 3
c_target = np.array([cx_target, cy_target])

y_ref = np.zeros(ny)
y_ref[0:2] = c_target

y_ref_e = np.zeros(nx)
y_ref_e[0:2] = c_target


ocp.cost.yref = y_ref
ocp.cost.yref_e = y_ref_e


# ------ CONSTRAINTS -------
# Initial conditions
x0 = np.array([0.0, 0.0, 0.0, 0.0])
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



# ------ GENERATORE DI ANDATURA (gait) ------
# Cambio fase ogni T_phase time step
def gait_gen(t, T_phase):
    phase = int(np.floor(t / T_phase)) % 2
    if phase == 0:
        gait_pattern = ['FR', 'RL']
    else:
        gait_pattern = ['FL', 'RR']

    return gait_pattern


# ------ MPC SIMULATION ------
X = x0

sim_steps = 200
T_phase = 0.3

solve_time_hist = []
traj = []
alpha_traj = []
foot_traj = []
gait_history = []

t = 0
PRINT_TIME = 0

# SCOMMENTA PER FERMARE QUANDO RAGGIUNGE TARGET
# while (t < sim_steps and (X[0] - c_target[0])**2 + (X[1] - c_target[1])**2 
#        > 0.01**2):
while t < sim_steps:

    # GENERO L'ANDATURA
    gait_pattern = gait_gen(t * dt, T_phase)
    gait_history.append(gait_pattern)
    if gait_pattern == ['FR', 'RL']:
        p_val = np.array([X[0]+0.2, X[1]-0.2, X[0]-0.2, X[1]+0.2])
    else:
        p_val = np.array([X[0]+0.2, X[1]+0.2, X[0]-0.2, X[1]-0.2])
    foot_traj.append(p_val)


    # Passo i piedi a tutti parametri nell'orizzonte
    for j in range(N):
        solver.set(j, 'p', p_val)

    solver.set(0, 'lbx', X)
    solver.set(0, 'ubx', X) 
    
    # Solve the ocp
    solver.solve()    


    if PRINT_TIME:
         # Computational time
        solve_time = solver.get_stats("time_tot")
        print(f"Computational time at step {t+1}: {solve_time}")
        solve_time_hist.append(solve_time)
        if t == sim_steps-1:
            print(f"AVG. Computational time: {np.mean(solve_time_hist)*1000:.3f}ms")
    
    u_apply = solver.get(0, 'u')
    x_next = solver.get(1, 'x')

    # HISTORY
    traj.append(X.copy())
    alpha_traj.append(u_apply.copy())
    X = x_next
    t += 1


traj = np.array(traj)
foot_traj = np.array(foot_traj)
gait_history = np.array(gait_history)
p0x = foot_traj[:, 0]
p0y = foot_traj[:, 1]
p1x = foot_traj[:, 2]
p1y = foot_traj[:, 3]

mask_phaseA = (gait_history[:, 0] == 'FR')
mask_phaseB = (gait_history[:, 0] == 'FL')



# ------ PLOT ------

time_axis = np.linspace(0, sim_steps * dt, len(traj))
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(time_axis, traj[:, 0])
plt.xlabel("time"); plt.ylabel("CoM X")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(time_axis, traj[:, 1])
plt.xlabel("time"); plt.ylabel("CoM Y")
plt.grid()


plt.figure()
plt.plot(traj[:, 0], traj[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()

plt.figure()
plt.plot(time_axis, alpha_traj)
plt.title("Alpha"); plt.xlabel("time"); plt.ylabel("alpha")
plt.grid()


# PLOT DELLA SAGOMA DEL ROBOT

from matplotlib.patches import Rectangle

# --- Nel tuo blocco di PLOT finale ---
plt.figure()
ax = plt.gca() # Ottieni gli assi correnti per aggiungere le patches

# 1. Disegna i piedi (il tuo codice esistente)
plt.scatter(p0x[mask_phaseA], p0y[mask_phaseA], c='r', marker='o', label='FR')
plt.scatter(p1x[mask_phaseA], p1y[mask_phaseA], c='r', marker='x', label='RL')
plt.scatter(p0x[mask_phaseB], p0y[mask_phaseB], c='b', marker='o', label='FL')
plt.scatter(p1x[mask_phaseB], p1y[mask_phaseB], c='b', marker='x', label='RR')

# 2. Disegna la traiettoria del CoM
plt.plot(traj[:, 0], traj[:, 1], 'k', label='Traj CoM')

# 3. DISEGNA IL CORPO DEL ROBOT (Rettangolo)
width = 0.4  # Larghezza rettangolo
height = 0.4 # Lunghezza rettangolo

# Disegniamo il rettangolo ogni 20 step per non affollare il grafico
for i in range(0, len(traj)):
    # Il rettangolo deve essere centrato in traj[i, 0] e traj[i, 1]
    # L'angolo in basso a sinistra si calcola sottraendo metà dimensione
    corner_x = traj[i, 0] - width/2
    corner_y = traj[i, 1] - height/2
    
    rect = Rectangle((corner_x, corner_y), width, height, 
                     linewidth=1, edgecolor='g', facecolor='g', alpha=0.2)
    ax.add_patch(rect)

plt.legend()
plt.axis('equal')
plt.title("Vista X-Y con corpo del Robot")
plt.grid(True)




plt.show()