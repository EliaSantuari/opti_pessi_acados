import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from lipm_model import create_lipm_ocp
from gait_planner import GaitPlanner
from visualizer import plot_simulation_results
from config import RobotConfig, MPCWeights, SimulationConfig, ObstacleConfig, Limits



def main():
    # ---- Settings and initialization -----
    sim_cfg = SimulationConfig()
    robot_cfg = RobotConfig()
    obs_cfg = ObstacleConfig()
    weights = MPCWeights()
    limits = Limits()


    # Push simulation
    PUSH = 1
    push_step = 50


    # steps_per_phase indicates the numer of time steps after which the robot changes feet
    gait_planner = GaitPlanner(off_x=robot_cfg.off_x, off_y=robot_cfg.off_y, steps_per_phase=sim_cfg.steps_per_phase)

    # ---- Initialization of the state ----
    X_sim = np.zeros(11)
    X_sim[0:2] = robot_cfg.x_init
    X_sim[2] = robot_cfg.theta_init
    # Initialization of the gait
    init_gait = gait_planner.get_gait_horizon(0, sim_cfg.N_horizon)[0]
    initial_hips = gait_planner.compute_hip_positions(X_sim[0:3], init_gait)
    # Initial position of the selected feet exactly underneath the hips
    X_sim[6:8] = initial_hips[0:2]
    X_sim[8:10] = initial_hips[2:4]

    X_aug = np.concatenate([X_sim, X_sim])

    # Initialization lists for warm start
    X_sol_t_prec = [X_aug for _ in range(sim_cfg.N_horizon + 1)]

    u_init = np.zeros(22)
    dt_nominal = limits.dt_max
    u_init[7] = dt_nominal
    u_init[18] = dt_nominal
    U_sol_t_prec = [u_init for _ in range(sim_cfg.N_horizon)]


    # ---- Create the OCP ----
    solver = create_lipm_ocp(
        N=sim_cfg.N_horizon, 
        c_target=sim_cfg.c_target, 
        theta_target=sim_cfg.theta_target, 
        x_init=X_aug,
        robot_cfg=robot_cfg,
        weights=weights,
        sim_conf=sim_cfg    
        )

    # ---- Data logging structures ----
    history_X = []
    history_U = []
    history_obs = []
    # Lists for tracking distances between feet and hips
    dist_p0_curr, dist_p1_curr, dist_p0_next, dist_p1_next = [], [], [], []
    # Computational time history
    time_hist = []

    foot_positions_world = {'FL': [], 'FR': [], 'RL': [], 'RR': []}

    t_global = 0.0

    # Target reached flag
    reached_target_flag = 0


    # ---- Update references and parameters over horizon
    yref = np.zeros(34)
    yref[0:2] = sim_cfg.c_target
    yref[17:19] = sim_cfg.c_target
    # 
    yref_e = np.zeros(12)
    yref_e[0:2] = sim_cfg.c_target
    yref_e[6:8] = sim_cfg.c_target




    print("--- Starting Simulation MPC ---")

    for step in range(sim_cfg.sim_steps):
        #### Definition of the obstacle behaviour ####
        if obs_cfg.obs_type == "static":
            obs_pos = obs_cfg.pos_init
        elif obs_cfg.obs_type == "dynamic":
            p_top = obs_cfg.top_pos_dyn
            p_bot = obs_cfg.bot_pos_dyn
            # Vettore direzione e distanza tra i due punti
            vec = p_bot - p_top
            dist_tot = np.linalg.norm(vec)
            if dist_tot > 1e-5:
                dir_u = vec / dist_tot # Vettore unitario da top a bot
                # Distanza totale che l'ostacolo avrebbe percorso in linea retta
                s_percorso = obs_cfg.speed * t_global
                # Calcolo della posizione nel ciclo di andata e ritorno (lunghezza totale = 2 * dist_tot)
                s_ciclo = s_percorso % (2 * dist_tot)
                if s_ciclo <= dist_tot:
                    # Fase di andata (da top a bot)
                    obs_pos = p_top + dir_u * s_ciclo
                else:
                    # Fase di ritorno (da bot a top)
                    obs_pos = p_bot - dir_u * (s_ciclo - dist_tot)
            else:
                # Se i due punti coincidono
                obs_pos = p_top
        elif obs_cfg.obs_type == "adversarial":
            if step == 0:
                obs_pos = obs_cfg.pos_init
            robot_pos = X_sim[0:2]
            dir_to_robot = robot_pos - obs_pos
            dist_to_robot = np.linalg.norm(dir_to_robot)
            if dist_to_robot > 1e-3:
                dir_to_robot = dir_to_robot / dist_to_robot
            dt_step = dt_chosen if 'dt_chosen' in locals() else (limits.dt_max)
            obs_pos = obs_pos + dir_to_robot * obs_cfg.speed * dt_step
        elif obs_cfg.obs_type == "circular":
            # Estraiamo centro e posizione iniziale
            center_x, center_y = obs_cfg.center[0], obs_cfg.center[1]
            start_x, start_y = obs_cfg.pos_init[0], obs_cfg.pos_init[1]
            # 1. Calcoliamo il raggio effettivo come distanza tra centro e partenza
            dx = start_x - center_x
            dy = start_y - center_y
            radius = np.sqrt(dx**2 + dy**2)
            # Evitiamo divisioni per zero se partenza e centro coincidono
            if radius < 1e-5:
                obs_pos = np.array([start_x, start_y])
            else:
                # 2. Calcoliamo l'angolo di partenza (fase iniziale) per t=0
                theta_0 = np.arctan2(dy, dx)
                # 3. Velocità angolare (omega = v / r)
                # Il segno definisce il verso (es. positivo = antiorario, negativo = orario)
                omega = obs_cfg.speed / radius 
                # 4. Equazioni parametriche del cerchio con fase iniziale
                obs_pos = np.array([
                    center_x + radius * np.cos(omega * t_global + theta_0),
                    center_y + radius * np.sin(omega * t_global + theta_0)
                ])
        history_obs.append(obs_pos.copy())




        # Get the current gait 
        current_gait = gait_planner.get_gait_horizon(step, sim_cfg.N_horizon)[0]

        # Set to zero the time for this loop
        X_sim[10] = 0.0 # Used only for propagation of uncertainty of obstacle in pessimistic branch
        X_aug = np.concatenate([X_sim, X_sim])
        # X_aug[21] = 0.0
        
        # Set initial state
        solver.set(0, "lbx", X_aug)
        solver.set(0, "ubx", X_aug)


        ### Set the parameters
        # Get the horizon of the gait
        gait_horizon = gait_planner.get_gait_horizon(step, sim_cfg.N_horizon)

        for k in range(sim_cfg.N_horizon+1):
            gait_k = gait_horizon[k]
            offset_0 = gait_planner.hip_offsets[gait_k[0]]
            offset_1 = gait_planner.hip_offsets[gait_k[1]]
            r_dyn = obs_cfg.r_obs + 0.1
            
            # Set params
            p_val = np.hstack([offset_0, offset_1, obs_pos, r_dyn, obs_cfg.y_dot_max])
            solver.set(k, "p", p_val)                    


        ### I need to warm up the controller and the state
        for k in range(sim_cfg.N_horizon):
            gait_k = gait_horizon[k]

            # Indeces for shifts: last step repeat last value
            idx_shift_u = k + 1 if k < (sim_cfg.N_horizon-1) else k
            idx_shift_x = k + 1

            x_prev = X_sol_t_prec[idx_shift_x]
            u_prev = U_sol_t_prec[idx_shift_u]

            # Optimistic
            com_guess_op = x_prev[0:2]
            theta_guess_op = x_prev[2]
            vel_guess_op = x_prev[3:5]
            vel_ang_guess_op = x_prev[5]

            # Pessimistic
            com_guess_pe = x_prev[11:13]
            theta_guess_pe = x_prev[13]
            vel_guess_pe = x_prev[14:16]
            vel_ang_guess_pe = x_prev[16]

            # Guess of the hips position
            hips_guess_op = gait_planner.compute_hip_positions([com_guess_op[0], com_guess_op[1], theta_guess_op], gait_k)
            hip0_guess_op = hips_guess_op[0:2]
            hip1_guess_op = hips_guess_op[2:4]

            hips_guess_pe = gait_planner.compute_hip_positions([com_guess_pe[0], com_guess_pe[1], theta_guess_pe], gait_k)
            hip0_guess_pe = hips_guess_pe[0:2]
            hip1_guess_pe = hips_guess_pe[2:4]

            # Distances of com and hips from obs
            dist_com_op = np.linalg.norm(obs_pos - com_guess_op)
            dist_hip0_op = np.linalg.norm(obs_pos - hip0_guess_op)
            dist_hip1_op = np.linalg.norm(obs_pos - hip1_guess_op)

            dist_com_pe = np.linalg.norm(obs_pos - com_guess_pe)
            dist_hip0_pe = np.linalg.norm(obs_pos - hip0_guess_pe)
            dist_hip1_pe = np.linalg.norm(obs_pos - hip1_guess_pe)

            # Who is the closest?
            min_dist_op = min(dist_com_op, dist_hip0_op, dist_hip1_op)
            min_dist_pe = min(dist_com_pe, dist_hip0_pe, dist_hip1_pe)

            # Take the closer point
            if min_dist_op == dist_hip0_op:
                closest_pt_op = hip0_guess_op
            elif min_dist_op == dist_hip1_op:
                closest_pt_op = hip1_guess_op
            else:
                closest_pt_op = com_guess_op

            if min_dist_pe == dist_hip0_pe:
                closest_pt_pe = hip0_guess_pe
            elif min_dist_pe == dist_hip1_pe:
                closest_pt_pe = hip1_guess_pe
            else:
                closest_pt_pe = com_guess_pe

            # Compute the semiplanes
            dir_to_obs_op = obs_pos - closest_pt_op
            dist_to_obs_op = np.linalg.norm(dir_to_obs_op) + 1e-5
            a_guess_op = dir_to_obs_op / dist_to_obs_op
            b_guess_op = -np.dot(a_guess_op, (closest_pt_op + obs_pos) / 2.0)

            dir_to_obs_pe = obs_pos - closest_pt_pe
            dist_to_obs_pe = np.linalg.norm(dir_to_obs_pe) + 1e-5
            a_guess_pe = dir_to_obs_pe / dist_to_obs_pe
            b_guess_pe = -np.dot(a_guess_pe, (closest_pt_pe + obs_pos) / 2.0)

            # Construction of U_guess
            u_guess_op = np.array([
                hips_guess_op[0], hips_guess_op[1], hips_guess_op[2], hips_guess_op[3],
                0.5,                            # alpha
                0.0, 0.0,                       # beta, gamma
                u_prev[7],           # time
                a_guess_op[0], a_guess_op[1],
                b_guess_op
            ])
            u_guess_pe = np.array([
                hips_guess_pe[0], hips_guess_pe[1], hips_guess_pe[2], hips_guess_pe[3],
                0.5,                            # alpha
                0.0, 0.0,                       # beta, gamma
                u_prev[18],           # time
                a_guess_pe[0], a_guess_pe[1],
                b_guess_pe
            ])
            # First steps must be the same! We could use the pessimistic to be more conservative
            if k == 0:
                u_guess_op = u_guess_pe

            solver.set(k, 'u', np.concatenate([u_guess_op, u_guess_pe]))

            # Construction of X_guess
            if k > 0:
                x_guess_op = np.zeros(11)
                x_guess_op[0:2] = com_guess_op
                x_guess_op[2] = theta_guess_op
                x_guess_op[3:5] = vel_guess_op
                x_guess_op[5] = vel_ang_guess_op
                x_guess_op[6:10] = x_prev[6:10]
                x_guess_op[10] = u_prev[7]

                x_guess_pe = np.zeros(11)
                x_guess_pe[0:2] = com_guess_pe
                x_guess_pe[2] = theta_guess_pe
                x_guess_pe[3:5] = vel_guess_pe
                x_guess_pe[5] = vel_ang_guess_pe
                x_guess_pe[6:10] = x_prev[17:21]
                x_guess_pe[10] = u_prev[18]

                solver.set(k, 'x', np.concatenate([x_guess_op, x_guess_pe]))
        solver.set(sim_cfg.N_horizon, 'x', np.concatenate([x_guess_op, x_guess_pe]))

        
            
        # ---- Solve the OCP ----
        solver.solve()

        # Computation time
        solve_time = solver.get_stats('time_tot')
        time_hist.append(solve_time)


        # Extract the state and the control computed by the solver
        u_opt_22 = solver.get(0, 'u')
        X_next_22 = solver.get(1, 'x')

        X_sol_t_prec = []
        U_sol_t_prec = []
        for k in range(sim_cfg.N_horizon):
            X_sol_t_prec.append(solver.get(k, 'x'))
            U_sol_t_prec.append(solver.get(k, 'u'))
        X_sol_t_prec.append(solver.get(sim_cfg.N_horizon, 'x'))


        u_apply = u_opt_22[0:11]
        X_next_sim = X_next_22[11:22]


        #### Logging ####
        # Save dt_var
        dt_chosen = u_apply[7]
        t_global += dt_chosen
        
        # Save footprints
        if step % gait_planner.steps_per_phase == 0:
            foot_positions_world[current_gait[0]].append(u_apply[0:2])
            foot_positions_world[current_gait[1]].append(u_apply[2:4])


        
        
        # Future (X_next) position scheduled by the controller
        hips_next = gait_planner.compute_hip_positions(X_next_sim[0:3], current_gait)
        # Compute the Euclidian distance and save it
        dist_p0_curr.append(np.linalg.norm(X_sim[6:8] - hips_next[0:2]))
        dist_p1_curr.append(np.linalg.norm(X_sim[8:10] - hips_next[2:4]))
        dist_p0_next.append(np.linalg.norm(u_apply[0:2] - hips_next[0:2]))
        dist_p1_next.append(np.linalg.norm(u_apply[2:4] - hips_next[2:4]))

        x_hist_step = np.copy(X_next_sim)
        theta = X_next_sim[2]
        vx_glob = X_next_sim[3]
        vy_glob = X_next_sim[4]

        x_hist_step[3] = vx_glob * np.cos(theta) + vy_glob * np.sin(theta)
        x_hist_step[4] = -vx_glob * np.sin(theta) + vy_glob * np.cos(theta)

        history_X.append(x_hist_step)
        history_U.append(u_apply)

        # Print some data for debugging and monitoring
        if step % 1 == 0:
            print(f"Step {step:02} | Pos: [{X_sim[0]:.2f}, {X_sim[1]:.2f}] | Theta: {np.rad2deg(X_sim[2]):.1f} deg | dt: {dt_chosen*1000:.1f} ms | Comp. time: {solve_time*1000:.1f} ms")


        # Control if the robot reached the target, if so stop the simulation
        dist_to_target = np.linalg.norm(X_next_sim[0:2] - sim_cfg.c_target)
        if dist_to_target < 0.1: # 5 cm threshold
            reached_target_flag += 1
            if reached_target_flag == 5:
                print(f"\n[INFO] Target {sim_cfg.c_target} reached successfully at step {step}!")
                X_sim = X_next_sim
                break

        # For the future step - in the real implementation here we will have the sensors data
        X_sim = X_next_sim

    print(f"Average computational time: {np.mean(time_hist)*1000:.2f} ms")

    # ---- Plotting ----
    plot_simulation_results(
        history_X, history_U, history_obs, foot_positions_world, 
        (dist_p0_curr, dist_p1_curr, dist_p0_next, dist_p1_next),
        (sim_cfg.c_target, sim_cfg.theta_target), (obs_cfg.pos_init, obs_cfg.r_obs, obs_cfg.y_dot_max)
    )


if __name__ == "__main__":
    main()