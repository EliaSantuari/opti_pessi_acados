import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from lipm_model import create_lipm_ocp
from gait_planner import GaitPlanner
from visualizer import plot_simulation_results
from config import RobotConfig, MPCWeights, SimulationConfig, ObstacleConfig, Limits

#### Definition of the obstacle behaviour ####
def get_obs_position(t, obs_cfg, robot_pos=None, t_current=0.0):
    if obs_cfg.obs_type == "static":
        return obs_cfg.pos_init.copy()
        
    elif obs_cfg.obs_type == "dynamic":
        p_top = obs_cfg.top_pos_dyn
        p_bot = obs_cfg.bot_pos_dyn
        # Vettore direzione e distanza tra i due punti
        vec = p_bot - p_top
        dist_tot = np.linalg.norm(vec)
        if dist_tot > 1e-5:
            dir_u = vec / dist_tot # Vettore unitario da top a bot
                # Distanza totale che l'ostacolo avrebbe percorso in linea retta
            s_percorso = obs_cfg.speed * t # Calcolo della posizione nel ciclo di andata e ritorno (lunghezza totale = 2 * dist_tot)
            s_ciclo = s_percorso % (2 * dist_tot)
            if s_ciclo <= dist_tot: # Fase di andata (da top a bot)
                return p_top + dir_u * s_ciclo
            else: # Fase di ritorno (da bot a top)
                return p_bot - dir_u * (s_ciclo - dist_tot)
        return p_top.copy()
        
    elif obs_cfg.obs_type == "circular":
        # Estraiamo centro e posizione iniziale
        center_x, center_y = obs_cfg.center[0], obs_cfg.center[1]
        start_x, start_y = obs_cfg.pos_init[0], obs_cfg.pos_init[1]
        # 1. Calcoliamo il raggio effettivo come distanza tra centro e partenza
        dx, dy = start_x - center_x, start_y - center_y
        radius = np.sqrt(dx**2 + dy**2)
        # Evitiamo divisioni per zero se partenza e centro coincidono
        if radius < 1e-5:
            return np.array([start_x, start_y])
        # 2. Calcoliamo l'angolo di partenza (fase iniziale) per t=0
        theta_0 = np.arctan2(dy, dx)
        # 3. Velocità angolare (omega = v / r)
                        # Il segno definisce il verso (es. positivo = antiorario, negativo = orario)
        omega = obs_cfg.speed / radius 
        # 4. Equazioni parametriche del cerchio con fase iniziale
        return np.array([
            center_x + radius * np.cos(omega * t + theta_0),
            center_y + radius * np.sin(omega * t + theta_0)
        ])
        
    elif obs_cfg.obs_type == "adversarial":
        # Per l'adversarial è più complesso predire il futuro perché dipende dal robot,
        # per semplicità manteniamo l'ultima posizione nota
        if robot_pos is None:
            return obs_cfg.pos_init.copy()
        dir_to_robot = robot_pos - obs_cfg.pos_init
        dist = np.linalg.norm(dir_to_robot)
        if dist > 1e-3:
            dir_to_robot = dir_to_robot / dist
        return obs_cfg.pos_init + dir_to_robot * obs_cfg.speed * (t-t_current)












def main():
    # ---- Settings and initialization -----
    sim_cfg = SimulationConfig()
    robot_cfg = RobotConfig()
    obs_cfg = ObstacleConfig()
    weights = MPCWeights()
    limits = Limits()


    # Push simulation
    PUSH = 0
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

    
    previous_phase = -1
    t_global = 0.0

    # Target reached flag
    reached_target_flag = 0


    print("--- Starting Simulation MPC ---")

    for step in range(sim_cfg.sim_steps):
        x_hist_step = np.copy(X_sim)
        theta = X_sim[2]
        vx_glob = X_sim[3]
        vy_glob = X_sim[4]

        x_hist_step[3] = vx_glob * np.cos(theta) + vy_glob * np.sin(theta)
        x_hist_step[4] = -vx_glob * np.sin(theta) + vy_glob * np.cos(theta)

        history_X.append(x_hist_step)


        obs_pos_current = get_obs_position(t_global, obs_cfg, X_sim[0:2], t_current=t_global)
        history_obs.append(obs_pos_current.copy())

        # if step == push_step and PUSH:
        #     # Parametri fisici della spinta
        #     F_push_x = -100.0  # Spinta di 100 Newton (circa 10 kg) all'indietro
        #     F_push_y = -50.0     # Nessuna spinta laterale
        #     dt_push = 0.1      # Durata stimata dell'impatto (100 millisecondi)
        #     # Calcolo della variazione di velocità basata sulla massa
        #     delta_vx = (F_push_x * dt_push) / robot_cfg.m
        #     delta_vy = (F_push_y * dt_push) / robot_cfg.m
        #     # Applica l'impulso
        #     X_sim[3] += delta_vx
        #     X_sim[4] += delta_vy
        #     print(f"Applied force. Delta Vx: {delta_vx:.3f} m/s, Delta Vy: {delta_vy:.3f} m/s")


        # ---- Gait and hips managing ----
        # Compute the current phase of the gait from step
        current_phase = (step // gait_planner.steps_per_phase) % 2
        # Get the gait horizon and then the current gait 
        current_gait = gait_planner.get_gait_horizon(step, sim_cfg.N_horizon)[0]

        # When the phase changes I need to set the hips on the selected ones
        if current_phase != previous_phase:
            # The first step will not enter here
            if previous_phase != -1: 
                new_hips = gait_planner.compute_hip_positions(X_sim[0:3], current_gait)
                X_sim[6:8] = new_hips[0:2]
                X_sim[8:10] = new_hips[2:4]
            previous_phase = current_phase

        X_sim[10] = 0.0
        X_aug = np.concatenate([X_sim, X_sim])
        # Set initial state
        solver.set(0, "lbx", X_aug)
        solver.set(0, "ubx", X_aug)
        solver.set(0, "x", X_aug)


        # ---- Update references and parameters over horizon
        yref = np.zeros(34)
        yref[0:2] = sim_cfg.c_target
        yref[17:19] = sim_cfg.c_target
        
        yref_e = np.zeros(12)
        yref_e[0:2] = sim_cfg.c_target
        yref_e[6:8] = sim_cfg.c_target

        # Estimate of the dt
        dt_guess = (limits.dt_min + limits.dt_max) / (2*sim_cfg.steps_per_phase)
    

        # Pass the parameters of the hips in the current phase
        for k in range(sim_cfg.N_horizon):
            offset0 = gait_planner.hip_offsets[current_gait[0]]
            offset1 = gait_planner.hip_offsets[current_gait[1]]

            # Prevision -> compute where the obstacle will be in the next step t_k
            t_k = t_global + k * dt_guess
            ################ CHEAT!!!! ################
            obs_pos_k = get_obs_position(t_k, obs_cfg, X_sim[0:2], t_current=t_global)

            # Virtual paraurti
            virtual_r_obs = obs_cfg.r_obs * 1.8

            # Parameters: [hip0_x, hip0_y, hip1_x, hip1_y, obs_x, obs_y, obs_cfg.r_obs]
            p_val = np.hstack([offset0, offset1, obs_pos_k, virtual_r_obs, obs_cfg.y_dot_max])

            solver.set(k, 'p', p_val)
            solver.set(k, 'yref', yref) # Update the intermediate target

            # Initialization of the line (WARM START) (line pointing towards the obstacle)
            dir_to_obs = obs_pos_k - X_sim[0:2]
            dist_to_obs = np.linalg.norm(dir_to_obs) + 1e-5
            # Normal points towards the obstacle
            a_guess = dir_to_obs / dist_to_obs
            # b_guess place a line exactly in between the robot and the obstacle
            b_guess = -np.dot(a_guess, (X_sim[0:2] + obs_pos_k) / 2.0)


            # u_guess uses the ACTUAL position of the feet (X[6:10]) as guess fot eh future
            u_guess_11 = np.array([
                X_sim[6], X_sim[7], X_sim[8], X_sim[9],  # p0_next, p1_next 
                0.5,                     # alpha
                0.0, 0.0,                # f_diff
                limits.dt_max,                  # dt_var
                a_guess[0], a_guess[1],  # ax, ay
                b_guess                  # b 
            ])
            solver.set(k, 'u', np.concatenate([u_guess_11, u_guess_11]))

        # Set the parameter of the final step because is missing
        solver.set(sim_cfg.N_horizon, 'p', p_val)
        solver.set(sim_cfg.N_horizon, 'yref', yref_e)
            
        # ---- Solve the OCP ----
        solver.solve()

        # Computation time
        solve_time = solver.get_stats('time_tot')
        time_hist.append(solve_time)


        # Extract the state and the control computed by the solver
        u_opt_22 = solver.get(0, 'u')

        # Extract only optimistic part
        u_apply = u_opt_22[0:11]
        # X_next_sim = X_next_22[0:11]
        X_next_sim = solver.get(1, "x")[0:11]

        # ---- Logging -----
        # Save dt_var
        dt_chosen = u_apply[7]
        t_global += dt_chosen


        # Update for adversarial obstacle
        if obs_cfg.obs_type == "adversarial":
            dir_to_robot = X_sim[0:2] - obs_cfg.pos_init
            dist = np.linalg.norm(dir_to_robot)
            if dist > 1e-3:
                dir_to_robot = dir_to_robot / dist
            obs_cfg.pos_init += dir_to_robot * obs_cfg.speed * dt_chosen

        # Save footprints
        if step % gait_planner.steps_per_phase == 0:
            foot_positions_world[current_gait[0]].append(u_apply[0:2])
            foot_positions_world[current_gait[1]].append(u_apply[2:4])

        # Compute kinematic distances between foot and hip
        # p0_curr, p1_curr = X_sim[6:8], X_sim[8:10]
        # p0_next, p1_next = u_apply[0:2], u_apply[2:4]
        
        
        # Future (X_next) position scheduled by the controller
        hips_next = gait_planner.compute_hip_positions(X_next_sim[0:3], current_gait)
        # Compute the Euclidian distance and save it
        dist_p0_curr.append(np.linalg.norm(X_sim[6:8] - hips_next[0:2]))
        dist_p1_curr.append(np.linalg.norm(X_sim[8:10] - hips_next[2:4]))
        dist_p0_next.append(np.linalg.norm(u_apply[0:2] - hips_next[0:2]))
        dist_p1_next.append(np.linalg.norm(u_apply[2:4] - hips_next[2:4]))

        
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