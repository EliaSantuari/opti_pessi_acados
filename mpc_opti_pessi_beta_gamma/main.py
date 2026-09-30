import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from lipm_model import create_lipm_ocp
from gait_planner import GaitPlanner


 
# Function used to draw tha shape of the robot and its orientation
def draw_robot(ax, c_x, c_y, theta, alpha=0.3):
    """Draws tha shape of the robot and its orientation"""
    # Local coordinates of the shoulders: FL, RL, RR, FR, FL (to close the rectangle)
    corners = np.array([
        [0.2, 0.2],
        [-0.2, 0.2],
        [-0.2, -0.2],
        [0.2, -0.2],
        [0.2, 0.2]
    ])
    
    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Rotate and translate the points in real world
    corners_world = (R @ corners.T).T + np.array([c_x, c_y])
    
    # Draw the perimeter of the body in the plot
    ax.plot(corners_world[:, 0], corners_world[:, 1], color='gray', alpha=alpha, linewidth=1.5)
    
    # Draw a red line fromt the center towards the orientation
    front_mid = (corners_world[0] + corners_world[3]) / 2 
    ax.plot([c_x, front_mid[0]], [c_y, front_mid[1]], color='red', alpha=alpha, linewidth=2)




def main():
    N_horizon = 8
    sim_steps = 150
    c_target = np.array([0, 0])
    theta_target = np.deg2rad(-180)

    # steps_per_phase indicates the numer of time steps after which the robot changes feet
    gait_planner = GaitPlanner(steps_per_phase=4)
    X_sim = np.zeros(11)

    # Initialization of the gait
    init_gait = gait_planner.get_gait_horizon(0, N_horizon)[0]
    initial_hips = gait_planner.compute_hip_positions(X_sim[0:3], init_gait)
    # Initial position of the selected feet exactly underneath the hips
    X_sim[6:8] = initial_hips[0:2]
    X_sim[8:10] = initial_hips[2:4]

    X_aug = np.concatenate([X_sim, X_sim])

    # Create the OCP
    solver = create_lipm_ocp(N=N_horizon, c_target=c_target, theta_target=theta_target, x_init=X_aug)

    # For storing the history
    history_X = []
    history_U = []
    foot_positions_world = {'FL': [], 'FR': [], 'RL': [], 'RR': []}
    previous_phase = -1
    history_obs = []

    # Computational time history
    time_hist = []
    
    # Lists for tracking distances between feet and hips
    dist_p0_curr_list = []
    dist_p1_curr_list = []
    dist_p0_next_list = []
    dist_p1_next_list = []


    # Definition of the dynamic obstacle
    obs_pos_init = np.array([1, 1])
    obs_r = 0.2
    obs_speed = 0.3
    y_dot_max = 0.3

    # Target reached flag
    reached_target_flag = 0

    t_global = 0.0

    print("--- Starting Simulation MPC ---")

    for step in range(sim_steps):
        # obs_pos = obs_pos_init + np.array([-obs_speed * t_global * 0.0 , -obs_speed * t_global])
        obs_pos = obs_pos_init
        # robot_pos = X_sim[0:2]
        # dir_to_robot = robot_pos - obs_pos
        # dist_to_robot = np.linalg.norm(dir_to_robot)

        # if dist_to_robot > 1e-3:
        #     dir_to_robot = dir_to_robot / dist_to_robot

        # dt_step = dt_chosen if 'dt_chosen' in locals() else (0.275 / 4.0)

        # obs_pos = obs_pos + dir_to_robot * obs_speed * dt_step
        history_obs.append(obs_pos.copy())

        X_sim[10] = 0.0
        X_aug = np.concatenate([X_sim, X_sim])
        # Set initial state
        solver.set(0, "lbx", X_aug)
        solver.set(0, "ubx", X_aug)

        # Compute the current phase of the gait from step
        current_phase = (step // gait_planner.steps_per_phase) % 2
        # Get the gait horizon and then the current gait 
        gait_horizon = gait_planner.get_gait_horizon(step, N_horizon)
        current_gait = gait_horizon[0]

        # When the phase changes I need to set the hips on the selected ones
        if current_phase != previous_phase:
            # The first step will not enter here
            if previous_phase != -1: 
                new_hips = gait_planner.compute_hip_positions(X_sim[0:3], current_gait)
                X_sim[6:8] = new_hips[0:2]
                X_sim[8:10] = new_hips[2:4]
            previous_phase = current_phase


        yref = np.zeros(40)
        yref[0:2] = c_target
        # yref[2] = theta_target
        yref[20:22] = c_target
        # yref[22] = theta_target
        
        yref_e = np.zeros(14)
        yref_e[0:2] = c_target
        yref_e[2] = theta_target
        yref_e[7:9] = c_target
        yref_e[9] = theta_target


        # Pass the parameters of the hips in the current phase
        for k in range(N_horizon):
            offset0 = gait_planner.hip_offsets[current_gait[0]]
            offset1 = gait_planner.hip_offsets[current_gait[1]]

            # Parameters: [hip0_x, hip0_y, hip1_x, hip1_y, obs_x, obs_y, obs_r]
            p_val = np.hstack([offset0, offset1, obs_pos, obs_r, y_dot_max])

            solver.set(k, 'p', p_val)
            solver.set(k, 'yref', yref) # Update the intermediate target

            # Initialization of the line (WARM START) (line pointing towards the obstacle)
            dir_to_obs = obs_pos - X_sim[0:2]
            dist_to_obs = np.linalg.norm(dir_to_obs) + 1e-5

            
            
            # Normal points towards the obstacle
            a_guess = dir_to_obs / dist_to_obs
            # b_guess place a line exactly in between the robot and the obstacle
            b_guess = -np.dot(a_guess, (X_sim[0:2] + obs_pos) / 2.0)
            
            # u_guess uses the ACTUAL position of the feet (X[6:10]) as guess fot eh future
            u_guess_11 = np.array([
                X_sim[6], X_sim[7], X_sim[8], X_sim[9],  # p0_next, p1_next 
                0.5,                     # alpha
                0.5, 0.5,                # f_diff
                0.0625,                  # dt_var
                a_guess[0], a_guess[1],  # ax, ay
                b_guess                  # b 
            ])

            u_guess = np.concatenate([u_guess_11, u_guess_11])
            solver.set(k, 'u', u_guess)

        # Set the parameter of the final step because is missing
        solver.set(N_horizon, 'p', p_val)
        solver.set(N_horizon, 'yref', yref_e)

        
        # Push
        if step == 30 and 0:
            X[3] -= 1.5
            X[4] -= 0.0
            
        # Solve the OCP
        # solver.set(0, 'lbx', X)
        # solver.set(0, 'ubx', X)
        solver.solve()



        # Computation time
        solve_time = solver.get_stats('time_tot')
        time_hist.append(solve_time)


        # Extract the state and the control computed by the solver
        u_opt_22 = solver.get(0, 'u')
        X_next_22 = solver.get(1, 'x')

        # Check for the trajectory OP vs PE difference
        if step == 30 and 0:
            print(f"\n--- Trajectory OP vs PE (Step {step}) ---")
            for k in range(N_horizon + 1):
                x_k = solver.get(k, 'x')
                cy_op = x_k[1]   # Y CoM optimistic
                cy_pe = x_k[12]  # Y CoM pessimistic
                diff = abs(cy_pe - cy_op)
                print(f"Nodo {k}: Y_op = {cy_op:.4f} | Y_pe = {cy_pe:.4f} | Delta = {diff:.4f} m")

        # Extract only optimistic part
        u_apply = u_opt_22[0:11]
        X_next_sim = X_next_22[0:11]

        # -----------------
        # Saving stuff for plotting

        # Save dt_var
        dt_chosen = u_apply[7]

        t_global += dt_chosen
        
        # Save footprints
        if step % gait_planner.steps_per_phase == 0:
            foot_positions_world[current_gait[0]].append(u_apply[0:2])
            foot_positions_world[current_gait[1]].append(u_apply[2:4])

        # Compute kinematic distances between foot and hip
        p0_curr, p1_curr = X_sim[6:8], X_sim[8:10]
        p0_next, p1_next = u_apply[0:2], u_apply[2:4]
        
        # Current (X) position of the hips 
        hips_curr = gait_planner.compute_hip_positions(X_sim[0:3], current_gait)
        hip0_curr, hip1_curr = hips_curr[0:2], hips_curr[2:4]
        
        # Future (X_next) position scheduled by the controller
        hips_next = gait_planner.compute_hip_positions(X_next_sim[0:3], current_gait)
        hip0_next, hip1_next = hips_next[0:2], hips_next[2:4]
        
        # Compute the Euclidian distance and save it
        dist_p0_curr_list.append(np.linalg.norm(p0_curr - hip0_next))
        dist_p1_curr_list.append(np.linalg.norm(p1_curr - hip1_next))
        dist_p0_next_list.append(np.linalg.norm(p0_next - hip0_next))
        dist_p1_next_list.append(np.linalg.norm(p1_next - hip1_next))

        
        # Print some data for debugging and monitoring
        if step % 1 == 0:
            print(f"Step {step:02} | Pos: [{X_sim[0]:.2f}, {X_sim[1]:.2f}] | Theta: {np.rad2deg(X_sim[2]):.1f} deg | dt: {dt_chosen*1000:.1f} ms | Comp. time: {solve_time*1000:.1f} ms")

        x_hist_step = np.copy(X_next_sim)
        theta = X_next_sim[2]
        vx_glob = X_next_sim[3]
        vy_glob = X_next_sim[4]

        x_hist_step[3] = vx_glob * np.cos(theta) + vy_glob * np.sin(theta)
        x_hist_step[4] = -vx_glob * np.sin(theta) + vy_glob * np.cos(theta)

        history_X.append(x_hist_step)
        history_U.append(u_apply)


        # Control if the robot reached the target, if so stop the simulation
        dist_to_target = np.linalg.norm(X_next_sim[0:2] - c_target)
        if dist_to_target < 0.05 and 0: # 2 cm threshold
            reached_target_flag += 1
            if reached_target_flag == 5:
                print(f"\n[INFO] Target {c_target} reached successfully at step {step}!")
                X_sim = X_next_sim
                break

        # For the future step - in the real implementation here we will have the sensors data
        X_sim = X_next_sim

    print(f"Average computational time: {np.mean(time_hist)*1000:.2f} ms")




    # --- PLOTTING TRAJECTORY WITH ROBOT SHAPE ---
    traj = np.array(history_X)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(traj[:, 0], traj[:, 1], 'k--', label='CoM Trajectory', alpha=0.5)
    ax.plot(c_target[0], c_target[1], 'rx', markersize=15, markeredgewidth=3, label='Target')

    # Draw the footprint
    style_map = {'FL': ('blue', '^'), 'FR': ('cyan', 'v'), 'RL': ('green', '<'), 'RR': ('orange', '>')}
    for leg, pos_list in foot_positions_world.items():
        if len(pos_list) > 0:
            pos_arr = np.array(pos_list)
            color, marker = style_map[leg]
            ax.scatter(pos_arr[:, 0], pos_arr[:, 1], c=color, marker=marker, 
                       label=f'Piede {leg}', s=90, edgecolors='black', alpha=0.8, zorder=3)

    # Draw the shape of the robot every N step
    draw_interval = 1
    for i in range(0, len(traj), draw_interval):
        # Highlight the first and last frame
        alpha_val = 0.8 if (i == 0 or i >= len(traj) - draw_interval) else 0.2
        draw_robot(ax, traj[i, 0], traj[i, 1], traj[i, 2], alpha=alpha_val)
    # Draw also for the last step
    draw_robot(ax, traj[-1, 0], traj[-1, 1], traj[-1, 2], alpha=0.9)
    # Draw the obstacle
    ax.plot(obs_pos[0], obs_pos[1], 'ro', markersize=6, label='Centro Ostacolo')
    
    # Scegliamo ogni quanti step disegnare il cerchio per pulizia visiva (es. ogni 10 step)
    step_interval = 10
    dt_avg = 0.275 / 4.0 # Tempo medio di uno step

    for i in range(0, len(traj), step_interval):
        # Il tempo cumulativo cresce con gli step della simulazione
        t_accumulated = i * dt_avg
        r_current = obs_r + y_dot_max * t_accumulated
        
        # Disegniamo un cerchio giallo sfumato per questo step
        circle = plt.Circle(
            obs_pos, r_current, 
            color='orange', alpha=0.1 + (i / len(traj)) * 0.3, # Più trasparente all'inizio, più opaco alla fine
            edgecolor='orange', linewidth=1, linestyle='--', fill=False
        )
        ax.add_patch(circle)
    # Draw a small dot in the middle of the obstacle
    ax.plot(obs_pos[0], obs_pos[1], 'rx')
    ax.set_title('CoM Trajectory, Footprints, Shape and Orientation of the robot ')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend(loc = 'best')
    ax.grid(True)
    ax.axis('equal')



    # --- PLOTTING VELOCITY TRAJECTORY ---
    fig_v, axs_v = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Longitudinal Velocity
    axs_v[0].plot(traj[:, 3], 'b-', linewidth=2, label=r'$\dot{c}_x$ (Velocity X)')
    axs_v[0].axhline(y=0.6, color='r', linestyle='--', alpha=0.8, label='Max (0.6 m/s)')
    axs_v[0].axhline(y=-0.6, color='r', linestyle='--', alpha=0.8, label='Min (-0.6 m/s)')
    axs_v[0].set_title('Longitudinal Velocity of the CoM')
    axs_v[0].set_ylabel('Velocity [m/s]')
    axs_v[0].legend(loc='upper right')
    axs_v[0].grid(True)
    
    # Lateral Velocity
    axs_v[1].plot(traj[:, 4], 'g-', linewidth=2, label=r'$\dot{c}_y$ (Velocity Y)')
    axs_v[1].axhline(y=0.3, color='r', linestyle='--', alpha=0.8, label='Max (0.3 m/s)')
    axs_v[1].axhline(y=-0.3, color='r', linestyle='--', alpha=0.8, label='Min (-0.3 m/s)')
    axs_v[1].set_title('Lateral Velocity of the CoM')
    axs_v[1].set_ylabel('Velocity [m/s]')
    axs_v[1].legend(loc='upper right')
    axs_v[1].grid(True)
    
    # Angular Velocity
    axs_v[2].plot(traj[:, 5], 'm-', linewidth=2, label=r'$\dot{\theta}$ (Yaw Rate)')
    axs_v[2].axhline(y=0.8, color='r', linestyle='--', alpha=0.8, label='Max (0.8 rad/s)')
    axs_v[2].axhline(y=-0.8, color='r', linestyle='--', alpha=0.8, label='Min (-0.8 rad/s)')
    axs_v[2].set_title('Angular Velocity (Yaw Rate)')
    axs_v[2].set_xlabel('Simulation step')
    axs_v[2].set_ylabel('Velocity [rad/s]')
    axs_v[2].legend(loc='upper right')
    axs_v[2].grid(True)



    # --- DISTANCE BETWEEN FOOT AND HIPS ---
    max_ext = np.sqrt(0.03) # Must be the same of lipm_model.py: max_ext_sq = 0.02
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Next foot vs next hip
    axs[0].plot(dist_p0_next_list, 'b-', label='Foot 0 (Next)')
    axs[0].plot(dist_p1_next_list, 'g-', label='Foot 1 (Next)')
    axs[0].axhline(y=max_ext, color='r', linestyle='--', label=f'Upper limit ({max_ext:.2f}m)')
    axs[0].set_title('Future extension (distance between next foot and next hip)')
    axs[0].set_ylabel('Distance [m]')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)
    
    # Current foot vs next hip
    axs[1].plot(dist_p0_curr_list, 'b-', label='Foot 0 (Curr)')
    axs[1].plot(dist_p1_curr_list, 'g-', label='Foot 1 (Curr)')
    axs[1].axhline(y=max_ext, color='r', linestyle='--', label=f'Upper limit ({max_ext:.2f}m)')
    axs[1].set_title('Current extension (distance between current foot and next hip)')
    axs[1].set_xlabel('Simulation step')
    axs[1].set_ylabel('Distance [m]')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)



    # --- PLOTTING OF THE CONTROL VARIABLES ---
    traj_u = np.array(history_U)
    
    fig_u, axs_u = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    
    # Alpha (balance of the CoP)
    axs_u[0].plot(traj_u[:, 4], 'm-', linewidth=2, label=r'$\alpha$ (CoP weight)')
    axs_u[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Balance 50-50')
    axs_u[0].set_title('Control: balancing of the weight (Alpha)')
    axs_u[0].set_ylabel(r'$\alpha \in [0.1, 0.9]$')
    axs_u[0].legend(loc='upper right')
    axs_u[0].grid(True)
    
    # beta (longitudinal force proportion)
    axs_u[1].plot(traj_u[:, 5], 'b-', linewidth=2, label=r'$\beta$')
    axs_u[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Bilanciamento 50%')
    axs_u[1].set_title('Control: Longitudinal force ratio (Beta)')
    axs_u[1].set_ylabel(r'$\beta$')
    axs_u[1].legend(loc='upper right')
    axs_u[1].grid(True)
    
    # gamma (tangential force proportion)
    axs_u[2].plot(traj_u[:, 6], 'g-', linewidth=2, label=r'$\gamma$')
    axs_u[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Bilanciamento 50%')
    axs_u[2].set_title('Control: Tangential force ratio (Gamma)')
    axs_u[2].set_ylabel(r'$\gamma$')
    axs_u[2].legend(loc='upper right')
    axs_u[2].grid(True)
    
    # dt_var
    dt_ms = traj_u[:, 7] * 1000 # ms
    axs_u[3].plot(dt_ms, 'r-', linewidth=2, label='optimal dt')
    # Add lines for limits of the time step: must be the same as lipm_model.py (dt_min = 0.2 / 4.0, dt_max = 0.35 / 4.0)
    dt_min_ms, dt_max_ms = 50.0, 87.5 
    axs_u[3].axhline(y=dt_min_ms, color='k', linestyle=':', linewidth=2, label='Min (50 ms)')
    axs_u[3].axhline(y=dt_max_ms, color='k', linestyle=':', linewidth=2, label='Max (87.5 ms)')
    axs_u[3].axhline(y=68.75, color='orange', linestyle='--', alpha=0.7, label='Nominal (62.5 ms)')

    axs_u[3].set_title('Control: variable time step (optimal time step)')
    axs_u[3].set_ylabel('Time [ms]')
    axs_u[3].set_xlabel('Simulation time')
    axs_u[3].legend(loc='upper right')
    axs_u[3].grid(True)



    # --- PLOT DI VERIFICA DELLA SICUREZZA (DISTANZA VS RAGGIO DINAMICO) ---
    safety_distances = []
    dynamic_radii_history = []
    
    dt_avg = 0.275 / 4.0
    
    for i, state in enumerate(history_X):
        # Posizione del CoM del robot a questo step
        com_pos = state[0:2]
        
        # Distanza minima tra il centro dell'ostacolo e il CoM 
        # (oppure potremmo considerare il margine del corpo del robot, sottraendo ~0.2m di raggio equivalente del corpo)
        dist_to_obs = np.linalg.norm(com_pos - obs_pos)
        safety_distances.append(dist_to_obs)
        
        # Raggio dinamico dell'ostacolo accumulato a questo step temporale
        t_cumul = i * dt_avg
        r_dyn = obs_r + y_dot_max * t_cumul
        dynamic_radii_history.append(r_dyn)

    fig_sec, ax_sec = plt.subplots(figsize=(12, 4))
    ax_sec.plot(safety_distances, 'b-', linewidth=2, label='Distanza CoM - Ostacolo')
    ax_sec.plot(dynamic_radii_history, 'orange', linestyle='--', linewidth=2, label='Raggio Dinamico di Sicurezza ($r_{obs} + \dot{y}_{max} t$)')
    
    ax_sec.set_title('Verifica di Sicurezza: Margine Anti-Collisione')
    ax_sec.set_xlabel('Step di simulazione')
    ax_sec.set_ylabel('Distanza [m]')
    ax_sec.legend(loc='upper right')
    ax_sec.grid(True)




    # --- PLOT: TRAIETTORIA ROBOT VS TRAIETTORIA OSTACOLO ---
    traj = np.array(history_X)
    traj_obs = np.array(history_obs) 
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Traiettoria CoM del Robot
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Traiettoria CoM Robot')
    
    # 2. Traiettoria del centro dell'ostacolo
    ax.plot(traj_obs[:, 0], traj_obs[:, 1], 'r--', linewidth=2, label='Percorso Ostacolo')
    
    # Target
    ax.plot(c_target[0], c_target[1], 'gx', markersize=15, markeredgewidth=3, label='Target')
    
    # Posizione iniziale (Start)
    ax.plot(traj[0, 0], traj[0, 1], 'bo', markersize=6, label='Partenza Robot')
    ax.plot(traj_obs[0, 0], traj_obs[0, 1], 'ro', markersize=6, label='Partenza Ostacolo')

    # Disegna il perimetro dell'ostacolo a intervalli regolari per mostrare l'ingombro nel tempo
    step_interval = max(1, len(traj) // 8) # Mostra circa 8 posizioni fantasma
    
    for i in range(0, len(traj_obs), step_interval):
        # L'opacità aumenta man mano che il tempo passa
        alpha_val = 0.2 + 0.6 * (i / len(traj_obs)) 
        circle = plt.Circle(
            traj_obs[i], obs_r, 
            color='red', fill=False, linestyle='-', linewidth=1.5, alpha=alpha_val
        )
        ax.add_patch(circle)
        
    # Posizione finale dell'ostacolo (pieno per distinguerlo)
    circle_final = plt.Circle(traj_obs[-1], obs_r, color='red', fill=True, alpha=0.3)
    ax.add_patch(circle_final)

    ax.set_title('Confronto Traiettorie: Robot vs Ostacolo Dinamico')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend(loc='best')
    ax.grid(True)
    ax.axis('equal')





    # --- PLOTTING TRAJECTORY WITH ROBOT SHAPE ---
    traj = np.array(history_X)
    traj_u = np.array(history_U) # Array dei controlli per estrarre la linea
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(traj[:, 0], traj[:, 1], 'k--', label='CoM Trajectory', alpha=0.5)
    ax.plot(c_target[0], c_target[1], 'rx', markersize=15, markeredgewidth=3, label='Target')

    # Draw the footprint
    style_map = {'FL': ('blue', '^'), 'FR': ('cyan', 'v'), 'RL': ('green', '<'), 'RR': ('orange', '>')}
    for leg, pos_list in foot_positions_world.items():
        if len(pos_list) > 0:
            pos_arr = np.array(pos_list)
            color, marker = style_map[leg]
            ax.scatter(pos_arr[:, 0], pos_arr[:, 1], c=color, marker=marker, 
                       label=f'Piede {leg}', s=90, edgecolors='black', alpha=0.8, zorder=3)

    # Draw the shape of the robot every N step
    draw_interval = 1
    for i in range(0, len(traj), draw_interval):
        # Highlight the first and last frame
        alpha_val = 0.8 if (i == 0 or i >= len(traj) - draw_interval) else 0.2
        draw_robot(ax, traj[i, 0], traj[i, 1], traj[i, 2], alpha=alpha_val)
    # Draw also for the last step
    draw_robot(ax, traj[-1, 0], traj[-1, 1], traj[-1, 2], alpha=0.9)
    # Draw the obstacle
    ax.plot(obs_pos[0], obs_pos[1], 'ro', markersize=6, label='Centro Ostacolo')
    
    # Scegliamo ogni quanti step disegnare ostacolo e linea per pulizia visiva (es. ogni 10 step)
    step_interval = 10
    dt_avg = 0.275 / 4.0 # Tempo medio di uno step

    for i in range(0, len(traj), step_interval):
        alpha_val = 0.1 + (i / len(traj)) * 0.3 # Più trasparente all'inizio, opaco alla fine
        
        # 1. Disegna l'ostacolo accumulato
        t_accumulated = i * dt_avg
        r_current = obs_r + y_dot_max * t_accumulated
        circle = plt.Circle(
            obs_pos, r_current, 
            color='orange', alpha=alpha_val,
            edgecolor='orange', linewidth=1, linestyle='--', fill=False
        )
        ax.add_patch(circle)
        
        # 2. Disegna la linea di separazione (a_x*x + a_y*y + b = 0)
        if i < len(traj_u):
            ax_line = traj_u[i, 8]
            ay_line = traj_u[i, 9]
            b_line = traj_u[i, 10]
            
            # Normalizziamo il vettore normale
            norm_a = np.hypot(ax_line, ay_line)
            if norm_a > 1e-5:
                # Trova il punto della retta più vicino all'origine per centrare il segmento
                x0 = -ax_line * b_line / (norm_a**2)
                y0 = -ay_line * b_line / (norm_a**2)
                
                # Vettore direzione della retta (perpendicolare alla normale)
                dir_x = -ay_line / norm_a
                dir_y = ax_line / norm_a
                
                # Lunghezza del segmento visibile (es. 1.5 metri per lato dal centro)
                L = 1.5 
                ax.plot([x0 - L * dir_x, x0 + L * dir_x], 
                        [y0 - L * dir_y, y0 + L * dir_y], 
                        color='purple', linestyle='-.', linewidth=1.5, alpha=alpha_val)

    # Elemento fittizio per aggiungere la linea di separazione alla legenda
    ax.plot([], [], color='purple', linestyle='-.', linewidth=1.5, label='Linea di Separazione')

    # Draw a small dot in the middle of the obstacle
    ax.plot(obs_pos[0], obs_pos[1], 'rx')
    ax.set_title('CoM Trajectory, Footprints, Shape, Orientation & Separation Line')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend(loc='best')
    ax.grid(True)
    ax.axis('equal')






    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()