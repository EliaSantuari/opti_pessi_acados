import numpy as np
import matplotlib.pyplot as plt
from config import RobotConfig, SimulationConfig, Limits

robot_cfg = RobotConfig()
sim_cfg = SimulationConfig()
limits = Limits()

def draw_robot(ax, c_x, c_y, theta, alpha=0.3):
    """Draws the shape of the robot and its orientation"""
    corners = np.array([
        [robot_cfg.off_x, robot_cfg.off_y], [-robot_cfg.off_x, robot_cfg.off_y], [-robot_cfg.off_x, -robot_cfg.off_y], [robot_cfg.off_x, -robot_cfg.off_y], [robot_cfg.off_x, robot_cfg.off_y]
    ])
    R = np.array([
        [np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)]
    ])
    corners_world = (R @ corners.T).T + np.array([c_x, c_y])
    ax.plot(corners_world[:, 0], corners_world[:, 1], color='gray', alpha=alpha, linewidth=1.5)
    
    front_mid = (corners_world[0] + corners_world[3]) / 2 
    ax.plot([c_x, front_mid[0]], [c_y, front_mid[1]], color='red', alpha=alpha, linewidth=2)


def plot_simulation_results(history_X, history_U, history_obs, foot_positions_world, distances, target, obs_params):
    """Raccoglie e genera tutti i grafici della simulazione con i relativi limiti."""
    traj = np.array(history_X)
    traj_u = np.array(history_U)
    traj_obs = np.array(history_obs)
    
    c_target, theta_target = target
    obs_pos, obs_r, y_dot_max = obs_params
    dist_p0_curr, dist_p1_curr, dist_p0_next, dist_p1_next = distances

    # --- 1. Trajectory & Robot Shape ---
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(traj[:, 0], traj[:, 1], 'k--', label='CoM Trajectory', alpha=0.5)
    ax1.plot(c_target[0], c_target[1], 'rx', markersize=15, markeredgewidth=3, label='Target')

    style_map = {'FL': ('blue', '^'), 'FR': ('cyan', 'v'), 'RL': ('green', '<'), 'RR': ('orange', '>')}
    
    # Disegna le impronte e aggiunge la NUMERAZIONE
    for leg, pos_list in foot_positions_world.items():
        if len(pos_list) > 0:
            pos_arr = np.array(pos_list)
            color, marker = style_map[leg]
            ax1.scatter(pos_arr[:, 0], pos_arr[:, 1], c=color, marker=marker, 
                       label=f'Piede {leg}', s=90, edgecolors='black', alpha=0.8, zorder=3)
            
            # Aggiunge il numero del passo accanto all'impronta
            for step_idx, pos in enumerate(pos_arr):
                ax1.annotate(str(step_idx + 1), (pos[0], pos[1]), textcoords="offset points", 
                             xytext=(6, 6), ha='center', fontsize=9, color=color, weight='bold')

    # Disegna la sagoma del robot e le GAMBE
    # Aumentiamo l'intervallo per non creare un "verme" nero incomprensibile
    # draw_interval = max(1, len(traj) // 10) # Disegna circa 10-15 sagome in tutto
    draw_interval = 4
    
    for i in range(0, len(traj), draw_interval):
        alpha_val = 0.8 if (i == 0 or i >= len(traj) - draw_interval) else 0.3
        c_x, c_y, theta = traj[i, 0], traj[i, 1], traj[i, 2]
        
        # 1. Disegna il corpo
        draw_robot(ax1, c_x, c_y, theta, alpha=alpha_val)
        
        # 2. Estrai la posizione dei 2 piedi attualmente a terra (da history_X)
        p0_x, p0_y = traj[i, 6], traj[i, 7]
        p1_x, p1_y = traj[i, 8], traj[i, 9]
        
        # 3. Disegna le linee (gambe) che collegano il CoM ai piedi attivi in quel momento
        ax1.plot([c_x, p0_x], [c_y, p0_y], color='black', linestyle=':', linewidth=1.5, alpha=alpha_val)
        ax1.plot([c_x, p1_x], [c_y, p1_y], color='black', linestyle=':', linewidth=1.5, alpha=alpha_val)
    
    # Disegna sempre l'ultima sagoma e le ultime gambe a fine simulazione
    draw_robot(ax1, traj[-1, 0], traj[-1, 1], traj[-1, 2], alpha=0.9)
    ax1.plot([traj[-1, 0], traj[-1, 6]], [traj[-1, 1], traj[-1, 7]], color='black', linestyle=':', linewidth=1.5, alpha=0.9)
    ax1.plot([traj[-1, 0], traj[-1, 8]], [traj[-1, 1], traj[-1, 9]], color='black', linestyle=':', linewidth=1.5, alpha=0.9)
    
    ax1.plot(obs_pos[0], obs_pos[1], 'ro', markersize=6, label='Centro Ostacolo')

    # --- 2. Velocities (con linee dei limiti max/min) ---
    fig_v, axs_v = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Longitudinal Velocity
    axs_v[0].plot(traj[:, 3], 'b-', linewidth=2, label=r'$\dot{c}_x$ (Velocity X)')
    axs_v[0].axhline(y=limits.v_max_x, color='r', linestyle='--', alpha=0.8, label=f'Max ({limits.v_max_x})')
    axs_v[0].axhline(y=-limits.v_max_x, color='r', linestyle='--', alpha=0.8, label=f'Min ({limits.v_max_x})')
    axs_v[0].set_title('Longitudinal Velocity of the CoM')
    axs_v[0].set_ylabel('Velocity [m/s]')
    axs_v[0].legend(loc='upper right')
    axs_v[0].grid(True)
    
    # Lateral Velocity
    axs_v[1].plot(traj[:, 4], 'g-', linewidth=2, label=r'$\dot{c}_y$ (Velocity Y)')
    axs_v[1].axhline(y=limits.v_max_y, color='r', linestyle='--', alpha=0.8, label=f'Max ({limits.v_max_y})')
    axs_v[1].axhline(y=-limits.v_max_y, color='r', linestyle='--', alpha=0.8, label=f'Min ({-limits.v_max_y})')
    axs_v[1].set_title('Lateral Velocity of the CoM')
    axs_v[1].set_ylabel('Velocity [m/s]')
    axs_v[1].legend(loc='upper right')
    axs_v[1].grid(True)
    
    # Angular Velocity
    axs_v[2].plot(traj[:, 5], 'm-', linewidth=2, label=r'$\dot{\theta}$ (Yaw Rate)')
    axs_v[2].axhline(y=limits.theta_dot, color='r', linestyle='--', alpha=0.8, label=f'Max ({limits.theta_dot})')
    axs_v[2].axhline(y=-limits.theta_dot, color='r', linestyle='--', alpha=0.8, label=f'Min ({-limits.theta_dot})')
    axs_v[2].set_title('Angular Velocity (Yaw Rate)')
    axs_v[2].set_xlabel('Simulation step')
    axs_v[2].set_ylabel('Velocity [rad/s]')
    axs_v[2].legend(loc='upper right')
    axs_v[2].grid(True)

    # --- 3. Controls (con linee di bilanciamento e limiti dt) ---
    fig_u, axs_u = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    
    # Alpha
    axs_u[0].plot(traj_u[:, 4], 'm-', linewidth=2, label=r'$\alpha$ (CoP weight)')
    axs_u[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Balance 50-50')
    axs_u[0].set_title('Control: balancing of the weight (Alpha)')
    axs_u[0].set_ylabel(r'$\alpha \in [0.1, 0.9]$')
    axs_u[0].legend(loc='upper right')
    axs_u[0].grid(True)
    
    # f_diff_x
    axs_u[1].plot(traj_u[:, 5], 'b-', linewidth=2, label=r'$\Delta f_x$')
    axs_u[1].axhline(y=0.0, color='gray', linestyle='--', alpha=0.7)
    axs_u[1].set_title('Control: longitudinal differential force')
    axs_u[1].set_ylabel('Force [N]')
    axs_u[1].legend(loc='upper right')
    axs_u[1].grid(True)
    
    # f_diff_y
    axs_u[2].plot(traj_u[:, 6], 'g-', linewidth=2, label=r'$\Delta f_y$')
    axs_u[2].axhline(y=0.0, color='gray', linestyle='--', alpha=0.7)
    axs_u[2].set_title('Control: tangential differential force')
    axs_u[2].set_ylabel('Force [N]')
    axs_u[2].legend(loc='upper right')
    axs_u[2].grid(True)
    
    # dt_var
    dt_ms = traj_u[:, 7] * 1000 
    axs_u[3].plot(dt_ms, 'r-', linewidth=2, label='optimal dt')
    dt_min_ms = (limits.dt_min/sim_cfg.steps_per_phase)*1000
    dt_max_ms = (limits.dt_max/sim_cfg.steps_per_phase)*1000
    dt_ref = (dt_max_ms+dt_min_ms)/2
    axs_u[3].axhline(y=dt_min_ms, color='k', linestyle=':', linewidth=2, label=f'Min ({dt_min_ms})')
    axs_u[3].axhline(y=dt_max_ms, color='k', linestyle=':', linewidth=2, label=f'Max ({dt_max_ms})')
    axs_u[3].axhline(y=dt_ref, color='orange', linestyle='--', alpha=0.7, label=f'Nominal ({dt_ref})')
    axs_u[3].set_title('Control: variable time step (optimal time step)')
    axs_u[3].set_ylabel('Time [ms]')
    axs_u[3].set_xlabel('Simulation time')
    axs_u[3].legend(loc='upper right')
    axs_u[3].grid(True)

    # --- 4. DISTANCE BETWEEN FOOT AND HIPS (con limite di estensione max_ext) ---
    max_ext = np.sqrt(robot_cfg.max_ext_sq)
    
    fig_d, axs_d = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Next foot vs next hip
    axs_d[0].plot(dist_p0_next, 'b-', label='Foot 0 (Next)')
    axs_d[0].plot(dist_p1_next, 'g-', label='Foot 1 (Next)')
    axs_d[0].axhline(y=max_ext, color='r', linestyle='--', label=f'Upper limit ({max_ext:.2f}m)')
    axs_d[0].set_title('Future extension (distance between next foot and next hip)')
    axs_d[0].set_ylabel('Distance [m]')
    axs_d[0].legend(loc='upper right')
    axs_d[0].grid(True)
    
    # Current foot vs next hip
    axs_d[1].plot(dist_p0_curr, 'b-', label='Foot 0 (Curr)')
    axs_d[1].plot(dist_p1_curr, 'g-', label='Foot 1 (Curr)')
    axs_d[1].axhline(y=max_ext, color='r', linestyle='--', label=f'Upper limit ({max_ext:.2f}m)')
    axs_d[1].set_title('Current extension (distance between current foot and next hip)')
    axs_d[1].set_xlabel('Simulation step')
    axs_d[1].set_ylabel('Distance [m]')
    axs_d[1].legend(loc='upper right')
    axs_d[1].grid(True)



    # --- 5. PLOT PULITO: COM ROBOT VS OSTACOLO IN MOVIMENTO ---
    fig_clean, ax_clean = plt.subplots(figsize=(10, 8))
    
    # 1. Traiettoria del baricentro (CoM) del robot
    ax_clean.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2.5, label='Traiettoria CoM Robot')
    ax_clean.plot(traj[0, 0], traj[0, 1], 'bo', markersize=8, label='Partenza Robot')
    ax_clean.plot(traj[-1, 0], traj[-1, 1], 'b*', markersize=12, label='Posizione Finale Robot')
    
    # 2. Traiettoria dell'ostacolo in movimento
    ax_clean.plot(traj_obs[:, 0], traj_obs[:, 1], 'r--', linewidth=2, label='Traiettoria Ostacolo')
    ax_clean.plot(traj_obs[0, 0], traj_obs[0, 1], 'ro', markersize=8, label='Partenza Ostacolo')
    ax_clean.plot(traj_obs[-1, 0], traj_obs[-1, 1], 'r*', markersize=12, label='Posizione Finale Ostacolo')
    
    # 3. Target globale
    ax_clean.plot(c_target[0], c_target[1], 'gx', markersize=15, markeredgewidth=3, label='Target')

    # 4. Disegno dei "fantasmi" dell'ostacolo per mostrarne l'evoluzione temporale
    # Scegliamo di disegnare l'ostacolo in circa 8 posizioni distribuite lungo la simulazione
    step_interval = max(1, len(traj_obs) // 8) 
    
    for i in range(0, len(traj_obs), step_interval):
        # L'opacità aumenta per far capire la direzione del movimento (più scuro = più recente)
        alpha_val = 0.2 + 0.6 * (i / len(traj_obs)) 
        circle = plt.Circle(
            traj_obs[i], obs_r, 
            color='red', fill=False, linestyle='-', linewidth=1.5, alpha=alpha_val
        )
        ax_clean.add_patch(circle)
        
    # Disegna l'ostacolo pieno nella sua posizione finale
    circle_final = plt.Circle(traj_obs[-1], obs_r, color='red', fill=True, alpha=0.3)
    ax_clean.add_patch(circle_final)

    ax_clean.set_title('Traiettoria Pura: CoM Robot vs Ostacolo Dinamico')
    ax_clean.set_xlabel('X [m]')
    ax_clean.set_ylabel('Y [m]')
    ax_clean.legend(loc='best')
    ax_clean.grid(True)
    ax_clean.axis('equal')


# --- 1. Trajectory & Robot Shape (Tutti i piedi campionati collegati) ---
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    # 1. Traiettoria del CoM
    ax1.plot(traj[:, 0], traj[:, 1], color='#40c4c4', linewidth=2.5, label='Opti-Pessi MPC', zorder=2)
    ax1.plot(traj[:, 0], traj[:, 1], 'o', color='#40c4c4', markersize=4, zorder=2)

    # 2. Start e Goal
    ax1.plot(traj[0, 0], traj[0, 1], 'bs', markersize=7, label='Start', zorder=3)
    ax1.plot(c_target[0], c_target[1], 'v', color='blue', markersize=7, label='Goal', zorder=3)

    # 3. Ostacolo
    ax1.plot(obs_pos[0], obs_pos[1], 'rs', markersize=7, label='Obstacle', zorder=3)

    # 4. Piedi p0 e p1
    p0_traj = traj[:, 6:8]
    p1_traj = traj[:, 8:10]
    
    ax1.scatter(p0_traj[::draw_interval, 0], p0_traj[::draw_interval, 1], c='green', marker='^', s=40, label='p0', alpha=0.9, zorder=3)
    ax1.scatter(p1_traj[::draw_interval, 0], p1_traj[::draw_interval, 1], c='purple', marker='v', s=40, label='p1', alpha=0.9, zorder=3)

    # 5. Collegamento CoM -> p0 e CoM -> p1 SIMULTANEO per ogni punto campionato
    for i in range(0, len(traj), draw_interval):
        c_x, c_y = traj[i, 0], traj[i, 1]
        
        p0_x, p0_y = p0_traj[i, 0], p0_traj[i, 1]
        p1_x, p1_y = p1_traj[i, 0], p1_traj[i, 1]
        
        # Disegna la linea per p0
        ax1.plot([c_x, p0_x], [c_y, p0_y], color='gray', linestyle='-', linewidth=0.5, alpha=0.6)
        # Disegna la linea per p1
        ax1.plot([c_x, p1_x], [c_y, p1_y], color='gray', linestyle='-', linewidth=0.5, alpha=0.6)

    sim_T = len(traj) * 0.05
    ax1.set_title(f'Opti-Pessi MPC, N = {sim_cfg.N_horizon}, sim_T = {sim_T:.2f} s', fontsize=14)
    
    ax1.set_xlabel('X [m]', fontsize=12)
    ax1.set_ylabel('Y [m]', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, linestyle='-', alpha=0.6)
    ax1.axis('equal')

    plt.tight_layout()
    plt.show()