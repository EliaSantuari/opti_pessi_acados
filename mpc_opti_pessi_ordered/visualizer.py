import numpy as np
import matplotlib.pyplot as plt
from config import RobotConfig, SimulationConfig, Limits, ObstacleConfig

robot_cfg = RobotConfig()
sim_cfg = SimulationConfig()
limits = Limits()
obs_cfg = ObstacleConfig()

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




from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle


def animate_robot_feet(history_X, history_obs, obs_r,
                       interval=80, tail_frames=40):
    """
    Anima CoM, impronte/traettorie dei piedi e ostacolo.

    history_X: stati con colonne
        0,1 = posizione CoM
        6,7 = piede p0
        8,9 = piede p1
    history_obs: posizione dell'ostacolo a ogni step, forma (N, 2)
    obs_r: raggio dell'ostacolo [m]
    interval: intervallo tra frame [ms]
    tail_frames: numero di step dopo cui le tracce scompaiono
    """
    traj = np.asarray(history_X)
    obs_traj = np.asarray(history_obs)

    if traj.ndim != 2 or traj.shape[1] < 10:
        raise ValueError("history_X deve avere almeno 10 colonne.")
    if obs_traj.ndim != 2 or obs_traj.shape[1] < 2:
        raise ValueError("history_obs deve avere forma (N, 2).")

    n = min(len(traj), len(obs_traj))
    if n == 0:
        raise ValueError("Le storie della simulazione sono vuote.")

    traj = traj[:n]
    obs_traj = obs_traj[:n]

    com = traj[:, 0:2]
    p0 = traj[:, 6:8]
    p1 = traj[:, 8:10]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Limiti del grafico: includono CoM, piedi e ostacolo
    all_x = np.concatenate((com[:, 0], p0[:, 0], p1[:, 0], obs_traj[:, 0]))
    all_y = np.concatenate((com[:, 1], p0[:, 1], p1[:, 1], obs_traj[:, 1]))
    margin = obs_r + 0.3
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Animazione CoM, impronte dei piedi e ostacolo")
    ax.grid(True, alpha=0.4)

    # Traiettorie recenti
    com_line, = ax.plot([], [], color="black", linewidth=1.8,
                        alpha=0.65, label="Traiettoria CoM")
    obs_line, = ax.plot([], [], color="red", linestyle="--",
                        linewidth=1.5, alpha=0.6, label="Traiettoria ostacolo")

    # Linee delle gambe: CoM -> piedi attuali
    leg0_line, = ax.plot([], [], color="forestgreen", linewidth=1.8,
                         label="CoM–p0")
    leg1_line, = ax.plot([], [], color="purple", linewidth=1.8,
                         label="CoM–p1")

    # Tracce dei piedi, con segmenti che sfumano nel tempo
    p0_trail = LineCollection([], linewidths=2.0, zorder=2)
    p1_trail = LineCollection([], linewidths=2.0, zorder=2)
    ax.add_collection(p0_trail)
    ax.add_collection(p1_trail)

    # Impronte recenti
    p0_marks = ax.scatter([], [], marker="^", s=75, c="green",
                          edgecolors="black", zorder=4, label="Impronte p0")
    p1_marks = ax.scatter([], [], marker="v", s=75, c="purple",
                          edgecolors="black", zorder=4, label="Impronte p1")

    # CoM corrente
    com_dot, = ax.plot([], [], "ko", markersize=8, zorder=6, label="CoM")

    # Ostacolo corrente
    obstacle_now = Circle((0, 0), obs_r, color="red", alpha=0.25,
                          zorder=1, label="Ostacolo")
    ax.add_patch(obstacle_now)
    obstacle_center, = ax.plot([], [], "ro", markersize=5, zorder=5)

    # Cerchi delle posizioni passate dell'ostacolo
    ghost_circles = []

    def fading_segments(points, start, end, color):
        """Crea segmenti colorati con trasparenza decrescente nel passato."""
        if end - start < 2:
            return [], []

        segments, colors = [], []
        for j in range(start, end - 1):
            segments.append([points[j], points[j + 1]])
            age = (end - 1) - j
            alpha = max(0.05, 0.8 * (1 - age / max(1, tail_frames)))
            colors.append((*plt.matplotlib.colors.to_rgb(color), alpha))
        return segments, colors

    def update(frame):
        nonlocal ghost_circles

        start = max(0, frame - tail_frames + 1)

        # CoM e sua scia
        com_line.set_data(com[start:frame + 1, 0], com[start:frame + 1, 1])
        com_dot.set_data([com[frame, 0]], [com[frame, 1]])

        # Piedi correnti e collegamenti al CoM
        leg0_line.set_data([com[frame, 0], p0[frame, 0]],
                           [com[frame, 1], p0[frame, 1]])
        leg1_line.set_data([com[frame, 0], p1[frame, 0]],
                           [com[frame, 1], p1[frame, 1]])

        # Tracce dei piedi
        seg0, col0 = fading_segments(p0, start, frame + 1, "green")
        seg1, col1 = fading_segments(p1, start, frame + 1, "purple")
        p0_trail.set_segments(seg0)
        p0_trail.set_colors(col0)
        p1_trail.set_segments(seg1)
        p1_trail.set_colors(col1)

        # Impronte: le più vecchie diventano trasparenti
        # idx = np.arange(start, frame + 1)
        idx = np.arange(0, frame + 1, sim_cfg.steps_per_phase)  # step 0, 4, 8, 12, ...
        idx = idx[idx >= start]
        ages = frame - idx
        alphas = np.maximum(0.05, 0.85 * (1 - ages / max(1, tail_frames)))

        p0_marks.set_offsets(p0[idx])
        p1_marks.set_offsets(p1[idx])
        p0_marks.set_facecolors([
            (0.0, 0.5, 0.0, a) for a in alphas
        ])
        p1_marks.set_facecolors([
            (0.5, 0.0, 0.5, a) for a in alphas
        ])

        # Traiettoria dell'ostacolo
        obs_line.set_data(obs_traj[start:frame + 1, 0],
                          obs_traj[start:frame + 1, 1])

        # Aggiorna cerchi fantasma dell'ostacolo
        for circle in ghost_circles:
            circle.remove()
        ghost_circles = []

        # Mostra al massimo circa 12 posizioni passate, più quella corrente
        stride = max(1, tail_frames // 12)
        past_indices = list(range(start, frame, stride))

        for j in past_indices:
            age = frame - j
            alpha = max(0.04, 0.45 * (1 - age / max(1, tail_frames)))
            ghost = Circle(
                obs_traj[j, :2], obs_r,
                fill=False, edgecolor="red", linewidth=1.4, alpha=alpha,
                zorder=1
            )
            ax.add_patch(ghost)
            ghost_circles.append(ghost)

        # Ostacolo corrente: centro e raggio
        obstacle_now.center = obs_traj[frame, :2]
        obstacle_center.set_data(
            [obs_traj[frame, 0]], [obs_traj[frame, 1]]
        )

        ax.set_title(
            f"Step {frame + 1}/{n} — raggio ostacolo: {obs_r:.2f} m"
        )

        return (
            com_line, com_dot, leg0_line, leg1_line,
            p0_trail, p1_trail, p0_marks, p1_marks,
            obs_line, obstacle_now, obstacle_center
        )

    ax.legend(loc="best")
    animation = FuncAnimation(
        fig, update, frames=n, interval=interval,
        blit=False, repeat=False
    )

    return animation




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
    draw_interval = sim_cfg.steps_per_phase
    
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

    # ---------------------------------------------------------
    # 3. Ostacolo (Traiettoria e posizioni durante la simulazione)
    # Sostituisci `obs_traj` con il nome del tuo array se diverso (es. `obs_pos` se è 2D)
    # Disegna la linea continua della traiettoria dell'ostacolo
    ax1.plot(traj_obs[:, 0], traj_obs[:, 1], color='red', linestyle='--', linewidth=1.5, label='Obs Path', zorder=2)
    
    # Disegna la posizione dell'ostacolo in sincronia con il campionamento del robot
    ax1.scatter(traj_obs[::draw_interval, 0], traj_obs[::draw_interval, 1], 
                c='red', marker='s', s=40, label='Obstacle', alpha=0.7, zorder=3)
    # ---------------------------------------------------------

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



    #--- 7. Plotting della Distanza CoM - Ostacolo ---
    n_steps = min(len(history_X), len(history_obs))
    
    # CORRETTO: Rinominata la variabile per evitare il "variable shadowing"
    com_obs_distances = []
    for i in range(n_steps):
        com_pos = history_X[i][0:2] 
        obs_pos = history_obs[i]    
        dist = np.linalg.norm(com_pos - obs_pos)
        com_obs_distances.append(dist)
    
    fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
    ax_dist.plot(range(n_steps), com_obs_distances, label='Distanza CoM - Ostacolo', color='blue', linewidth=2)
    
    ax_dist.axhline(y=obs_r, color='red', linestyle='--', label='Raggio Ostacolo (Impatto)')

    ax_dist.set_xlabel('Step di simulazione')
    ax_dist.set_ylabel('Distanza (m)')
    ax_dist.set_title('Distanza Robot-Ostacolo nel tempo')
    ax_dist.grid(True)
    ax_dist.legend()
    
    ani = animate_robot_feet(
    history_X=history_X,
    history_obs=history_obs,
    obs_r=obs_r,
    interval=80,
    tail_frames=40
)


    plt.tight_layout()
    plt.show()

