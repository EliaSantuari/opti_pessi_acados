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



def plot_friction_cones(cone_0_hist, cone_1_hist, f0_hist, f1_hist):
    cone0 = np.array(cone_0_hist)
    cone1 = np.array(cone_1_hist)

    plt.figure(figsize=(10, 6))
    plt.suptitle('Friction Cones VS Actual Forces', fontsize=14, fontweight='bold')

    # Primo grafico - Piede 0
    plt.subplot(2, 1, 1)
    plt.plot(cone0, label='Friction cone of foot 0', color='#1f77b4', linewidth=1)
    plt.plot(f0_hist, label="Force on foot 0", color='#ff7f0e', linewidth=1)
    plt.ylabel('Forces [N]')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    # Secondo grafico - Piede 1
    plt.subplot(2, 1, 2)
    plt.plot(cone1, label='Friction cone of foot 1', color='#1f77b4', linewidth=1)
    plt.plot(f1_hist, label="Force on foot 1", color='#ff7f0e', linewidth=1)
    plt.xlabel('Step')
    plt.ylabel('Forces [N]')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    # Sistemazione del layout per evitare sovrapposizioni
    plt.tight_layout()




def plot_computational_time(comp_time_hist, deadline_ms = 25):
    times = np.array(comp_time_hist)*1000
    
    # Calcolo statistiche
    mean_time = np.mean(times)
    max_time = np.max(times)
    missed_deadlines = np.sum(times > deadline_ms)
    miss_rate = (missed_deadlines / len(times)) * 100
    
    # print(f"--- Statistiche MPC ---")
    # print(f"Tempo Medio: {mean_time:.3f} ms")
    # print(f"Tempo Massimo (Worst-case): {max_time:.3f} ms")
    # print(f"Deadline Miss Rate (> {deadline_ms} ms): {miss_rate:.2f}% ({missed_deadlines} step)")
    
    # Creazione del plot
    plt.figure(figsize=(10, 5))
    plt.plot(times, label='Tempo di computazione', color='#1f77b4', linewidth=1)
    
    # Linee di riferimento
    plt.axhline(y=deadline_ms, color='red', linestyle='--', 
                label=f'Real-time Deadline ({deadline_ms} ms)')
    plt.axhline(y=mean_time, color='green', linestyle='-.', 
                label=f'Media: {mean_time:.2f} ms')
    
    # Formattazione
    plt.title('Analisi dei Tempi di Computazione MPC')
    plt.xlabel('Step di controllo')
    plt.ylabel('Tempo di computazione (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

def animate_robot_feet(history_X, history_obs, pred_op, pred_pe, local_target_hist=None, interval=80, tail_frames=40):
    traj = np.asarray(history_X)
    n = min(len(traj), len(history_obs))
    traj = traj[:n]
    history_obs = history_obs[:n]

    # Determiniamo il numero di ostacoli dal primo step
    first_step_obs = history_obs[0]
    n_obs = len(first_step_obs) if isinstance(first_step_obs, list) else 1

    obs_traj = np.zeros((n, n_obs, 2))
    obs_radii = np.zeros((n, n_obs))

    for i, step_obs in enumerate(history_obs):
        if isinstance(step_obs, list):
            for m, obs_dict in enumerate(step_obs):
                if m < n_obs:
                    obs_traj[i, m] = np.asarray(obs_dict['pos'])[:2]
                    obs_radii[i, m] = float(obs_dict['r_obs'])
        else:
            obs_traj[i, 0] = np.asarray(step_obs['pos'])[:2]
            obs_radii[i, 0] = float(step_obs['r_obs'])

    pred_op = None if pred_op is None else np.asarray(pred_op)
    pred_pe = None if pred_pe is None else np.asarray(pred_pe)

    loc_target = None
    if local_target_hist is not None:
        loc_target = np.asarray(local_target_hist)
        if loc_target.ndim != 2 or loc_target.shape[1] != 2:
            loc_target = None

    if pred_op is not None: n = min(n, len(pred_op))
    if pred_pe is not None: n = min(n, len(pred_pe))
    if loc_target is not None: n = min(n, len(loc_target))

    traj, obs_traj, obs_radii = traj[:n], obs_traj[:n], obs_radii[:n]
    if pred_op is not None: pred_op = pred_op[:n]
    if pred_pe is not None: pred_pe = pred_pe[:n]
    if loc_target is not None: loc_target = loc_target[:n]

    com = traj[:, 0:2]
    p0 = traj[:, 6:8]
    p1 = traj[:, 8:10]

    fig, ax = plt.subplots(figsize=(9, 7))

    max_radius = float(np.max(obs_radii)) if obs_radii.size > 0 else 0.5
    all_x = np.concatenate((com[:, 0], p0[:, 0], p1[:, 0], obs_traj[:, :, 0].flatten()))
    all_y = np.concatenate((com[:, 1], p0[:, 1], p1[:, 1], obs_traj[:, :, 1].flatten()))

    if pred_op is not None:
        all_x = np.concatenate((all_x, pred_op[:, :, 0].ravel()))
        all_y = np.concatenate((all_y, pred_op[:, :, 1].ravel()))
    if pred_pe is not None:
        all_x = np.concatenate((all_x, pred_pe[:, :, 0].ravel()))
        all_y = np.concatenate((all_y, pred_pe[:, :, 1].ravel()))
    if loc_target is not None:
        all_x = np.concatenate((all_x, loc_target[:, 0].flatten()))
        all_y = np.concatenate((all_y, loc_target[:, 1].flatten()))

    margin = max_radius + 0.3
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Animazione CoM, impronte dei piedi e ostacoli")
    ax.grid(True, alpha=0.4)

    target_x, target_y = float(sim_cfg.c_target[0]), float(sim_cfg.c_target[1])
    ax.plot(target_x, target_y, marker="x", markersize=10, markeredgewidth=2.5, color="dodgerblue", zorder=5, label="Target Globale")

    com_line, = ax.plot([], [], color="black", linewidth=1.8, alpha=0.65, label="Traiettoria CoM")

    obs_lines = [ax.plot([], [], color="red", linestyle="--", linewidth=1.5, alpha=0.6)[0] for _ in range(n_obs)]
    obstacles_now = []
    obstacle_centers = []
    for m in range(n_obs):
        init_r = obs_radii[0, m]
        patch = Circle((0, 0), init_r, color="red", alpha=0.25, zorder=1)
        ax.add_patch(patch)
        obstacles_now.append(patch)
        center, = ax.plot([], [], "ro", markersize=5, zorder=5)
        obstacle_centers.append(center)

    obs_lines[0].set_label("Traiettoria ostacoli")
    obstacles_now[0].set_label("Ostacoli")

    leg0_line, = ax.plot([], [], color="forestgreen", linewidth=1.8, label="CoM–p0")
    leg1_line, = ax.plot([], [], color="purple", linewidth=1.8, label="CoM–p1")

    p0_trail = LineCollection([], linewidths=2.0, zorder=2)
    p1_trail = LineCollection([], linewidths=2.0, zorder=2)
    ax.add_collection(p0_trail)
    ax.add_collection(p1_trail)

    p0_marks = ax.scatter([], [], marker="^", s=75, c="green", edgecolors="black", zorder=4, label="Impronte p0")
    p1_marks = ax.scatter([], [], marker="v", s=75, c="purple", edgecolors="black", zorder=4, label="Impronte p1")
    com_dot, = ax.plot([], [], "ko", markersize=8, zorder=6, label="CoM")

    pred_op_line, = ax.plot([], [], color="deepskyblue", linestyle="--", marker=".", markersize=5, linewidth=2, label="Horizon ottimistico", zorder=7)
    pred_pe_line, = ax.plot([], [], color="darkorange", linestyle="--", marker=".", markersize=5, linewidth=2, label="Horizon pessimistico", zorder=7)
    
    loc_target_mark, = ax.plot([], [], marker="*", markersize=12, markeredgewidth=1.5, color="red", zorder=10, label="Local Target")

    ghost_circles = []

    def fading_segments(points, start, end, color):
        if end - start < 2: return [], []
        segments, colors = [], []
        for j in range(start, end - 1):
            segments.append([points[j], points[j + 1]])
            age = (end - 1) - j
            alpha = max(0.05, 0.8 * (1 - age / max(1, tail_frames)))
            colors.append((*plt.matplotlib.colors.to_rgb(color), alpha))
        return segments, colors

    def update(frame):
        nonlocal ghost_circles
        if pred_op is not None: pred_op_line.set_data(pred_op[frame, :, 0], pred_op[frame, :, 1])
        if pred_pe is not None: pred_pe_line.set_data(pred_pe[frame, :, 0], pred_pe[frame, :, 1])

        if loc_target is not None:
            loc_target_mark.set_data([loc_target[frame, 0]], [loc_target[frame, 1]])

        start = max(0, frame - tail_frames + 1)
        com_line.set_data(com[start:frame + 1, 0], com[start:frame + 1, 1])
        com_dot.set_data([com[frame, 0]], [com[frame, 1]])
        leg0_line.set_data([com[frame, 0], p0[frame, 0]], [com[frame, 1], p0[frame, 1]])
        leg1_line.set_data([com[frame, 0], p1[frame, 0]], [com[frame, 1], p1[frame, 1]])

        seg0, col0 = fading_segments(p0, start, frame + 1, "green")
        seg1, col1 = fading_segments(p1, start, frame + 1, "purple")
        p0_trail.set_segments(seg0)
        p0_trail.set_colors(col0)
        p1_trail.set_segments(seg1)
        p1_trail.set_colors(col1)

        idx = np.arange(0, frame + 1, sim_cfg.steps_per_phase)
        idx = idx[idx >= start]
        ages = frame - idx
        alphas = np.maximum(0.05, 0.85 * (1 - ages / max(1, tail_frames)))
        p0_marks.set_offsets(p0[idx])
        p1_marks.set_offsets(p1[idx])
        p0_marks.set_facecolors([(0.0, 0.5, 0.0, a) for a in alphas])
        p1_marks.set_facecolors([(0.5, 0.0, 0.5, a) for a in alphas])

        for circle in ghost_circles:
            circle.remove()
        ghost_circles = []

        stride = max(1, tail_frames // 12)
        past_indices = list(range(start, frame, stride))

        for m in range(n_obs):
            obs_lines[m].set_data(obs_traj[start:frame + 1, m, 0], obs_traj[start:frame + 1, m, 1])
            
            obstacles_now[m].center = obs_traj[frame, m, :2]
            obstacles_now[m].set_radius(obs_radii[frame, m])
            obstacle_centers[m].set_data([obs_traj[frame, m, 0]], [obs_traj[frame, m, 1]])

            for j in past_indices:
                age = frame - j
                alpha = max(0.04, 0.45 * (1 - age / max(1, tail_frames)))
                ghost = Circle(obs_traj[j, m, :2], obs_radii[j, m], fill=False, edgecolor="red", linewidth=1.4, alpha=alpha, zorder=1)
                ax.add_patch(ghost)
                ghost_circles.append(ghost)

        ax.set_title(f"Step {frame + 1}/{n} — Ostacoli attivi: {n_obs}")

    ax.legend(loc="best")
    animation = FuncAnimation(fig, update, frames=n, interval=interval, blit=False, repeat=False)
    return animation


def plot_simulation_results(history_X, history_U, history_obs, foot_positions_world, distances, target, obs_params, history_pred_op, history_pred_pe, local_target_hist):
    traj = np.array(history_X)
    traj_u = np.array(history_U)

    # PARSING SICURO STORICO OSTACOLI CON RAGGI
    n_steps = len(history_obs)
    n_obs = len(history_obs[0]) if isinstance(history_obs[0], list) else 1
    
    traj_obs = np.zeros((n_steps, n_obs, 2))
    obs_radii = np.zeros((n_steps, n_obs))
    
    for i, step_obs in enumerate(history_obs):
        if isinstance(step_obs, list):
            for m, obs_dict in enumerate(step_obs):
                if m < n_obs:
                    traj_obs[i, m] = np.asarray(obs_dict['pos'])[:2]
                    obs_radii[i, m] = float(obs_dict['r_obs'])
        else:
            traj_obs[i, 0] = np.asarray(step_obs['pos'])[:2]
            obs_radii[i, 0] = float(step_obs['r_obs'])

    dt_mean = np.mean(traj_u[:, 7])
    
    c_target, theta_target = target
    dist_p0_curr, dist_p1_curr, dist_p0_next, dist_p1_next = distances
    draw_interval = sim_cfg.steps_per_phase

    # --- 2. Velocities ---
    fig_v, axs_v = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axs_v[0].plot(traj[:, 3], 'b-', linewidth=2, label=r'$\dot{c}_x$')
    axs_v[0].axhline(y=limits.v_max_x, color='r', linestyle='--'); axs_v[0].axhline(y=-limits.v_max_x, color='r', linestyle='--')
    axs_v[0].set_title('Longitudinal Velocity'); axs_v[0].grid(True)
    
    axs_v[1].plot(traj[:, 4], 'g-', linewidth=2, label=r'$\dot{c}_y$')
    axs_v[1].axhline(y=limits.v_max_y, color='r', linestyle='--'); axs_v[1].axhline(y=-limits.v_max_y, color='r', linestyle='--')
    axs_v[1].set_title('Lateral Velocity'); axs_v[1].grid(True)
    
    axs_v[2].plot(traj[:, 5], 'm-', linewidth=2, label=r'$\dot{\theta}$')
    axs_v[2].axhline(y=limits.theta_dot, color='r', linestyle='--'); axs_v[2].axhline(y=-limits.theta_dot, color='r', linestyle='--')
    axs_v[2].set_title('Angular Velocity'); axs_v[2].grid(True)

    # --- 3. Controls  ---
    fig_u, axs_u = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axs_u[0].plot(traj_u[:, 4], 'm-', linewidth=2); axs_u[0].axhline(y=0.5, color='gray', linestyle='--')
    axs_u[0].set_title('Alpha'); axs_u[0].grid(True)
    axs_u[1].plot(traj_u[:, 5], 'b-', linewidth=2); axs_u[1].set_title('Beta'); axs_u[1].grid(True)
    axs_u[2].plot(traj_u[:, 6], 'g-', linewidth=2); axs_u[2].set_title('Gamma'); axs_u[2].grid(True)
    
    dt_ms = traj_u[:, 7] * 1000 
    axs_u[3].plot(dt_ms, 'r-', linewidth=2, label='optimal dt')
    dt_min_ms = (limits.dt_min/sim_cfg.steps_per_phase)*1000; dt_max_ms = (limits.dt_max/sim_cfg.steps_per_phase)*1000
    axs_u[3].axhline(y=dt_min_ms, color='k', linestyle=':'); axs_u[3].axhline(y=dt_max_ms, color='k', linestyle=':')
    axs_u[3].set_title('Optimal dt [ms]'); axs_u[3].grid(True)

    # --- 4. Extensions  ---
    max_ext = np.sqrt(robot_cfg.max_ext_sq)
    fig_d, axs_d = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs_d[0].plot(dist_p0_next, 'b-'); axs_d[0].plot(dist_p1_next, 'g-'); axs_d[0].axhline(y=max_ext, color='r', linestyle='--')
    axs_d[0].set_title('Future extension'); axs_d[0].grid(True)
    axs_d[1].plot(dist_p0_curr, 'b-'); axs_d[1].plot(dist_p1_curr, 'g-'); axs_d[1].axhline(y=max_ext, color='r', linestyle='--')
    axs_d[1].set_title('Current extension'); axs_d[1].grid(True)

    # --- 5. PLOT PULITO ---
    fig_clean, ax_clean = plt.subplots(figsize=(10, 8))
    ax_clean.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2.5, label='Traiettoria CoM Robot')
    ax_clean.plot(traj[0, 0], traj[0, 1], 'bo', markersize=8); ax_clean.plot(traj[-1, 0], traj[-1, 1], 'b*', markersize=12)
    ax_clean.plot(c_target[0], c_target[1], 'gx', markersize=15, markeredgewidth=3, label='Target')

    step_interval = max(1, len(traj_obs) // 8) 
    
    for m in range(n_obs):
        label_str = 'Ostacoli' if m == 0 else "" 
        ax_clean.plot(traj_obs[:, m, 0], traj_obs[:, m, 1], 'r--', linewidth=2, label=label_str)
        ax_clean.plot(traj_obs[0, m, 0], traj_obs[0, m, 1], 'ro', markersize=8)
        ax_clean.plot(traj_obs[-1, m, 0], traj_obs[-1, m, 1], 'r*', markersize=12)
        
        for i in range(0, len(traj_obs), step_interval):
            alpha_val = 0.2 + 0.6 * (i / len(traj_obs)) 
            # NOTA: qui uso obs_radii specifico per il plot, non obs_r
            ax_clean.add_patch(plt.Circle(traj_obs[i, m], obs_radii[i, m], color='red', fill=False, linewidth=1.5, alpha=alpha_val))
        ax_clean.add_patch(plt.Circle(traj_obs[-1, m], obs_radii[-1, m], color='red', fill=True, alpha=0.3))

    ax_clean.set_title('Traiettoria CoM Robot vs Ostacoli Dinamici')
    ax_clean.legend(loc='best'); ax_clean.grid(True); ax_clean.axis('equal')

    # --- 1. Plot Globale Impronte e Traiettorie ---
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(traj[:, 0], traj[:, 1], color='#40c4c4', linewidth=2.5, label='Opti-Pessi MPC', zorder=2)
    ax1.plot(traj[:, 0], traj[:, 1], 'o', color='#40c4c4', markersize=4, zorder=2)
    ax1.plot(traj[0, 0], traj[0, 1], 'bs', markersize=7, label='Start', zorder=3)
    ax1.plot(c_target[0], c_target[1], 'v', color='blue', markersize=7, label='Goal', zorder=3)

    for m in range(n_obs):
        label_str = 'Obs Path' if m == 0 else ""
        ax1.plot(traj_obs[:, m, 0], traj_obs[:, m, 1], color='red', linestyle='--', linewidth=1.5, label=label_str, zorder=2)
        ax1.scatter(traj_obs[::draw_interval, m, 0], traj_obs[::draw_interval, m, 1], c='red', marker='s', s=40, alpha=0.7, zorder=3)
    
    p0_traj, p1_traj = traj[:, 6:8], traj[:, 8:10]
    ax1.scatter(p0_traj[::draw_interval, 0], p0_traj[::draw_interval, 1], c='green', marker='^', s=40, label='p0', alpha=0.9, zorder=3)
    ax1.scatter(p1_traj[::draw_interval, 0], p1_traj[::draw_interval, 1], c='purple', marker='v', s=40, label='p1', alpha=0.9, zorder=3)

    for i in range(0, len(traj), draw_interval):
        ax1.plot([traj[i, 0], p0_traj[i, 0]], [traj[i, 1], p0_traj[i, 1]], color='gray', linestyle='-', linewidth=0.5, alpha=0.6)
        ax1.plot([traj[i, 0], p1_traj[i, 0]], [traj[i, 1], p1_traj[i, 1]], color='gray', linestyle='-', linewidth=0.5, alpha=0.6)

    ax1.set_title(f'Opti-Pessi MPC, N = {sim_cfg.N_horizon}, sim_T = {len(traj)*dt_mean:.2f} s', fontsize=14)
    ax1.legend(loc='upper left'); ax1.grid(True); ax1.axis('equal')

    #--- 7. Distanza CoM - Ostacolo ---
    n_steps_dist = min(len(history_X), len(traj_obs))
    
    com_obs_distances = {m: [] for m in range(n_obs)}
    for i in range(n_steps_dist):
        com_pos = history_X[i][0:2] 
        for m in range(n_obs):
            # RIPARATO: np.linalg.norm con il segno meno corretto
            com_obs_distances[m].append(np.linalg.norm(com_pos - traj_obs[i, m]))
    
    fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
    colors = plt.cm.get_cmap('tab10', max(10, n_obs))
    for m in range(n_obs):
        ax_dist.plot(range(n_steps_dist), com_obs_distances[m], label=f'Ostacolo {m+1}', color=colors(m), linewidth=2)
        # Plot del raggio dinamico (media o massimo dell'ostacolo per referenza)
        mean_radius = np.mean(obs_radii[:, m])
        ax_dist.axhline(y=mean_radius, color=colors(m), linestyle='--', alpha=0.5)

    ax_dist.set_title('Distanza Robot-Ostacoli'); ax_dist.grid(True); ax_dist.legend()
    
    ani = animate_robot_feet(history_X=history_X, history_obs=history_obs, pred_op=history_pred_op, pred_pe=history_pred_pe, interval=100, tail_frames=10, local_target_hist=local_target_hist)

    plt.tight_layout()
    plt.show()

