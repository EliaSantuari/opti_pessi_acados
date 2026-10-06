import numpy as np
from lipm_model import create_lipm_ocp
from gait_planner import GaitPlanner
from visualizer import plot_simulation_results
from config import RobotConfig, MPCWeights, SimulationConfig, ObstacleConfig, Limits


def get_obs_position(t, obs_cfg, robot_pos=None, t_current=0.0):
    """Posizione dell'ostacolo; stessa legge usata nella simulazione originale."""
    if obs_cfg.obs_type == "static":
        return obs_cfg.pos_init.copy()

    if obs_cfg.obs_type == "dynamic":
        p_top, p_bot = obs_cfg.top_pos_dyn, obs_cfg.bot_pos_dyn
        vec = p_bot - p_top
        dist_tot = np.linalg.norm(vec)
        if dist_tot <= 1e-5:
            return p_top.copy()
        direction = vec / dist_tot
        s_cycle = (obs_cfg.speed * t) % (2.0 * dist_tot)
        if s_cycle <= dist_tot:
            return p_top + direction * s_cycle
        return p_bot - direction * (s_cycle - dist_tot)

    if obs_cfg.obs_type == "circular":
        center = np.asarray(obs_cfg.center, dtype=float)
        start = np.asarray(obs_cfg.pos_init, dtype=float)
        delta = start - center
        radius = np.linalg.norm(delta)
        if radius < 1e-5:
            return start.copy()
        theta0 = np.arctan2(delta[1], delta[0])
        omega = obs_cfg.speed / radius
        theta = omega * t + theta0
        return center + radius * np.array([np.cos(theta), np.sin(theta)])

    if obs_cfg.obs_type == "adversarial":
        if robot_pos is None:
            return obs_cfg.pos_init.copy()
        direction = np.asarray(robot_pos) - obs_cfg.pos_init
        distance = np.linalg.norm(direction)
        if distance > 1e-3:
            direction = direction / distance
        return obs_cfg.pos_init + direction * obs_cfg.speed * max(0.0, t - t_current)

    raise ValueError(f"obs_type non riconosciuto: {obs_cfg.obs_type}")


def split_branches(v):
    """Restituisce viste/copie dei vettori ottimistico e pessimistico."""
    return np.asarray(v[:11]).copy(), np.asarray(v[11:22]).copy()


def make_plane_guess(x_branch, obs_pos, obstacle_radius, gait_planner, gait_pair, safety=0.03):
    """Piano iniziale: robot nel semipiano negativo, ostacolo in quello positivo."""
    c = x_branch[:2]
    theta = x_branch[2]
    direction = np.asarray(obs_pos) - c
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        direction = np.array([1.0, 0.0])
    else:
        direction /= norm

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    # Inviluppo proiettato del corpo e dei piedi correnti.
    corners_local = np.array([
        [ gait_planner.off_x,  gait_planner.off_y],
        [-gait_planner.off_x,  gait_planner.off_y],
        [-gait_planner.off_x, -gait_planner.off_y],
        [ gait_planner.off_x, -gait_planner.off_y],
    ])
    body_points = c + (rot @ corners_local.T).T
    feet = np.array([x_branch[6:8], x_branch[8:10]])
    robot_max = max(np.max(body_points @ direction), np.max(feet @ direction)) + safety
    obstacle_min = np.dot(direction, obs_pos) - obstacle_radius - safety
    # Il piano è equidistante tra il bordo del robot e il bordo dell'ostacolo.
    boundary = 0.5 * (robot_max + obstacle_min)
    b = -boundary
    return direction[0], direction[1], b


def nominal_control(x_branch, dt_guess, obs_pos, obs_radius, gait_planner, gait_pair):
    """Controllo di appoggio usato solo quando non esiste una soluzione precedente."""
    a_x, a_y, b = make_plane_guess(
        x_branch, obs_pos, obs_radius, gait_planner, gait_pair
    )
    return np.array([
        x_branch[6], x_branch[7], x_branch[8], x_branch[9],
        0.5, 0.5, 0.5, dt_guess, a_x, a_y, b
    ], dtype=float)


def main():
    sim_cfg = SimulationConfig()
    robot_cfg = RobotConfig()
    obs_cfg = ObstacleConfig()
    weights = MPCWeights()
    limits = Limits()
    gait_planner = GaitPlanner(
        off_x=robot_cfg.off_x,
        off_y=robot_cfg.off_y,
        steps_per_phase=sim_cfg.steps_per_phase,
    )
    N = sim_cfg.N_horizon

    # Stato iniziale e piedi del primo appoggio.
    X_sim = np.zeros(11)
    X_sim[:2] = robot_cfg.x_init
    X_sim[2] = robot_cfg.theta_init
    gait_pair = gait_planner.get_gait_horizon(0, N)[0]
    hips = gait_planner.compute_hip_positions(X_sim[:3], gait_pair)
    X_sim[6:10] = hips
    X_aug = np.r_[X_sim, X_sim]

    solver = create_lipm_ocp(
        N=N,
        c_target=sim_cfg.c_target,
        theta_target=sim_cfg.theta_target,
        x_init=X_aug,
        robot_cfg=robot_cfg,
        weights=weights,
        limits=limits,
        sim_conf=sim_cfg,
    )

    history_X, history_U, history_obs = [], [], []
    dist_p0_curr, dist_p1_curr, dist_p0_next, dist_p1_next = [], [], [], []
    time_hist = []
    foot_positions_world = {"FL": [], "FR": [], "RL": [], "RR": []}
    previous_phase = -1
    t_global = 0.0
    reached_target_count = 0

    # Traiettorie ottime dell'iterazione precedente: sono la base del warm start.
    prev_X = None  # lista di N+1 vettori di stato (22)
    prev_U = None  # lista di N vettori di controllo (22)
    last_good_u = None
    dt_guess = limits.dt_max / sim_cfg.steps_per_phase

    print("--- Starting Simulation MPC (shifted primal warm start) ---")

    for step in range(sim_cfg.sim_steps):
        phase = (step // gait_planner.steps_per_phase) % 2
        gait_pair = gait_planner.get_gait_horizon(step, N)[0]
        phase_changed = phase != previous_phase

        # Al cambio di coppia, i due nuovi piedi vengono collocati sotto i rispettivi fianchi.
        if phase_changed and previous_phase != -1:
            new_hips = gait_planner.compute_hip_positions(X_sim[:3], gait_pair)
            X_sim[6:10] = new_hips
        previous_phase = phase
        X_sim[10] = 0.0
        X_aug = np.r_[X_sim, X_sim]

        x_hist_step = X_sim.copy()
        theta = X_sim[2]
        x_hist_step[3] = X_sim[3] * np.cos(theta) + X_sim[4] * np.sin(theta)
        x_hist_step[4] = -X_sim[3] * np.sin(theta) + X_sim[4] * np.cos(theta)
        history_X.append(x_hist_step)

        obs_current = get_obs_position(t_global, obs_cfg, X_sim[:2], t_current=t_global)
        history_obs.append(obs_current.copy())

        # Shift di una tappa: x_guess[k]=x_precedente[k+1], u_guess[k]=u_precedente[k+1].
        # Il nodo terminale viene estrapolato mantenendo l'ultima previsione disponibile.
        if prev_X is None:
            X_guess = [X_aug.copy() for _ in range(N + 1)]
            U_guess = []
            for k in range(N):
                obs_k = get_obs_position(
                    t_global + (k + 1) * dt_guess, obs_cfg, X_sim[:2], t_current=t_global
                )
                op = nominal_control(X_sim, dt_guess, obs_k, obs_cfg.r_obs,
                                     gait_planner, gait_pair)
                pe = op.copy()
                U_guess.append(np.r_[op, pe])
        else:
            X_guess = [prev_X[min(k + 1, N)].copy() for k in range(N + 1)]
            U_guess = [prev_U[min(k + 1, N - 1)].copy() for k in range(N)]
            X_guess[0] = X_aug.copy()

            # Dopo il cambio di fase i vecchi punti di appoggio sono di un'altra coppia.
            # Reinizializza i punti futuri sotto i fianchi previsti, poi riprendi lo shift
            # normale alle iterazioni successive.
            if phase_changed:
                for k in range(N):
                    state_op, state_pe = split_branches(X_guess[k])
                    hip_op = gait_planner.compute_hip_positions(state_op[:3], gait_pair)
                    hip_pe = gait_planner.compute_hip_positions(state_pe[:3], gait_pair)
                    U_guess[k][0:4] = hip_op
                    U_guess[k][11:15] = hip_pe

            # Ripristina la coerenza delle posizioni dei piedi nello stato previsto.
            for k in range(1, N + 1):
                X_guess[k][6:10] = U_guess[k - 1][0:4]
                X_guess[k][17:21] = U_guess[k - 1][11:15]

        # Il vincolo di non anticipatività impone l'uguaglianza dei primi 8 controlli al nodo 0.
        U_guess[0][11:19] = U_guess[0][0:8]

        # Costruisci i parametri di ogni nodo usando la previsione dell'ostacolo sul relativo tempo.
        offset0 = gait_planner.hip_offsets[gait_pair[0]]
        offset1 = gait_planner.hip_offsets[gait_pair[1]]
        obstacle_predictions = []
        for k in range(N + 1):
            state_op, _ = split_branches(X_guess[k])
            t_pred = t_global + k * dt_guess
            obs_k = get_obs_position(t_pred, obs_cfg, state_op[:2], t_current=t_global)
            obstacle_predictions.append(obs_k)
            p_val = np.r_[offset0, offset1, obs_k, obs_cfg.r_obs, obs_cfg.y_dot_max]
            solver.set(k, "p", p_val)
            if k < N:
                yref = np.zeros(34)
                yref[0:2] = sim_cfg.c_target
                yref[17:19] = sim_cfg.c_target
                solver.set(k, "yref", yref)
            else:
                yref_e = np.zeros(12)
                yref_e[0:2] = sim_cfg.c_target
                yref_e[6:8] = sim_cfg.c_target
                solver.set(k, "yref", yref_e)

        # Rigenera i parametri a,b dei piani separatori rispetto alla previsione di ogni nodo.
        # In ramo pessimistico si tiene conto del raggio crescente presente nei vincoli del modello.
        for k in range(N):
            obs_k = obstacle_predictions[k]
            state_op, state_pe = split_branches(X_guess[k])
            a0, a1, b0 = make_plane_guess(
                state_op, obs_k, obs_cfg.r_obs, gait_planner, gait_pair
            )
            pe_radius = obs_cfg.r_obs + 0.2 + obs_cfg.y_dot_max * state_pe[10]
            ap0, ap1, bp = make_plane_guess(
                state_pe, obs_k, pe_radius, gait_planner, gait_pair
            )
            U_guess[k][8:11] = [a0, a1, b0]
            U_guess[k][19:22] = [ap0, ap1, bp]
            # La non anticipatività riguarda solo i primi 8 controlli, non i piani.
            solver.set(k, "u", U_guess[k])
            solver.set(k, "x", X_guess[k])
        solver.set(N, "x", X_guess[N])

        # Stato iniziale fissato dalle misure correnti.
        solver.set(0, "lbx", X_aug)
        solver.set(0, "ubx", X_aug)
        solver.set(0, "x", X_aug)

        status = solver.solve()
        solve_time = float(solver.get_stats("time_tot"))
        time_hist.append(solve_time)
        if status != 0:
            raise RuntimeError(
                f"acados non ha converso allo step {step} (status={status}); "
                "interrompo per non applicare un comando non verificato."
            )

        # Memorizza l'intera soluzione, non solo il primo comando.
        solved_X = [np.asarray(solver.get(k, "x")).copy() for k in range(N + 1)]
        solved_U = [np.asarray(solver.get(k, "u")).copy() for k in range(N)]
        if not all(np.all(np.isfinite(v)) for v in solved_X + solved_U):
            raise RuntimeError(f"Soluzione non finita allo step {step}.")
        prev_X, prev_U = solved_X, solved_U

        u_full = solved_U[0]
        u_apply = u_full[:11].copy()
        X_next_aug = solved_X[1]
        X_next_sim = X_next_aug[:11].copy()
        if not np.all(np.isfinite(u_apply)) or not np.all(np.isfinite(X_next_sim)):
            raise RuntimeError(f"Stato o controllo non finito allo step {step}.")
        last_good_u = u_apply.copy()

        dt_chosen = float(u_apply[7])
        t_global += dt_chosen
        if obs_cfg.obs_type == "adversarial":
            direction = X_sim[:2] - obs_cfg.pos_init
            norm = np.linalg.norm(direction)
            if norm > 1e-3:
                direction /= norm
            obs_cfg.pos_init = obs_cfg.pos_init + direction * obs_cfg.speed * dt_chosen

        if step % gait_planner.steps_per_phase == 0:
            foot_positions_world[gait_pair[0]].append(u_apply[0:2].copy())
            foot_positions_world[gait_pair[1]].append(u_apply[2:4].copy())

        hips_next = gait_planner.compute_hip_positions(X_next_sim[:3], gait_pair)
        dist_p0_curr.append(np.linalg.norm(X_sim[6:8] - hips_next[0:2]))
        dist_p1_curr.append(np.linalg.norm(X_sim[8:10] - hips_next[2:4]))
        dist_p0_next.append(np.linalg.norm(u_apply[0:2] - hips_next[0:2]))
        dist_p1_next.append(np.linalg.norm(u_apply[2:4] - hips_next[2:4]))
        history_U.append(u_apply.copy())

        print(
            f"Step {step:03d} | Pos: [{X_sim[0]:.2f}, {X_sim[1]:.2f}] | "
            f"Theta: {np.rad2deg(X_sim[2]):.1f} deg | dt: {dt_chosen*1000:.1f} ms | "
            f"Solve: {solve_time*1000:.1f} ms"
        )

        if np.linalg.norm(X_next_sim[:2] - sim_cfg.c_target) < 0.1:
            reached_target_count += 1
            if reached_target_count >= 5:
                print(f"\n[INFO] Target {sim_cfg.c_target} raggiunto allo step {step}.")
                X_sim = X_next_sim
                break
        else:
            reached_target_count = 0

        X_sim = X_next_sim

    if time_hist:
        print(f"Tempo medio di calcolo: {np.mean(time_hist) * 1000:.2f} ms")

    plot_simulation_results(
        history_X,
        history_U,
        history_obs,
        foot_positions_world,
        (dist_p0_curr, dist_p1_curr, dist_p0_next, dist_p1_next),
        (sim_cfg.c_target, sim_cfg.theta_target),
        (obs_cfg.pos_init, obs_cfg.r_obs, obs_cfg.y_dot_max),
    )


if __name__ == "__main__":
    main()
