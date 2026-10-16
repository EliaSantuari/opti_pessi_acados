import numpy as np
import casadi as cs
from lipm_model import create_lipm_ocp
from gait_planner import GaitPlanner
from config import RobotConfig, MPCWeights, SimulationConfig, ObstacleConfig, Limits
from obstacle_manager import ObstacleManager, CircularObstacle
import pandas as pd
import time

def run_monte_carlo():
    # ---- Settings base -----
    sim_cfg = SimulationConfig()
    robot_cfg = RobotConfig()
    obs_cfg = ObstacleConfig()
    weights = MPCWeights()
    limits = Limits()
    
    SEED = 1234
    np.random.seed(SEED)
    N_SIMULATIONS = 100
    
    N_OBS_MAX = 1 # Usiamo 1 solo ostacolo circolare per questo scenario
    n_u_branch = 8 + (3 * N_OBS_MAX)
    
    # Struttura dati per salvare i risultati
    results = []

    print(f"--- Avvio di {N_SIMULATIONS} simulazioni Monte Carlo ---")

    for sim_idx in range(N_SIMULATIONS):
        # 1. Campionamento parametri random per questa simulazione
        c_target = np.array(sim_cfg.c_target)
        
        # Posizione iniziale del robot (scatola 10x10 centrata nel target, dist >= 0.45m)
        while True:
            rx = c_target[0] + np.random.uniform(-5, 5)
            ry = c_target[1] + np.random.uniform(-5, 5)
            robot_init_pos = np.array([rx, ry])
            if np.linalg.norm(robot_init_pos - c_target) >= 0.45:
                break
                
        robot_cfg.x_init = robot_init_pos
        
        # Parametri ostacolo circolare
        obs_speed = np.random.uniform(0.0, 1.0)
        obs_orbit_radius = np.random.uniform(0.45, 5.0)
        obs_angle = np.random.uniform(0, 2 * np.pi)
        
        obs_center = c_target
        obs_init_pos = obs_center + obs_orbit_radius * np.array([np.cos(obs_angle), np.sin(obs_angle)])
        
        # Inizializzazione manager ostacoli
        obs_manager = ObstacleManager(n_obs_max=N_OBS_MAX)
        obs = CircularObstacle(
            start_pos=obs_init_pos, 
            center=obs_center, 
            radius_obs=obs_cfg.r_obs, 
            speed=obs_speed, 
            y_dot_max=obs_cfg.y_dot_max
        )
        obs_manager.add_obstacle(obs)
        
        # Inizializzazioni per la simulazione
        gait_planner = GaitPlanner(off_x=robot_cfg.off_x, off_y=robot_cfg.off_y, steps_per_phase=sim_cfg.steps_per_phase)
        
        X_sim = np.zeros(11)
        X_sim[0:2] = robot_cfg.x_init
        X_sim[2] = robot_cfg.theta_init
        
        init_gait = gait_planner.get_gait_horizon(0, sim_cfg.N_horizon)[0]
        initial_hips = gait_planner.compute_hip_positions(X_sim[0:3], init_gait)
        X_sim[6:8] = initial_hips[0:2]
        X_sim[8:10] = initial_hips[2:4]
        
        X_aug = np.concatenate([X_sim, X_sim])
        X_sol_t_prec = [X_aug for _ in range(sim_cfg.N_horizon + 1)]
        
        u_init = np.zeros(2 * n_u_branch)
        u_init[7] = limits.dt_max
        u_init[n_u_branch + 7] = limits.dt_max
        U_sol_t_prec = [u_init for _ in range(sim_cfg.N_horizon)]

        solver = create_lipm_ocp(
            N=sim_cfg.N_horizon, 
            c_target=c_target, 
            theta_target=sim_cfg.theta_target, 
            x_init=X_aug,
            robot_cfg=robot_cfg,
            weights=weights,
            sim_conf=sim_cfg,
            obs_conf=obs_cfg
        )

        time_hist = []
        t_global = 0.0
        dt_chosen = limits.dt_max
        
        reached_target = False
        target_reached_step = -1
        reached_target_flag = 0
        min_dist_to_target = float('inf')
        solver_failed = False

        # Loop di simulazione
        for step in range(sim_cfg.sim_steps):
            robot_pos = X_sim[0:2]
            obs_manager.update_all(dt_chosen, t_global=t_global, robot_pos=robot_pos)
            active_obstacles = obs_manager.get_active_obstacles()

            current_gait = gait_planner.get_gait_horizon(step, sim_cfg.N_horizon)[0]

            X_sim[10] = 0.0
            X_aug = np.concatenate([X_sim, X_sim])
            solver.set(0, "lbx", X_aug)
            solver.set(0, "ubx", X_aug)

            # Local target setup
            dir_to_target = c_target - X_sim[0:2]
            dist_to_final = np.linalg.norm(dir_to_target)
            if dist_to_final < min_dist_to_target:
                min_dist_to_target = dist_to_final

            max_lookahead = 0.8
            if dist_to_final > max_lookahead:
                local_target = X_sim[0:2] + (dir_to_target / dist_to_final) * max_lookahead
            else:
                local_target = c_target

            yref = np.zeros(34)
            yref[0:2] = local_target
            yref[17:19] = local_target
            yref_e = np.zeros(12)
            yref_e[0:2] = local_target
            yref_e[6:8] = local_target

            for k in range(sim_cfg.N_horizon):
                solver.set(k, 'yref', yref)
            solver.set(sim_cfg.N_horizon, 'yref', yref_e)

            gait_horizon = gait_planner.get_gait_horizon(step, sim_cfg.N_horizon)

            # Impostazione dei parametri p
            for k in range(sim_cfg.N_horizon+1):
                gait_k = gait_horizon[k]
                offset_0 = gait_planner.hip_offsets[gait_k[0]]
                offset_1 = gait_planner.hip_offsets[gait_k[1]]

                p_val = np.zeros(4 + (4 * N_OBS_MAX))
                p_val[0:4] = np.concatenate([offset_0, offset_1])

                for i in range(N_OBS_MAX):
                    idx_p = 4 + (4 * i)
                    if i < len(active_obstacles):
                        o = active_obstacles[i]
                        p_val[idx_p:idx_p+4] = [o['pos'][0], o['pos'][1], o['r_obs']+0.1, o['y_dot_max']]
                    else:
                        p_val[idx_p:idx_p+4] = [15.0 + (2*i), 15.0, 0.0, 0.0]
                solver.set(k, "p", p_val)                    

            # Warm-start controller
            for k in range(sim_cfg.N_horizon):
                gait_k = gait_horizon[k]
                idx_shift_u = k + 1 if k < (sim_cfg.N_horizon-1) else k
                idx_shift_x = k + 1
                x_prev = X_sol_t_prec[idx_shift_x]
                u_prev = U_sol_t_prec[idx_shift_u]

                # (Logica base di warm start omessa parzialmente per brevità, riprendo il calcolo base del tuo script)
                com_guess_op = x_prev[0:2]
                hips_guess_op = gait_planner.compute_hip_positions([x_prev[0], x_prev[1], x_prev[2]], gait_k)
                com_guess_pe = x_prev[11:13]
                hips_guess_pe = gait_planner.compute_hip_positions([x_prev[11], x_prev[12], x_prev[13]], gait_k)
                
                u_guess_obs_op = []
                u_guess_obs_pe = []
                
                for i in range(N_OBS_MAX):
                    obs_pos_k = active_obstacles[i]['pos'] if i < len(active_obstacles) else np.array([15.0, 15.0])
                    
                    # Semipiani OP
                    min_dist_op = min(np.linalg.norm(obs_pos_k - com_guess_op), np.linalg.norm(obs_pos_k - hips_guess_op[0:2]), np.linalg.norm(obs_pos_k - hips_guess_op[2:4]))
                    closest_pt_op = hips_guess_op[0:2] if min_dist_op == np.linalg.norm(obs_pos_k - hips_guess_op[0:2]) else (hips_guess_op[2:4] if min_dist_op == np.linalg.norm(obs_pos_k - hips_guess_op[2:4]) else com_guess_op)
                    dir_to_obs_op = obs_pos_k - closest_pt_op
                    d_op = np.linalg.norm(dir_to_obs_op) + 1e-5
                    a_g_op = dir_to_obs_op / d_op
                    b_g_op = -np.dot(a_g_op, (closest_pt_op + obs_pos_k) / 2.0)
                    u_guess_obs_op.extend([a_g_op[0], a_g_op[1], b_g_op])

                    # Semipiani PE
                    min_dist_pe = min(np.linalg.norm(obs_pos_k - com_guess_pe), np.linalg.norm(obs_pos_k - hips_guess_pe[0:2]), np.linalg.norm(obs_pos_k - hips_guess_pe[2:4]))
                    closest_pt_pe = hips_guess_pe[0:2] if min_dist_pe == np.linalg.norm(obs_pos_k - hips_guess_pe[0:2]) else (hips_guess_pe[2:4] if min_dist_pe == np.linalg.norm(obs_pos_k - hips_guess_pe[2:4]) else com_guess_pe)
                    dir_to_obs_pe = obs_pos_k - closest_pt_pe
                    d_pe = np.linalg.norm(dir_to_obs_pe) + 1e-5
                    a_g_pe = dir_to_obs_pe / d_pe
                    b_g_pe = -np.dot(a_g_pe, (closest_pt_pe + obs_pos_k) / 2.0)
                    u_guess_obs_pe.extend([a_g_pe[0], a_g_pe[1], b_g_pe])

                u_guess_op_base = np.array([hips_guess_op[0], hips_guess_op[1], hips_guess_op[2], hips_guess_op[3], 0.5, 0.0, 0.0, u_prev[7]])
                u_guess_pe_base = np.array([hips_guess_pe[0], hips_guess_pe[1], hips_guess_pe[2], hips_guess_pe[3], 0.5, 0.0, 0.0, u_prev[n_u_branch+7]])

                u_g_op = np.concatenate([u_guess_op_base, u_guess_obs_op])
                u_g_pe = np.concatenate([u_guess_pe_base, u_guess_obs_pe])
                if k == 0: u_g_op = u_g_pe
                solver.set(k, 'u', np.concatenate([u_g_op, u_g_pe]))

                if k > 0:
                    x_guess_op = np.zeros(11)
                    x_guess_op[0:2] = com_guess_op
                    x_guess_op[2:6] = x_prev[2:6]
                    x_guess_op[6:10] = x_prev[6:10]
                    x_guess_op[10] = u_prev[7]

                    x_guess_pe = np.zeros(11)
                    x_guess_pe[0:2] = com_guess_pe
                    x_guess_pe[2:6] = x_prev[13:17]
                    x_guess_pe[6:10] = x_prev[17:21]
                    x_guess_pe[10] = u_prev[n_u_branch+7]
                    solver.set(k, 'x', np.concatenate([x_guess_op, x_guess_pe]))

            solver.set(sim_cfg.N_horizon, 'x', np.concatenate([x_guess_op, x_guess_pe]))

            # Risoluzione
            status = solver.solve()
            
            # Controllo fallimento solver
            if status != 0 and status != 2:
                solver_failed = True

            solve_time = solver.get_stats('time_tot')
            time_hist.append(solve_time)

            X_next_22 = solver.get(1, 'x')
            u_opt_22 = solver.get(0, 'u')
            u_apply = u_opt_22[0:n_u_branch]
            X_next_sim = X_next_22[11:22]

            X_sol_t_prec = [solver.get(k, 'x') for k in range(sim_cfg.N_horizon)] + [solver.get(sim_cfg.N_horizon, 'x')]
            U_sol_t_prec = [solver.get(k, 'u') for k in range(sim_cfg.N_horizon)]

            dt_chosen = u_apply[7]
            t_global += dt_chosen

            # Controllo target raggiunto
            if dist_to_final < 0.1:
                reached_target_flag += 1
                if reached_target_flag == 5:
                    reached_target = True
                    target_reached_step = step
                    break

            X_sim = X_next_sim

        # Fine della simulazione singola: colleziona le statistiche
        avg_comp_time = np.mean(time_hist) * 1000 if time_hist else 0.0
        time_to_reach = t_global if reached_target else float('inf')

        results.append({
            'sim_idx': sim_idx,
            'success': reached_target,
            'min_dist_to_target_m': min_dist_to_target if not reached_target else 0.0,
            'time_to_reach_s': time_to_reach,
            'avg_comp_time_ms': avg_comp_time,
            'solver_failed': solver_failed,
            'obs_speed': obs_speed,
            'obs_orbit_radius': obs_orbit_radius,
            'robot_init_pos': robot_init_pos.tolist()
        })
        
        print(f"Sim {sim_idx:03d} | Successo: {reached_target} | Tempo: {time_to_reach:.2f}s | Min Dist: {min_dist_to_target:.2f}m | T_comp: {avg_comp_time:.1f}ms | Fail: {solver_failed}")

    # Riepilogo finale
    df = pd.DataFrame(results)
    print("\n" + "="*40)
    print("      RISULTATI MONTE CARLO (100 run)")
    print("="*40)
    print(f"Target raggiunto:       {df['success'].sum()}/{N_SIMULATIONS}")
    print(f"Distanza minima (falliti): {df.loc[~df['success'], 'min_dist_to_target_m'].mean():.3f} m (media)")
    print(f"Tempo medio per arrivare: {df.loc[df['success'], 'time_to_reach_s'].mean():.2f} s")
    print(f"Tempo calcolo medio:    {df['avg_comp_time_ms'].mean():.2f} ms")
    print(f"Fallimenti del solver:  {df['solver_failed'].sum()} volte")
    print("="*40)

    # df.to_csv("monte_carlo_results.csv", index=False)
    # print("Risultati salvati in 'monte_carlo_results.csv'")

if __name__ == "__main__":
    run_monte_carlo()