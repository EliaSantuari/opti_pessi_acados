import numpy as np
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
    N_horizon = 10
    sim_steps = 200 
    c_target = np.array([2.0, 2.0])
    theta_target = np.deg2rad(45)

    # steps_per_phase indicates the numer of time steps after which the robot changes feet
    gait_planner = GaitPlanner(steps_per_phase=4)
    X = np.zeros(10)

    # Initialization of the gait
    init_gait = gait_planner.get_gait_horizon(0, N_horizon)[0]
    initial_hips = gait_planner.compute_hip_positions(X[0:3], init_gait)
    # Initial position of the selected feet exactly underneath the hips
    X[6:8] = initial_hips[0:2]
    X[8:10] = initial_hips[2:4]

    # Create the OCP
    solver = create_lipm_ocp(N=N_horizon, c_target=c_target, theta_target=theta_target, x_init=X)

    # For storing the history
    history_X = []
    history_U = []
    foot_positions_world = {'FL': [], 'FR': [], 'RL': [], 'RR': []}
    previous_phase = -1
    
    # Lists for tracking distances between feet and hips
    dist_p0_curr_list = []
    dist_p1_curr_list = []
    dist_p0_next_list = []
    dist_p1_next_list = []



    print("--- Starting Simulation MPC ---")

    for step in range(sim_steps):
        # Compute the current phase of the gait from step
        current_phase = (step // gait_planner.steps_per_phase) % 2
        # Get the gait horizon and then the current gait 
        gait_horizon = gait_planner.get_gait_horizon(step, N_horizon)
        current_gait = gait_horizon[0]

        # When the phase changes I need to set the hips on the selected ones
        if current_phase != previous_phase:
            # The first step will not enter here
            if previous_phase != -1: 
                new_hips = gait_planner.compute_hip_positions(X[0:3], current_gait)
                X[6:8] = new_hips[0:2]
                X[8:10] = new_hips[2:4]
            previous_phase = current_phase


        # "Following the carrot" game to limit the step length
        dir_target = c_target - X[0:2]
        dist_target = np.linalg.norm(dir_target)

        if dist_target > 0.15:
            local_target = X[0:2] + (dir_target / dist_target) * 0.15
        else:
            local_target = c_target # If the robot is close to the objective 

        # Update the new target for the whole horizon
        yref = np.zeros(18)
        yref[0:2] = local_target
        yref[2] = theta_target
        yref_e = np.zeros(6)
        yref_e[0:2] = local_target
        yref_e[2] = theta_target


        # Pass the parameters of the hips in the current phase
        for k in range(N_horizon):
            offset0 = gait_planner.hip_offsets[current_gait[0]]
            offset1 = gait_planner.hip_offsets[current_gait[1]]
            solver.set(k, 'p', np.hstack([offset0, offset1]))
            solver.set(k, 'yref', yref) # Update the intermediate target
        # Set the parameter of the final step because is missing
        solver.set(N_horizon, 'p', np.hstack([offset0, offset1]))
        solver.set(N_horizon, 'yref', yref_e)

        
        # Push
        if step == 30 and 0:
            X[3] -= 1.5
            X[4] -= 0.0
            
        # Solve the OCP
        solver.set(0, 'lbx', X)
        solver.set(0, 'ubx', X)
        solver.solve()

        # Extract the state and the control computed by the solver
        u_apply = solver.get(0, 'u')
        X_next = solver.get(1, 'x')

        solve_time = solver.get_stats('time_tot')

        # -----------------
        # Saving stuff for plotting

        # Save dt_var
        dt_chosen = u_apply[7]
        
        # Save footprints
        if step % gait_planner.steps_per_phase == 0:
            foot_positions_world[current_gait[0]].append(u_apply[0:2])
            foot_positions_world[current_gait[1]].append(u_apply[2:4])

        # Compute kinematic distances between foot and hip
        p0_curr, p1_curr = X[6:8], X[8:10]
        p0_next, p1_next = u_apply[0:2], u_apply[2:4]
        
        # Current (X) position of the hips 
        hips_curr = gait_planner.compute_hip_positions(X[0:3], current_gait)
        hip0_curr, hip1_curr = hips_curr[0:2], hips_curr[2:4]
        
        # Future (X_next) position scheduled by the controller
        hips_next = gait_planner.compute_hip_positions(X_next[0:3], current_gait)
        hip0_next, hip1_next = hips_next[0:2], hips_next[2:4]
        
        # Compute the Euclidian distance and save it
        dist_p0_curr_list.append(np.linalg.norm(p0_curr - hip0_next))
        dist_p1_curr_list.append(np.linalg.norm(p1_curr - hip1_next))
        dist_p0_next_list.append(np.linalg.norm(p0_next - hip0_next))
        dist_p1_next_list.append(np.linalg.norm(p1_next - hip1_next))

        
        # Print some data for debugging and monitoring
        if step % 1 == 0:
            print(f"Step {step:02} | Pos: [{X[0]:.2f}, {X[1]:.2f}] | Theta: {np.rad2deg(X[2]):.1f} deg | dt: {dt_chosen*1000:.1f} ms | Comp. time: {solve_time*1000:.1f} ms")

        history_X.append(X_next)
        history_U.append(u_apply)

        # For the future step - in the real implementation here we will have the sensors data
        X = X_next 



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
    axs_v[0].axhline(y=1.5, color='r', linestyle='--', alpha=0.8, label='Max (1.5 m/s)')
    axs_v[0].axhline(y=-1.5, color='r', linestyle='--', alpha=0.8, label='Min (-1.5 m/s)')
    axs_v[0].set_title('Longitudinal Velocity of the CoM')
    axs_v[0].set_ylabel('Velocity [m/s]')
    axs_v[0].legend(loc='upper right')
    axs_v[0].grid(True)
    
    # Lateral Velocity
    axs_v[1].plot(traj[:, 4], 'g-', linewidth=2, label=r'$\dot{c}_y$ (Velocity Y)')
    axs_v[1].axhline(y=0.45, color='r', linestyle='--', alpha=0.8, label='Max (0.45 m/s)')
    axs_v[1].axhline(y=-0.45, color='r', linestyle='--', alpha=0.8, label='Min (-0.45 m/s)')
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
    max_ext = np.sqrt(0.02) # Must be the same of lipm_model.py: max_ext_sq = 0.02
    
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
    
    # f_diff_x (longitudinal differential force)
    axs_u[1].plot(traj_u[:, 5], 'b-', linewidth=2, label=r'$\Delta f_x$')
    axs_u[1].axhline(y=0.0, color='gray', linestyle='--', alpha=0.7)
    axs_u[1].set_title('Control: longitudinal differential force (generates torque)')
    axs_u[1].set_ylabel('Force [N]')
    axs_u[1].legend(loc='upper right')
    axs_u[1].grid(True)
    
    # f_diff_y (tangential differential force)
    axs_u[2].plot(traj_u[:, 6], 'g-', linewidth=2, label=r'$\Delta f_y$')
    axs_u[2].axhline(y=0.0, color='gray', linestyle='--', alpha=0.7)
    axs_u[2].set_title('Control: tangential differential force (generates torque)')
    axs_u[2].set_ylabel('Force [N]')
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


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()