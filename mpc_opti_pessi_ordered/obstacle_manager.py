import numpy as np


class BaseObstacle:
    def __init__(self, start_pos, radius, speed, y_dot_max, obs_type):
        self.pos = np.array(start_pos, dtype=float)
        self.r_obs = radius
        self.speed = speed
        self.y_dot_max = y_dot_max
        self.obs_type = obs_type

    def update(self, dt, t_global=0.0, robot_pos=None):
        raise NotImplementedError("Method must be implemented in the sub classes")


class RandomObstacle(BaseObstacle):
    def __init__(self, start_pos, radius, speed, bounds, y_dot_max):
        super().__init__(start_pos, radius, speed, y_dot_max, "random")
        self.bounds = bounds
        self.heading = np.random.uniform(-np.pi, np.pi)

    def update(self, dt, t_global=0.0, robot_pos=None):
        noise = np.random.uniform(-1, 1) * dt
        self.heading += noise
        vx = self.speed * np.cos(self.heading)
        vy = self.speed * np.sin(self.heading)

        next_pos = self.pos + np.array([vx, vy]) * dt
        x_min, x_max, y_min, y_max = self.bounds

        if next_pos[0] < x_min or next_pos[0] > x_max:
            vx = -vx
            self.heading = np.arctan2(vy, vx)
            next_pos[0] = np.clip(next_pos[0], x_min, x_max)
        if next_pos[1] < y_min or next_pos[1] > y_max:
            vy = -vy
            self.heading = np.arctan2(vy, vx)
            next_pos[1] = np.clip(next_pos[1], y_min, y_max)

        self.pos = next_pos


class StaticObstacle(BaseObstacle):
    def __init__(self, start_pos, radius, y_dot_max):
        super().__init__(start_pos, radius, 0.0, y_dot_max, "static")

    def update(self, dt, t_global=0.0, robot_pos=None):
        pass # Stays still


class DynamicObstacle(BaseObstacle):
    def __init__(self, top_pos, bot_pos, radius, speed, y_dot_max):
        # start_pos = p_top
        super().__init__(top_pos, radius, speed, y_dot_max, "dunamic")
        self.p_top = np.array(top_pos, dtype=float)
        self.p_bot = np.array(bot_pos, dtype=float)
        self.vec = self.p_bot - self.p_top
        self.dist_tot = np.linalg.norm(self.vec)
        if self.dist_tot > 1e-5:
            self.dir_u = self.vec / self.dist_tot
        else:
            self.dir_u = np.zeros(2)

    def update(self, dt, t_global=0.0, robot_pos=None):
        if self.dist_tot > 1e-5:
            s_traveled = self.speed * t_global
            s_cycle = s_traveled % (2 * self.dist_tot)
            if s_cycle <= self.dist_tot:
                self.pos = self.p_top + self.dir_u * s_cycle
            else:
                self.pos = self.p_bot - self.dir_u * (s_cycle -  self.dist_tot)


class AdversarialObstacle(BaseObstacle):
    def __init__(self, start_pos, radius, speed, y_dot_max):
        super().__init__(start_pos, radius, speed, y_dot_max, "adversarial")

    def update(self, dt, t_global=0.0, robot_pos=None):
        if robot_pos is not None:
            dir_to_robot = robot_pos[:2] - self.pos
            dist_to_robot = np.linalg.norm(dir_to_robot)
            if dist_to_robot > 1e-3:
                dir_to_robot = dir_to_robot / dist_to_robot
                self.pos = self.pos + dir_to_robot * self.speed * dt


class CircularObstacle(BaseObstacle):
    def __init__(self, start_pos, center, radius_obs, speed, y_dot_max):
        super().__init__(start_pos, radius_obs, speed, y_dot_max, "circular")
        self.center = np.array(center, dtype=float)
        dx = self.pos[0] - self.center[0]
        dy = self.pos[1] - self.center[1]
        self.path_radius = np.sqrt(dx**2 + dy**2)

        if self.path_radius < 1e-5:
            self.theta_0 = 0.0
            self.omega = 0.0
        else:
            self.theta_0 = np.arctan2(dy, dx)
            self.omega = self.speed / self.path_radius

    def update(self, dt, t_global=0.0, robot_pos=None):
        if self.path_radius >= 1e-5:
            self.pos = np.array([
                self.center[0] + self.path_radius * np.cos(self.omega * t_global + self.theta_0),
                self.center[1] + self.path_radius * np.sin(self.omega * t_global + self.theta_0)
            ])


class ObstacleManager:
    def __init__(self, n_obs_max):
        self.n_obs_max = n_obs_max
        self.obstacles = []

    def add_obstacle(self, obs:BaseObstacle):
        if len(self.obstacles) < self.n_obs_max:
            self.obstacles.append(obs)
        else:
            print(f"ERROR: Max obstacles reached ({self.n_obs_max})")

    def update_all(self, dt, t_global=0.0, robot_pos=None):
        for obs in self.obstacles:
            obs.update(dt, t_global=t_global, robot_pos=robot_pos)

    def get_active_obstacles(self):
        return [
            {
                'pos': obs.pos.copy(),
                'r_obs': obs.r_obs,
                'y_dot_max': obs.y_dot_max,
                'type': obs.obs_type
            }
            for obs in self.obstacles
        ]