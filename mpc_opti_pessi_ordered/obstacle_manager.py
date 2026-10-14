import numpy as np


class RandomObstacle:
    def __init__(self, start_pos, radius, speed,  bounds, y_dot_max):
        """
        bounds: [x_min, x_max, y_min, y_max] define the are in which the obstacle can move
        """

        self.pos = np.array(start_pos, dtype=float)
        self.r_obs = radius
        self.speed = speed
        self.bounds = bounds

        self.y_dot_max = y_dot_max

        # Choose an initial random direction in rad
        self.heading = np.random.uniform(-np.pi, np.pi)

    def update(self, dt):
        # Add noise to the direction for a smooth movement but random
        noise = np.random.normal(0, 1)
        self.heading += noise * dt

        # Compute the velocity
        vx = self.speed * np.cos(self.heading)
        vy = self.speed * np.sin(self.heading)

        # Compute the next position
        next_pos = self.pos + np.array([vx, vy]) * dt

        # Check the bounds, if it touches bounds it bounces back
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


class ObstacleManager:
    def __init__(self, n_obs_max):
        self.n_obs_max = n_obs_max
        self.obstacles = []

    def add_random_obstacle(self, start_pos, radius, speed, bounds, y_dot_max):
        if len(self.obstacles) < self.n_obs_max:
            obs = RandomObstacle(start_pos, radius, speed, bounds, y_dot_max)
            self.obstacles.append(obs)
        else:
            print(f"ERROR: Max onstacle reached ({self.n_obs_max})")


    def update_all(self, dt):
        for obs in self.obstacles:
            obs.update(dt)

    def get_active_obstacles(self):
        return [
            {
                'pos': obs.pos.copy(),
                'r_obs':obs.r_obs,
                'y_dot_max': obs.y_dot_max
            }
            for obs in self.obstacles
        ]