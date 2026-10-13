import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pygame

from config import (
    CELL_SIZE,
    EPISODE_LIMIT,
    GOAL_PROGRESS_REWARD_SCALE,
    GOAL_MARGIN,
    GOAL_REWARD,
    MAX_TURN_RATE,
    MAZE_COLS,
    MAZE_ROWS,
    MOVEMENT_ACCELERATION,
    MOVEMENT_DRAG,
    MOVEMENT_SPEED,
    PLAYER_SIZE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    STEP_PENALTY,
    TURN_ACCELERATION,
    TURN_DRAG,
    WALL_BOUNCE_FACTOR,
    WALL_COLLISION_MARGIN,
    WALL_COLLISION_PENALTY,
    WALL_PROXIMITY_RADIUS,
    WALL_PROXIMITY_REWARD_SCALE,
)

TOP, RIGHT, BOTTOM, LEFT = range(4)
WALL_DIRECTIONS = (
    (0, -1, TOP, BOTTOM),
    (1, 0, RIGHT, LEFT),
    (0, 1, BOTTOM, TOP),
    (-1, 0, LEFT, RIGHT),
)

# I had to put this in a gymnasium environment so that it would work with stable_baselines3 PPO
class MazeGame(gym.Env):

    def __init__(self, visualize=False, seed=None):
        super().__init__()
        self.visualize = visualize and (pygame is not None)
        self.random = random if seed is None else random.Random(seed)

        if self.visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        self.maze = self._generate_maze()
        self.start_x = CELL_SIZE // 2
        self.start_y = CELL_SIZE // 2
        
        self.goal_rect = pygame.Rect(
            (MAZE_COLS - 1) * CELL_SIZE + GOAL_MARGIN,
            (MAZE_ROWS - 1) * CELL_SIZE + GOAL_MARGIN,
            CELL_SIZE - 2 * GOAL_MARGIN,
            CELL_SIZE - 2 * GOAL_MARGIN,
        )

        self.action_space = spaces.Discrete(3) 

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
        self.reset(seed=seed)

    def _generate_maze(self):
        maze = [[[True, True, True, True] for _ in range(MAZE_COLS)] for _ in range(MAZE_ROWS)]
        visited = [[False for _ in range(MAZE_COLS)] for _ in range(MAZE_ROWS)]

        def carve_passages(column, row):
            visited[row][column] = True
            directions = list(WALL_DIRECTIONS)
            self.random.shuffle(directions)

            for delta_column, delta_row, wall, opposite_wall in directions:
                next_column = column + delta_column
                next_row = row + delta_row
                is_inside_maze = (0 <= next_column < MAZE_COLS and 0 <= next_row < MAZE_ROWS)

                if is_inside_maze and not visited[next_row][next_column]:
                    maze[row][column][wall] = False
                    maze[next_row][next_column][opposite_wall] = False
                    carve_passages(next_column, next_row)

        carve_passages(0, 0)
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.random = random.Random(seed)

        self.x = self.start_x
        self.y = self.start_y
        self.angle = 0
        self.angular_velocity = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.steps = 0
        self.reached_goal = False

        return self._get_obs(), {}

    def _is_wall_at(self, x, y):
        column = int(x // CELL_SIZE)
        row = int(y // CELL_SIZE)
        if not (0 <= column < MAZE_COLS and 0 <= row < MAZE_ROWS):
            return True

        walls = self.maze[row][column]
        x_in_cell = x - column * CELL_SIZE
        y_in_cell = y - row * CELL_SIZE

        if y_in_cell < WALL_COLLISION_MARGIN and walls[TOP]: return True
        if x_in_cell > CELL_SIZE - WALL_COLLISION_MARGIN and walls[RIGHT]: return True
        if y_in_cell > CELL_SIZE - WALL_COLLISION_MARGIN and walls[BOTTOM]: return True
        if x_in_cell < WALL_COLLISION_MARGIN and walls[LEFT]: return True
        return False

    def _update_turn(self, steering):
        steering = max(-1.0, min(1.0, float(steering)))
        self.angular_velocity = (self.angular_velocity * TURN_DRAG + steering * TURN_ACCELERATION)
        self.angular_velocity = max(-MAX_TURN_RATE, min(MAX_TURN_RATE, self.angular_velocity))
        self.angle += self.angular_velocity

    def _update_velocity(self):
        movement_angle = math.radians(self.angle)
        self.velocity_x = (self.velocity_x * MOVEMENT_DRAG + math.cos(movement_angle) * MOVEMENT_ACCELERATION)
        self.velocity_y = (self.velocity_y * MOVEMENT_DRAG + math.sin(movement_angle) * MOVEMENT_ACCELERATION)
        speed = math.hypot(self.velocity_x, self.velocity_y)
        if speed > MOVEMENT_SPEED:
            speed_scale = MOVEMENT_SPEED / speed
            self.velocity_x *= speed_scale
            self.velocity_y *= speed_scale

    def _move_player(self):
        next_x = self.x + self.velocity_x
        next_y = self.y + self.velocity_y
        collided = False

        if self._is_wall_at(next_x, self.y):
            self.velocity_x = -self.velocity_x * WALL_BOUNCE_FACTOR
            collided = True
        else:
            self.x = next_x

        if self._is_wall_at(self.x, next_y):
            self.velocity_y = -self.velocity_y * WALL_BOUNCE_FACTOR
            collided = True
        else:
            self.y = next_y

        return collided

    def _wall_risk(self):
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
        nearest_wall = WALL_PROXIMITY_RADIUS

        for direction_x, direction_y in directions:
            for distance in range(2, WALL_PROXIMITY_RADIUS + 1, 2):
                if self._is_wall_at(
                    self.x + direction_x * distance,
                    self.y + direction_y * distance,
                ):
                    nearest_wall = min(nearest_wall, distance)
                    break

        proximity = max(0.0, (WALL_PROXIMITY_RADIUS - nearest_wall) / WALL_PROXIMITY_RADIUS)
        return proximity * proximity

    def _get_obs(self):

        max_dim = max(MAZE_COLS, MAZE_ROWS) * CELL_SIZE
        angle_rad = math.radians(self.angle)
        cos_ang, sin_ang = math.cos(angle_rad), math.sin(angle_rad)
        
        sensor_offsets = [-1.57, -0.94, -0.31, 0.31, 0.94, 1.57]
        ray_results = []
        sensor_distance = 50
        
        for offset in sensor_offsets:
            check_angle = angle_rad + offset
            target_x = self.x + math.cos(check_angle) * sensor_distance
            target_y = self.y + math.sin(check_angle) * sensor_distance

            ray_results.append(1.0 if self._is_wall_at(target_x, target_y) else 0.0)

        base_features = [
            self.x / max_dim, 
            self.y / max_dim,
            (self.goal_rect.centerx - self.x) / max_dim,
            (self.goal_rect.centery - self.y) / max_dim,
            cos_ang, 
            sin_ang, 
            self.game.velocity_x if hasattr(self, 'game') else self.velocity_x, 
            self.game.velocity_y if hasattr(self, 'game') else self.velocity_y, 
            self.angular_velocity
        ]
        
        return np.array(base_features + ray_results, dtype=np.float32)

    def step(self, action):
        steering = float(action - 1) 
        
        old_dist = math.hypot(self.goal_rect.centerx - self.x, self.goal_rect.centery - self.y)
        old_wall_risk = self._wall_risk()
        
        self._update_turn(steering)
        self._update_velocity()
        hit_wall = self._move_player()
        self.steps += 1
        self.reached_goal = self.goal_rect.collidepoint(self.x, self.y)
        
        new_dist = math.hypot(self.goal_rect.centerx - self.x, self.goal_rect.centery - self.y)
        new_wall_risk = self._wall_risk()
        
        reward = (old_dist - new_dist) * GOAL_PROGRESS_REWARD_SCALE - STEP_PENALTY
        reward += WALL_PROXIMITY_REWARD_SCALE * (old_wall_risk - new_wall_risk)
        if hit_wall:
            reward -= WALL_COLLISION_PENALTY
        if self.reached_goal:
            reward += GOAL_REWARD
            
        done = self.reached_goal
        truncated = self.steps >= EPISODE_LIMIT
        
        return self._get_obs(), reward, done, truncated, {}

    def render(self):
        if not self.visualize:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.fill((20, 20, 20))
        self._draw_maze()
        self._draw_player()
        pygame.display.flip()
        self.clock.tick(60)

    def _draw_maze(self):
        wall_color = (200, 200, 200)
        for row in range(MAZE_ROWS):
            for column in range(MAZE_COLS):
                x = column * CELL_SIZE
                y = row * CELL_SIZE
                walls = self.maze[row][column]

                if walls[TOP]:
                    pygame.draw.line(
                        self.screen, wall_color, (x, y), (x + CELL_SIZE, y), 2
                    )
                if walls[RIGHT]:
                    pygame.draw.line(
                        self.screen,
                        wall_color,
                        (x + CELL_SIZE, y),
                        (x + CELL_SIZE, y + CELL_SIZE),
                        2,
                    )
                if walls[BOTTOM]:
                    pygame.draw.line(
                        self.screen,
                        wall_color,
                        (x, y + CELL_SIZE),
                        (x + CELL_SIZE, y + CELL_SIZE),
                        2,
                    )
                if walls[LEFT]:
                    pygame.draw.line(
                        self.screen, wall_color, (x, y), (x, y + CELL_SIZE), 2
                    )

        pygame.draw.rect(self.screen, (0, 150, 255), self.goal_rect)
                    
    def _draw_player(self):
        facing_angle = math.radians(self.angle)
        front_point = (
            self.x + math.cos(facing_angle) * PLAYER_SIZE,
            self.y + math.sin(facing_angle) * PLAYER_SIZE,
        )
        left_point = (
            self.x + math.cos(facing_angle + 2.5) * PLAYER_SIZE,
            self.y + math.sin(facing_angle + 2.5) * PLAYER_SIZE,
        )
        right_point = (
            self.x + math.cos(facing_angle - 2.5) * PLAYER_SIZE,
            self.y + math.sin(facing_angle - 2.5) * PLAYER_SIZE,
        )
        pygame.draw.polygon(
            self.screen, (255, 200, 0), [front_point, left_point, right_point]
        )