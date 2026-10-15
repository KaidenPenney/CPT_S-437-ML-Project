import math
import os
import random
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from config import (
    ACTION_REPEAT,
    CELL_SIZE,
    EPISODE_LIMIT,
    EVAL_SEED_OFFSET,
    GOAL_MARGIN,
    GOAL_PROGRESS_REWARD,
    GOAL_REWARD,
    MAX_TURN_RATE,
    MAZE_COLS,
    MAZE_ROWS,
    MOVEMENT_ACCELERATION,
    MOVEMENT_DRAG,
    MOVEMENT_SPEED,
    NUM_EVAL_MAZES,
    OBS_INCLUDE_POSITION,
    PLAYER_SIZE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SENSOR_ANGLES,
    SENSOR_RANGE,
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
GOAL_CELL = (MAZE_COLS - 1, MAZE_ROWS - 1)


def generate_maze(rng):
    """Randomized depth-first search maze. maze[row][col] = [top, right, bottom, left] walls."""
    maze = [[[True, True, True, True] for _ in range(MAZE_COLS)] for _ in range(MAZE_ROWS)]
    visited = [[False for _ in range(MAZE_COLS)] for _ in range(MAZE_ROWS)]

    def carve_passages(column, row):
        visited[row][column] = True
        directions = list(WALL_DIRECTIONS)
        rng.shuffle(directions)

        for delta_column, delta_row, wall, opposite_wall in directions:
            next_column = column + delta_column
            next_row = row + delta_row
            is_inside_maze = (
                0 <= next_column < MAZE_COLS
                and 0 <= next_row < MAZE_ROWS
            )

            if is_inside_maze and not visited[next_row][next_column]:
                maze[row][column][wall] = False
                maze[next_row][next_column][opposite_wall] = False
                carve_passages(next_column, next_row)

    carve_passages(0, 0)
    return maze


def maze_key(maze):
    return tuple(tuple(tuple(cell) for cell in row) for row in maze)


def eval_maze_seeds():
    """
    Seeds of the held-out evaluation mazes. Different seeds can produce the same
    layout on a small grid, so duplicates are skipped to get NUM_EVAL_MAZES distinct mazes.
    """
    seeds, keys = [], set()
    seed = EVAL_SEED_OFFSET
    while len(seeds) < NUM_EVAL_MAZES:
        key = maze_key(generate_maze(random.Random(seed)))
        if key not in keys:
            keys.add(key)
            seeds.append(seed)
        seed += 1
    return seeds


def eval_maze_keys():
    """Layouts of the held-out evaluation mazes, so training can avoid them."""
    return {maze_key(generate_maze(random.Random(seed))) for seed in eval_maze_seeds()}


def observation_size():
    return len(SENSOR_ANGLES) + 6 + (4 if OBS_INCLUDE_POSITION else 0)


class MazeGame(gym.Env):
    """
    Actions: 0 = steer left, 1 = straight, 2 = steer right.

    A new maze is generated on every reset(). Passing reset(seed=s) generates the
    maze from random.Random(s), which is how the fixed evaluation mazes are made.
    Without a seed, the maze comes from the environment's own random stream.
    """

    def __init__(self, visualize=False, seed=None, action_repeat=ACTION_REPEAT,
                 excluded_mazes=None):
        super().__init__()
        self.visualize = visualize
        self.random = random.Random(seed)
        self.action_repeat = action_repeat
        self.excluded_mazes = excluded_mazes or set()

        if self.visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

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
            low=-np.inf, high=np.inf, shape=(observation_size(),), dtype=np.float32
        )

        self.maze = None

    def _new_maze(self, seed):
        if seed is not None:
            return generate_maze(random.Random(seed))
        while True:
            maze = generate_maze(self.random)
            if maze_key(maze) not in self.excluded_mazes:
                return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.maze = self._new_maze(seed)
        self._compute_goal_distances()

        self.x = self.start_x
        self.y = self.start_y
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.steps = 0
        self.reached_goal = False
        self.collided = False
        self.distance_traveled = 0.0
        self.collision_ticks = 0
        self.visited_cells = {self._cell_of(self.x, self.y)}

        return self._get_obs(), {}

    def _cell_of(self, x, y):
        column = min(MAZE_COLS - 1, max(0, int(x // CELL_SIZE)))
        row = min(MAZE_ROWS - 1, max(0, int(y // CELL_SIZE)))
        return column, row

    def _compute_goal_distances(self):
        """
        For every cell, find the point where the shortest path leaves it (the middle
        of the doorway toward the goal) and the path length from that point to the
        goal. Used for geodesic progress shaping.
        """
        goal_center = (self.goal_rect.centerx, self.goal_rect.centery)
        self.exit_point = {GOAL_CELL: goal_center}
        self.exit_distance = {GOAL_CELL: 0.0}
        queue = deque([GOAL_CELL])
        while queue:
            column, row = queue.popleft()
            walls = self.maze[row][column]
            for delta_column, delta_row, wall, _ in WALL_DIRECTIONS:
                neighbor = (column + delta_column, row + delta_row)
                if walls[wall] or neighbor in self.exit_point:
                    continue
                # Doorway between neighbor and this cell.
                door = (
                    (column + 0.5 - 0.5 * delta_column) * CELL_SIZE,
                    (row + 0.5 - 0.5 * delta_row) * CELL_SIZE,
                )
                self.exit_point[neighbor] = door
                self.exit_distance[neighbor] = self.exit_distance[(column, row)] + math.dist(
                    door, self.exit_point[(column, row)]
                )
                queue.append(neighbor)

    def goal_distance(self):
        """Shortest-path distance (pixels) from the player to the goal center."""
        cell = self._cell_of(self.x, self.y)
        return math.dist((self.x, self.y), self.exit_point[cell]) + self.exit_distance[cell]

    def _is_wall_at(self, x, y):
        column = int(x // CELL_SIZE)
        row = int(y // CELL_SIZE)
        if not (0 <= column < MAZE_COLS and 0 <= row < MAZE_ROWS):
            return True

        walls = self.maze[row][column]
        x_in_cell = x - column * CELL_SIZE
        y_in_cell = y - row * CELL_SIZE

        if y_in_cell < WALL_COLLISION_MARGIN and walls[TOP]:
            return True
        if x_in_cell > CELL_SIZE - WALL_COLLISION_MARGIN and walls[RIGHT]:
            return True
        if y_in_cell > CELL_SIZE - WALL_COLLISION_MARGIN and walls[BOTTOM]:
            return True
        if x_in_cell < WALL_COLLISION_MARGIN and walls[LEFT]:
            return True
        return False

    def _cast_ray(self, angle, max_distance):
        """Distance along a ray to the first wall, walking cell boundaries (DDA)."""
        direction_x, direction_y = math.cos(angle), math.sin(angle)
        column, row = self._cell_of(self.x, self.y)
        step_column = 1 if direction_x > 0 else -1
        step_row = 1 if direction_y > 0 else -1

        while True:
            if direction_x != 0:
                boundary_x = (column + 1) * CELL_SIZE if direction_x > 0 else column * CELL_SIZE
                distance_x = (boundary_x - self.x) / direction_x
            else:
                distance_x = math.inf
            if direction_y != 0:
                boundary_y = (row + 1) * CELL_SIZE if direction_y > 0 else row * CELL_SIZE
                distance_y = (boundary_y - self.y) / direction_y
            else:
                distance_y = math.inf

            walls = self.maze[row][column]
            if distance_x < distance_y:
                if distance_x >= max_distance:
                    return max_distance
                if walls[RIGHT if step_column > 0 else LEFT]:
                    return max(0.0, distance_x)
                column += step_column
            else:
                if distance_y >= max_distance:
                    return max_distance
                if walls[BOTTOM if step_row > 0 else TOP]:
                    return max(0.0, distance_y)
                row += step_row

    def _update_turn(self, steering):
        steering = max(-1.0, min(1.0, float(steering)))
        self.angular_velocity = (
            self.angular_velocity * TURN_DRAG + steering * TURN_ACCELERATION
        )
        self.angular_velocity = max(
            -MAX_TURN_RATE,
            min(MAX_TURN_RATE, self.angular_velocity),
        )
        self.angle = (self.angle + self.angular_velocity) % 360.0

    def _update_velocity(self):
        movement_angle = math.radians(self.angle)
        self.velocity_x = (
            self.velocity_x * MOVEMENT_DRAG
            + math.cos(movement_angle) * MOVEMENT_ACCELERATION
        )
        self.velocity_y = (
            self.velocity_y * MOVEMENT_DRAG
            + math.sin(movement_angle) * MOVEMENT_ACCELERATION
        )
        speed = math.hypot(self.velocity_x, self.velocity_y)
        if speed > MOVEMENT_SPEED:
            speed_scale = MOVEMENT_SPEED / speed
            self.velocity_x *= speed_scale
            self.velocity_y *= speed_scale

    def _move_player(self):
        start_x, start_y = self.x, self.y
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

        self.distance_traveled += math.hypot(self.x - start_x, self.y - start_y)
        return collided

    def _wall_risk(self):
        nearest_wall = min(
            self._cast_ray(angle, WALL_PROXIMITY_RADIUS)
            for angle in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
        )
        proximity = (WALL_PROXIMITY_RADIUS - nearest_wall) / WALL_PROXIMITY_RADIUS
        return proximity * proximity

    def _get_obs(self):
        angle_rad = math.radians(self.angle)
        cos_ang, sin_ang = math.cos(angle_rad), math.sin(angle_rad)

        ray_results = [
            self._cast_ray(angle_rad + math.radians(offset), SENSOR_RANGE) / SENSOR_RANGE
            for offset in SENSOR_ANGLES
        ]

        # Velocity in the player's own frame (forward, sideways), like an odometer.
        forward_speed = self.velocity_x * cos_ang + self.velocity_y * sin_ang
        side_speed = -self.velocity_x * sin_ang + self.velocity_y * cos_ang

        features = ray_results + [
            forward_speed / MOVEMENT_SPEED,
            side_speed / MOVEMENT_SPEED,
            self.angular_velocity / MAX_TURN_RATE,
            cos_ang,
            sin_ang,
            1.0 if self.collided else 0.0,
        ]

        if OBS_INCLUDE_POSITION:
            max_dim = max(MAZE_COLS, MAZE_ROWS) * CELL_SIZE
            features += [
                self.x / max_dim,
                self.y / max_dim,
                (self.goal_rect.centerx - self.x) / max_dim,
                (self.goal_rect.centery - self.y) / max_dim,
            ]

        return np.array(features, dtype=np.float32)

    def step(self, action):
        steering = float(action - 1)

        old_dist = self.goal_distance()
        old_wall_risk = self._wall_risk()

        collision_ticks = 0
        for _ in range(self.action_repeat):
            self._update_turn(steering)
            self._update_velocity()
            collision_ticks += self._move_player()
            self.steps += 1
            self.reached_goal = self.goal_rect.collidepoint(self.x, self.y)
            if self.visualize:
                self.render()
            if self.reached_goal or self.steps >= EPISODE_LIMIT:
                break

        self.collided = collision_ticks > 0
        self.collision_ticks += collision_ticks
        self.visited_cells.add(self._cell_of(self.x, self.y))

        new_dist = self.goal_distance()
        new_wall_risk = self._wall_risk()

        reward = (old_dist - new_dist) / CELL_SIZE * GOAL_PROGRESS_REWARD - STEP_PENALTY
        reward += WALL_PROXIMITY_REWARD_SCALE * (old_wall_risk - new_wall_risk)
        reward -= WALL_COLLISION_PENALTY * collision_ticks
        if self.reached_goal:
            reward += GOAL_REWARD

        done = self.reached_goal
        truncated = not done and self.steps >= EPISODE_LIMIT
        info = {}
        if done or truncated:
            info["episode_stats"] = {
                "success": float(self.reached_goal),
                "ticks": self.steps,
                "distance_traveled": self.distance_traveled,
                "collision_ticks": self.collision_ticks,
                "cells_visited": len(self.visited_cells),
            }

        return self._get_obs(), reward, done, truncated, info

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
