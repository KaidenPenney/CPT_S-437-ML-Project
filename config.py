SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
CELL_SIZE = 200
GOAL_MARGIN = 5
WALL_COLLISION_MARGIN = 2
PLAYER_SIZE = 12

MAZE_COLS = SCREEN_WIDTH // CELL_SIZE
MAZE_ROWS = SCREEN_HEIGHT // CELL_SIZE

MOVEMENT_SPEED = 2.0
MOVEMENT_ACCELERATION = 0.15
MOVEMENT_DRAG = 0.97
WALL_BOUNCE_FACTOR = 0.4
TURN_ACCELERATION = 0.9
TURN_DRAG = 0.9
MAX_TURN_RATE = 5.0

# The agent picks a new action every ACTION_REPEAT physics ticks. EPISODE_LIMIT is
# in physics ticks: at 2 px/tick and 200 px cells, 6000 ticks is enough to walk
# roughly 60 cells, which leaves room to explore a 6x4 maze.
ACTION_REPEAT = 4
EPISODE_LIMIT = 6000

# Distance sensors: rays cast at these angles (degrees, relative to heading),
# each reporting distance to the nearest wall, normalized to [0, 1].
SENSOR_ANGLES = (-90, -60, -30, 0, 30, 60, 90)
SENSOR_RANGE = 200
# If True, the observation also contains absolute position and the vector to the
# goal. That information is not available from on-board sensors, so it is off by
# default to keep the task partially observable as described in the proposal.
OBS_INCLUDE_POSITION = False

# Rewards (per agent decision unless noted).
# Progress is measured along the maze's shortest path (geodesic distance), not in
# a straight line, so that dead ends close to the goal are not rewarded.
# The shaping is potential-based, so it does not change the optimal policy.
GOAL_PROGRESS_REWARD = 0.5  # per cell of shortest-path progress
GOAL_REWARD = 10.0
STEP_PENALTY = 0.01
WALL_COLLISION_PENALTY = 0.1  # per physics tick spent colliding
WALL_PROXIMITY_RADIUS = 40
WALL_PROXIMITY_REWARD_SCALE = 0.5

# Held-out evaluation mazes are generated from seeds EVAL_SEED_OFFSET + i.
# Training environments never generate any of these layouts.
EVAL_SEED_OFFSET = 1_000_000
NUM_EVAL_MAZES = 200
