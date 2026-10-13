from controllers import KeyboardControls
from maze_game import MazeGame


def play_human():

    game = MazeGame(visualize=True)
    controls = KeyboardControls()

    game.reset()
    game_over = False

    while not game_over:

        raw_steering = controls.get_steering()

        action_idx = int(raw_steering + 1)
        action_idx = max(0, min(2, action_idx))

        _, _, terminated, truncated, _ = game.step(action_idx)
        game_over = terminated or truncated
        
        game.render()


if __name__ == "__main__":
    play_human()
