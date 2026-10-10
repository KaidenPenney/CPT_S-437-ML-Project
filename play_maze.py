from controllers import KeyboardControls
from maze_game import MazeGame


def play_human():
    game = MazeGame(visualize=True)
    controls = KeyboardControls()
    game_over = False

    while not game_over:
        steering = controls.get_steering()
        game_over = game.update(steering)
        game.render()


if __name__ == "__main__":
    play_human()
