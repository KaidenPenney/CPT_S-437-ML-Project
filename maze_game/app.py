"""Pygame front end for playing the maze by hand."""

from __future__ import annotations

import argparse

import pygame

from .game import DOWN, LEFT, RIGHT, UP, MazeGame
from .maze import DIRECTIONS

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
HUD_HEIGHT = 32
MAX_MAZE_PIXELS = (900, 640)  # the default tile size is the largest that fits in this
MOVE_DELAY_MS = 180  # how long a movement key is held before it starts repeating
MOVE_INTERVAL_MS = 50  # time between repeated moves while a key is held

KEY_TO_ACTION = {
    pygame.K_UP: UP,
    pygame.K_w: UP,
    pygame.K_RIGHT: RIGHT,
    pygame.K_d: RIGHT,
    pygame.K_DOWN: DOWN,
    pygame.K_s: DOWN,
    pygame.K_LEFT: LEFT,
    pygame.K_a: LEFT,
}
RESTART_KEYS = {pygame.K_r, pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE}


class MazeApp:
    def __init__(self, game: MazeGame, tile: int | None = None):
        self.game = game
        grid_h, grid_w = game.grid.shape
        self.tile = tile or max(
            4, min(MAX_MAZE_PIXELS[0] // grid_w, MAX_MAZE_PIXELS[1] // grid_h)
        )
        self.screen = pygame.display.set_mode(
            (grid_w * self.tile, grid_h * self.tile + HUD_HEIGHT)
        )
        self.font = pygame.font.Font(None, 26)
        self.last_time: float | None = None
        self.best_time: float | None = None
        self.last_shortest = 0
        self.solved_count = 0
        self._held_key: int | None = None
        self._repeat_at = 0
        self._render_maze()

    def new_maze(self) -> None:
        self.game.new_maze()
        self._render_maze()

    def _render_maze(self) -> None:
        """Draw the walls once per maze; each frame just blits this surface."""
        t = self.tile
        grid_h, grid_w = self.game.grid.shape
        self.maze_surface = pygame.Surface((grid_w * t, grid_h * t))
        self.maze_surface.fill(WHITE)
        for r, c in zip(*self.game.grid.nonzero()):
            self.maze_surface.fill(BLACK, (c * t, r * t, t, t))
        pygame.display.set_caption(
            f"Maze (seed {self.game.maze_seed})  -  arrows/WASD move, R new maze, Esc quit"
        )

    def handle_key(self, key: int) -> None:
        if key == pygame.K_r or (self.game.solved and key in RESTART_KEYS):
            self.new_maze()
        elif key in KEY_TO_ACTION:
            self._move(KEY_TO_ACTION[key])
            self._held_key = key
            self._repeat_at = pygame.time.get_ticks() + MOVE_DELAY_MS

    def update(self) -> None:
        """Keep moving while a movement key is held down."""
        if self._held_key is None:
            return
        if not pygame.key.get_pressed()[self._held_key]:
            self._held_key = None
            return
        now = pygame.time.get_ticks()
        if now >= self._repeat_at:
            self._move(KEY_TO_ACTION[self._held_key])
            self._repeat_at = now + MOVE_INTERVAL_MS

    def _move(self, action: int) -> None:
        if self.game.step(action) and self.game.solved:
            self.last_time = self.game.elapsed
            if self.best_time is None or self.last_time < self.best_time:
                self.best_time = self.last_time
            self.last_shortest = self.game.shortest_path_length()
            self.solved_count += 1

    def draw(self) -> None:
        self.screen.fill(WHITE)
        self.screen.blit(self.maze_surface, (0, HUD_HEIGHT))
        self._draw_tile_square(self.game.goal, GRAY)
        self._draw_agent()
        self._draw_hud()
        if self.game.solved:
            self._draw_solved_banner()

    def _draw_tile_square(self, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
        t = self.tile
        self.screen.fill(color, (pos[1] * t, pos[0] * t + HUD_HEIGHT, t, t))

    def _draw_agent(self) -> None:
        """A triangle pointing in the direction the agent last tried to move."""
        t = self.tile
        r, c = self.game.pos
        cx, cy = (c + 0.5) * t, (r + 0.5) * t + HUD_HEIGHT
        dr, dc = DIRECTIONS[self.game.heading]
        fx, fy = dc, dr  # forward, in screen x/y
        px, py = -fy, fx  # perpendicular to forward
        tip = (cx + fx * 0.45 * t, cy + fy * 0.45 * t)
        left = (cx - (fx - px) * 0.4 * t, cy - (fy - py) * 0.4 * t)
        right = (cx - (fx + px) * 0.4 * t, cy - (fy + py) * 0.4 * t)
        pygame.draw.polygon(self.screen, BLACK, (tip, left, right))

    def _draw_hud(self) -> None:
        def fmt(seconds: float | None) -> str:
            return "--" if seconds is None else f"{seconds:.2f}s"

        left = f"Time {self.game.elapsed:.2f}s    Moves {self.game.moves}"
        right = (
            f"Last {fmt(self.last_time)}    Best {fmt(self.best_time)}"
            f"    Solved {self.solved_count}"
        )
        y = (HUD_HEIGHT - self.font.get_height()) // 2
        self.screen.blit(self.font.render(left, True, BLACK), (8, y))
        right_text = self.font.render(right, True, BLACK)
        self.screen.blit(right_text, (self.screen.get_width() - right_text.get_width() - 8, y))

    def _draw_solved_banner(self) -> None:
        lines = [
            f"Solved in {self.last_time:.2f}s  ({self.game.moves} moves, shortest {self.last_shortest})",
            "Press R, Enter or Space for a new maze",
        ]
        texts = [self.font.render(line, True, BLACK) for line in lines]
        pad = 12
        width = max(text.get_width() for text in texts) + 2 * pad
        height = sum(text.get_height() for text in texts) + 3 * pad
        box = pygame.Rect(0, 0, width, height)
        box.center = self.screen.get_rect().center
        self.screen.fill(WHITE, box)
        pygame.draw.rect(self.screen, BLACK, box, 2)
        y = box.top + pad
        for text in texts:
            self.screen.blit(text, (box.centerx - text.get_width() // 2, y))
            y += text.get_height() + pad

    def run(self) -> None:
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    self.handle_key(event.key)
            self.update()
            self.draw()
            pygame.display.flip()
            clock.tick(60)


def _int_at_least(minimum: int):
    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return value

    return parse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play a randomly generated maze.")
    parser.add_argument("--rows", type=_int_at_least(2), default=30, help="maze height in cells (default 30)")
    parser.add_argument("--cols", type=_int_at_least(2), default=40, help="maze width in cells (default 40)")
    parser.add_argument("--seed", type=int, default=None, help="seed for a reproducible sequence of mazes")
    parser.add_argument("--tile", type=_int_at_least(1), default=None, help="tile size in pixels (default: fit the screen)")
    args = parser.parse_args(argv)

    pygame.init()
    try:
        MazeApp(MazeGame(args.rows, args.cols, args.seed), args.tile).run()
    finally:
        pygame.quit()
