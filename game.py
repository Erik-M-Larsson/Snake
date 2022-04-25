import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
# font =pygame.font.Font("arial.ttf", 25)
font = pygame.font.SysFont("arial", 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")
Game_board_size = namedtuple("Game_board_size", "rows, columns")


# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 100, 0)
GREEN2 = (0, 200, 200)
BLACK = (0, 0, 0)


size = Game_board_size(24, 32)
BLOCK_SIZE = 20
SPEED = 50


class SnakeGameAI:
    def __init__(self, w=size.columns, h=size.rows) -> None:
        self.w = w * BLOCK_SIZE
        self.h = h * BLOCK_SIZE

        # init display
        self.show_display = True
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snoken")
        self.clock = pygame.time.Clock()

        # init game state
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y),
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    self.show_display = not self.show_display
                    print(f"{self.show_display=}")

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -40
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 100
            # TODO bryt om ormen fyller spelplanen  
            self._place_food()
        else:
            self.snake.pop()
            # if action == [1, 0, 0]:
            #    reward = -1
            # else:
            #    reward = -2

        # 5. update ui and clock
        if self.show_display:
            self._update_ui()
            self.clock.tick(SPEED)
        else:
            self.clock.tick()

        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if (
            pt.x < 0
            or pt.x > self.w - BLOCK_SIZE
            or pt.y < 0
            or pt.y > self.h - BLOCK_SIZE
        ):
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # draw head
        pygame.draw.rect(
            self.display,
            GREEN1,
            pygame.Rect(self.snake[0].x, self.snake[0].y, BLOCK_SIZE, BLOCK_SIZE),
        )
        pygame.draw.rect(
            self.display,
            GREEN2,
            pygame.Rect(self.snake[0].x + 4, self.snake[0].y + 4, 12, 12),
        )

        # draw tail
        for pt in self.snake[1:]:
            pygame.draw.rect(
                self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )

        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
