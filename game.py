import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
# font =pygame.font.Font("arial.ttf", 25)
font = pygame.font.SysFont("arial", 25)


class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3


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
BLOCK_SIZE = 80
SPEED = 4


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
        if self.show_display:
            self._update_ui()
            pygame.time.wait(300)
            self.clock.tick(SPEED)

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            # print("Apple cannot be place inside snake")
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Quiting game")
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
            reward = -10
            if self.show_display:
                self.snake.pop()
                self._update_ui()
                pygame.time.wait(400)
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10

            if len(self.snake) == self.w * self.h / BLOCK_SIZE**2:
                if self.show_display:
                    print(f"Reached max length {len(self.snake)}")
                    self._update_ui()
                    pygame.time.wait(500)

                game_over = True
                return reward, game_over, self.score

            self._place_food()
        else:
            self.snake.pop()
            # if action == [1, 0, 0]:
            #    reward = -1
            # else:
            #    reward = -2t
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

        # draw apple
        self._draw_block(self.food.x, self.food.y, BLOCK_SIZE, BLACK, RED)

        # draw tail
        for pt in self.snake[1:]:
            self._draw_block(
                pt.x,
                pt.y,
                BLOCK_SIZE,
                BLUE1,
                BLUE2,
            )

        # draw head
        self._draw_block(self.snake[0].x, self.snake[0].y, BLOCK_SIZE, GREEN1, GREEN2)

        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):

        self.direction = action
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT.value:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT.value:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN.value:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP.value:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _draw_block(self, x, y, w, color_outer, color_inner):
        pygame.draw.rect(
            self.display,
            color_outer,
            pygame.Rect(x, y, w, w),
            border_radius=4,
        )
        w_boarder = w // 5
        pygame.draw.rect(
            self.display,
            color_inner,
            pygame.Rect(
                x + w_boarder,
                y + w_boarder,
                w - 2 * w_boarder,
                w - 2 * w_boarder,
            ),
            border_radius=3,
        )
