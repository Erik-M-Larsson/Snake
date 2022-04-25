import torch
import random
import numpy as np
from game import BLOCK_SIZE, SnakeGameAI, Direction, Point, Game_board_size
from collections import deque
from model import Linear_QNet, QTrainer
from helper import *

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

ROWS = 4  # 24  # even number
COLONS = 4  # 32  # even number


class Agent:
    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate <1
        self.memory = deque(maxlen=MAX_MEMORY)
        input_size = (ROWS + 2) * (COLONS + 2) + 2 * ROWS * COLONS
        hidden_size = 2
        while hidden_size < input_size:
            hidden_size *= 2

        print(f"{input_size=}, {hidden_size=}")
        self.model = Linear_QNet(input_size, hidden_size, hidden_size, 3)

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):

        # Blue
        state_blue = np.zeros((ROWS, COLONS), dtype=np.int16)

        for tail in game.snake[1:]:
            state_blue[
                int(tail.y / BLOCK_SIZE),
                int(tail.x / BLOCK_SIZE),
            ] = 1

        # Green
        state_green = np.zeros((ROWS + 2, COLONS + 2), dtype=np.int16)

        state_green[
            int(game.snake[0].y / BLOCK_SIZE + 1), int(game.snake[0].x / BLOCK_SIZE + 1)
        ] = 1

        # Red
        state_red = np.zeros((ROWS, COLONS), dtype=np.int16)

        state_red[int(game.food.y / BLOCK_SIZE), int(game.food.x / BLOCK_SIZE)] = 1

        return np.concatenate(
            [state_blue.flatten(), state_green.flatten(), state_red.flatten()]
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploition
        self.epsilon = 8000 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 10000) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []

    record = 0
    idx_img_save = 0
    show_every = 50
    save_every = 200

    agent = Agent()
    game = SnakeGameAI(COLONS, ROWS)

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train the long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score

            plot_scores.append(score)

            mean_score = sum(plot_scores[-100:]) / len(plot_scores[-100:])
            plot_mean_scores.append(mean_score)

            if game.show_display or agent.n_games % show_every == 0:
                print("Game:", agent.n_games, "Score:", score, "Record:", record)

                if agent.n_games % save_every == 0:
                    agent.model.save()
                    save_plot = True
                    idx_img_save += save_every
                else:
                    save_plot = False

                plot(plot_scores, plot_mean_scores, save_plot, idx_img_save)


if __name__ == "__main__":
    train()
