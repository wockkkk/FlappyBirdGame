import pygame
import torch
from flappy_env import FlappyBirdEnv
from dqn import DQNetwork


class AIPlayer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 600))
        self.clock = pygame.time.Clock()
        self.env = FlappyBirdEnv()
        self.env.seed(42)
        self.model = DQNetwork(self.env.observation_space.shape[0],
                               self.env.action_space.n)
        self.model.load_state_dict(torch.load('flappy_best_model.pth'))
        self.model.eval()

    def play(self):
        state = self.env.reset()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(state))
            action = q_values.argmax().item()

            next_state, _, done = self.env.step(action)
            self.env.render(self.screen)
            state = next_state

            if done:
                state = self.env.reset()

            self.clock.tick(30)  # 控制游戏速度


if __name__ == "__main__":
    player = AIPlayer()
    player.play()