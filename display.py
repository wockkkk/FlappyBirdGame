import pygame
import torch
from flappy_env import FlappyBirdEnv
from dqn128_64_32 import EnhancedDQN as DQN128
from qdn256_128_64 import EnhancedDQN as DQN256
from qdn256_128_64_32 import EnhancedDQN as DQN256_2

mod = 256
path = '256_model812round_34score.pth'
class AIPlayer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 600))
        self.clock = pygame.time.Clock()
        self.env = FlappyBirdEnv()
        self.env.seed(42)
        self.score = []
        if mod == 128:
            self.model = DQN128(self.env.observation_space.shape[0],
                                   self.env.action_space.n)
        elif mod == 256:
            self.model = DQN256(self.env.observation_space.shape[0],
                                   self.env.action_space.n)
        elif mod == 512:
            self.model = DQN256_2(self.env.observation_space.shape[0],
                                   self.env.action_space.n)
        self.model.load_state_dict(torch.load(path))
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

            next_state, _, done, _, _ = self.env.step(action)
            state = next_state
            # self.env.render(self.screen)
            # self.clock.tick(60)
            if done:
                if len(self.score) < 10:
                    print(self.env.score)
                    self.score.append(self.env.score)
                    state = self.env.reset()
                else:
                    n = 0
                    for i in self.score:
                        n += i
                    print(n / len(self.score))
                    break




if __name__ == "__main__":
    player = AIPlayer()
    player.play()