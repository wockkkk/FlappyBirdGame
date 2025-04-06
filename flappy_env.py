import gym
import numpy as np

from game import *  # 导入原始游戏代码

def columns_generator(self):
    gap_y = random.randint(200, 400)
    top = self.column_img.get_rect(midbottom=(400, gap_y - self.size))
    bottom = self.column_img.get_rect(midtop=(400, gap_y + self.size))
    if (gap_y - self.size) < 50 or (gap_y + self.size) > 550:
        return columns_generator(self)
    self.columns.append([top, bottom])

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        # 定义动作空间（0: 不跳，1: 跳）
        self.action_space = gym.spaces.Discrete(2)

        # 定义状态空间（bird_y, bird_speed, top, bottom, distance_to_next_column, next_top, next_bottom）
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float64
        )

        # 初始化游戏参数
        self.bird_rect = bird_rect
        self.bird_speed = bird_speed
        self.columns = []
        self.size = size
        self.score_add = 0
        self.score = 0
        self.game_over = False
        self.gravity = gravity
        self.screen = screen
        self.column_img = column_img
        self.clock = 0
        columns_generator(self)

    def seed(self,seed):
        random.seed(seed)

    def step(self, action):
        self.clock += 1
        # 执行动作
        if action == 1:
            self.bird_speed = -3  # 跳跃

        # 更新游戏状态
        self.bird_speed += self.gravity
        self.bird_rect.y += self.bird_speed

        if self.clock % 90 == 0:
            columns_generator(self)

        # 柱子移动
        for pair in self.columns:
            pair[0].x -= 2
            pair[1].x -= 2

        # 碰撞检测
        for column_pair in self.columns:
            if self.bird_rect.colliderect(column_pair[0]) or self.bird_rect.colliderect(column_pair[1]):
                self.game_over = True
        self.score_add = 0
        for pair in self.columns.copy():
            if pair[0].right <= self.bird_rect.left:
                self.columns.remove(pair)
                self.score_add = 1
                self.score += 1
                self.size = 150 - self.score * 3
        done = self.game_over or self.bird_rect.top <= 0 or self.bird_rect.bottom >= 600
        reward = (
                25.0 * self.score_add  # 成功通过
                + 0.02
                - 20.0 * done  # 失败惩罚
        )

        # 构造状态向量
        state = self._get_state()

        # 判断是否终止


        return state, reward, done, False, {}

    def reset(self, **kwargs):
        # 重置游戏状态
        self.bird_rect.center = (100, 300)
        self.bird_speed = 0
        self.size = 150
        self.columns = []
        columns_generator(self)
        self.score_add = 0
        self.clock = 0
        self.score = 0
        self.game_over = False
        return self._get_state()

    # noinspection PyMethodOverriding
    def render(self, screen):
        # 渲染游戏画面
        screen.fill((148, 222, 249))
        screen.blit(bird_img, self.bird_rect)
        for column_pair in self.columns:
            screen.blit(self.column_img, column_pair[0])
            screen.blit(self.column_img, column_pair[1])
        pygame.display.update()

    # 修改_get_state方法：
    def _get_state(self):
        nearest = sorted(self.columns, key=lambda p: p[0].x)
        gap_center = (nearest[0][0].bottom + nearest[0][1].top) / 2
        horizontal_dist = nearest[0][0].centerx - self.bird_rect.centerx
        try:
            next_column = nearest[1]
        except IndexError:
            next_column = nearest[0]
        return np.array([
            self.bird_rect.centery / 600,
            (self.bird_speed + 10) / 20,
            nearest[0][0].top / 600,
            nearest[0][1].bottom / 600,
            horizontal_dist / 400,
            next_column[0].top / 600,
            next_column[1].bottom / 600,
        ], dtype=np.float32)

