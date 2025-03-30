import gym
import numpy as np

from game import *  # 导入原始游戏代码（需调整为可调用形式）

def columns_generator(self):
    gap_y = random.randint(200, 400)
    top = self.column_img.get_rect(midbottom=(400, gap_y - self.size))
    bottom = self.column_img.get_rect(midtop=(400, gap_y + self.size))
    if (gap_y - self.size) < 50 or (gap_y + self.size) > 550:
        return self.columns_generator()
    self.columns.append([top, bottom])

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        # 定义动作空间（0: 不跳，1: 跳）
        self.action_space = gym.spaces.Discrete(2)

        # 定义状态空间（bird_y, bird_speed, top, bottom, distance_to_next_column）
        self.observation_space = gym.spaces.Box(
            low=np.array([0, -10, 50, 350, 0]),
            high=np.array([600, 10, 360, 550, 400]),
            dtype=np.float32
        )

        # 初始化游戏参数
        self.bird_rect = bird_rect
        self.bird_speed = bird_speed
        self.columns = columns
        self.size = size
        self.score_add = 0
        self.score = 0
        self.game_over = False
        self.gravity = gravity
        self.screen = screen
        self.column_img = column_img

    def seed(self,seed):
        random.seed(seed)

    def step(self, action):
        # 执行动作
        if action == 1:
            self.bird_speed = -3  # 跳跃

        # 更新游戏状态
        self.bird_speed += self.gravity
        self.bird_rect.y += self.bird_speed

        if len(self.columns) < 2:  # 保持场景中始终有2组柱子
            columns_generator(self)

        # 柱子移动
        for pair in self.columns:
            pair[0].x -= 2
            pair[1].x -= 2

        # 碰撞检测
        for column_pair in self.columns:
            if self.bird_rect.colliderect(column_pair[0]) or self.bird_rect.colliderect(column_pair[1]):
                self.game_over = True
        for pair in self.columns:
            if pair[0].x <= 75:
                self.columns.remove(pair)
                self.score_add = 1
                self.score += 1
                self.size = 150 - self.score * 3
            else:
                self.score_add = 0



        nearest = min(self.columns, key=lambda p: p[0].x if p[0].x > 50 else float('inf'))
        left = self.bird_rect.right - nearest[0].left
        if left < 0:
            left = 0
        # 修改reward计算部分
        reward = (
                15.0 * self.score_add
                + 0.1 * (1 - self.game_over)
                - 25.0 * self.game_over
                - 0.05 * abs(self.bird_speed)  # 抑制过度跳跃
        )
        gap_center = (nearest[0].bottom + nearest[1].top) / 2
        position_reward = 0.3 * (1 - abs(gap_center - self.bird_rect.centery) / 300)
        reward += position_reward   # 位置引导奖励

        # 构造状态向量
        state = self._get_state()

        # 判断是否终止
        done = self.game_over or self.bird_rect.top <= 0 or self.bird_rect.bottom >= 600

        return state, reward, done

    def reset(self, **kwargs):
        # 重置游戏状态
        self.bird_rect.center = (100, 300)
        self.bird_speed = 0
        self.columns = []
        columns_generator(self)
        self.score_add = 0
        self.score = 0
        self.size = 150
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
        nearest = min(self.columns, key=lambda p: p[0].x if p[0].x > 50 else float('inf'))
        gap_center = (nearest[0].bottom + nearest[1].top) / 2
        horizontal_dist = nearest[0].x - self.bird_rect.centerx

        next_column = self.columns[1] if len(self.columns) > 1 else self.columns[0]
        return np.array([
            self.bird_rect.centery / 600,
            (self.bird_speed + 10) / 20,
            horizontal_dist / 400,
            (next_column[0].x - self.bird_rect.x) / 400,  # 下一组柱子距离
            (self.size - 40) / 110
        ], dtype=np.float32)

