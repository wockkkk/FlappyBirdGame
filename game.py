import pygame
import random

# 初始化
pygame.init()
screen = pygame.display.set_mode((400, 600))
clock = pygame.time.Clock()
# 加载资源
bird_img = pygame.image.load("bird.png")
bird_rect = bird_img.get_rect(center=(100, 200))

column_img = pygame.transform.scale(pygame.image.load("pillar.png"), (50, 350))
columns = []
SPAWN_COLUMN = pygame.USEREVENT
pygame.time.set_timer(SPAWN_COLUMN, 1500)

# 游戏参数
gravity = 0.1
bird_speed = 0
game_over = False
score = 0
size = 150 # 40<=size<=150

def spawn_column():
    gap_y = random.randint(200, 400)
    top = column_img.get_rect(midbottom=(400, gap_y - size))
    bottom = column_img.get_rect(midtop=(400, gap_y + size))
    return [top, bottom]

if __name__ == "__main__":
    columns.append(spawn_column())
    # 主循环
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if ((event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE) or event.type == pygame.MOUSEBUTTONDOWN) and not game_over:
                bird_speed = -3
            if event.type == SPAWN_COLUMN and not game_over:
                columns.append(spawn_column())

        if not game_over:
            # 鸟的物理
            bird_speed += gravity
            bird_rect.y += bird_speed
            if bird_rect.top <= 0 or bird_rect.bottom >= 600:
                game_over = True

            # 柱子移动
            for column_pair in columns:
                column_pair[0].x -= 2
                column_pair[1].x -= 2

            # 碰撞检测
            for column in columns:
                 if bird_rect.colliderect(column[0]) or bird_rect.colliderect(column[1]):
                    game_over = True
            for pair in columns.copy():
                if pair[0].right <= bird_rect.left:
                    columns.remove(pair)
                    score += 1

        size = 150 - score * 3
        if size < 40:
            size = 40
        # 渲染
        screen.fill((148,222,249))
        screen.blit(bird_img, bird_rect)
        for column_pair in columns:
            screen.blit(column_img, column_pair[0])
            screen.blit(column_img, column_pair[1])
        pygame.display.update()
        clock.tick(60)

pygame.quit()
