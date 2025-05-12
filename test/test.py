import pygame
import sys

def render_and_save_image(output_file):
    # 初始化 pygame
    pygame.init()

    # 设置屏幕大小
    screen_width, screen_height = 800, 600
    screen = pygame.Surface((screen_width, screen_height))  # 创建一个不显示的 Surface

    # 填充背景颜色
    screen.fill((255, 255, 255))  # 白色背景

    # 绘制一些图形
    pygame.draw.rect(screen, (255, 0, 0), (100, 100, 200, 150))  # 红色矩形
    pygame.draw.circle(screen, (0, 255, 0), (400, 300), 100)     # 绿色圆形
    pygame.draw.line(screen, (0, 0, 255), (0, 0), (800, 600), 5) # 蓝色对角线

    # 保存为图片
    pygame.image.save(screen, output_file)
    print(f"Image saved to {output_file}")

    # 退出 pygame
    pygame.quit()

if __name__ == "__main__":

    output_file = r'd:/test.png'
    render_and_save_image(output_file)
