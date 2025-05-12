'''
This module models the problem to be solved. In this very simple example, the problem is to optimze a Robot that works in a Warehouse.
The Warehouse is divided into a rectangular grid. A Target is randomly placed on the grid and the Robot's goal is to reach the Target.
'''
import random
from enum import Enum
from pathlib import Path

import pygame
import sys
from os import path

from config import ConfigSingleton
from core.chip import Chip, ChipAction
from utils.file_util import get_root_dir
args = ConfigSingleton().get_args()

# The Warehouse is divided into a grid. Use these 'tiles' to represent the objects on the grid.
class GridTile(Enum):
    _FLOOR = 0
    ROBOT = 1
    TARGET = 2

    # Return the first letter of tile name, for printing to the console.
    def __str__(self):
        return self.name[:1]


class ChipVisualizer:

    # Initialize the grid size. Pass in an integer seed to make randomness (Targets) repeatable.
    def __init__(self, chip:Chip, fps, total_frames):
        self.chip = chip
        self.grid_rows = self.chip.rows
        self.grid_cols = self.chip.cols
        self.reset()

        self.fps = fps
        self.last_action = ''
        self._init_pygame()

        self.num_qubits = chip.num_qubits
        self.total_frames = total_frames  # 总渲染次数


    def _init_pygame(self):
        pygame.init()  # initialize pygame
        pygame.display.init()  # Initialize the display module

        img_path = Path(get_root_dir()) / 'assets' / 'chip_render'

        # Game clock
        self.clock = pygame.time.Clock()

        # Default font
        self.action_font = pygame.font.SysFont("Calibre", 30)
        self.action_info_height = self.action_font.get_height()

        # For rendering
        self.cell_height = 64
        self.cell_width = 64
        self.cell_size = (self.cell_width, self.cell_height)

        # Define game window size (width, height)
        self.window_size = (
            self.cell_width * self.grid_cols, self.cell_height * self.grid_rows + self.action_info_height + 20)

        # Initialize game window
        #self.window_surface = pygame.display.set_mode(self.window_size)
        self.window_surface = pygame.display.set_mode(self.window_size)

        # 进度条相关参数
        self.progress_bar_width = self.window_surface.get_width()  # 进度条宽度
        self.progress_bar_height = 20  # 进度条高度
        self.progress_bar_color = (0, 128, 0)  # 进度条颜色
        self.progress_bar_background_color = (200, 200, 200)  # 进度条背景颜色
        self.progress_bar_y = self.window_surface.get_height() - self.progress_bar_height  # 进度条的 Y 坐标
        self.current_frame = 0  # 当前渲染次数

        # 拖动状态
        self.is_dragging = False

        # Load & resize objects(qubits, magic state, broken qubits, etc.)
        self.qubit_imgs = self.get_qubits_object()


        file_name =  img_path / 'tile_light_grey.png'

        img = pygame.image.load(file_name)
        self.floor_img = pygame.transform.scale(img, self.cell_size)
        #
        # file_name = path.join(path.dirname(__file__), r"D:\sync\gym_custom_env-main\gym_custom_env-main\sprites\package.png")
        # img = pygame.image.load(file_name)
        # self.goal_img = pygame.transform.scale(img, self.cell_size)
        # self.goal_img = self.add_text_to_img(self.goal_img,1, )

    def _draw_progress_bar(self):
        # 计算进度条的当前宽度
        progress_ratio = self.current_frame / self.total_frames
        current_width = int(self.progress_bar_width * progress_ratio)

        # 绘制进度条背景
        progress_bar_rect = pygame.Rect(0, self.progress_bar_y, self.progress_bar_width, self.progress_bar_height)
        pygame.draw.rect(self.window_surface, self.progress_bar_background_color, progress_bar_rect)

        # 绘制当前进度
        current_progress_rect = pygame.Rect(0, self.progress_bar_y, current_width, self.progress_bar_height)
        pygame.draw.rect(self.window_surface, self.progress_bar_color, current_progress_rect)

    def get_qubits_object(self) -> list:
        qubit_imags = []
        img_path = Path(get_root_dir()) / 'assets' / 'chip_render'
        for q in range(1, self.chip.num_qubits + 1):
            file_name = img_path  / 'tile_red.png'
            img = pygame.image.load(file_name)
            img = pygame.transform.scale(img, self.cell_size)
            img = self.add_text_to_img(img,q)
            qubit_imags.append(img)
        print('qubits_imgs:',qubit_imags)
        return qubit_imags

    # 定义一个函数，在图片上绘制数字
    def add_text_to_img(self,image, number):
        # 准备字体对象
        font = pygame.font.Font(None, int(self.cell_width*0.8))  # None 表示使用默认字体，36 是字体大小
        # 创建图片副本
        image_copy = image.copy()

        # 渲染数字为文本表面
        text_surface = font.render(str(number), True, (255, 255, 255))  # 白色字体

        # 获取文本表面的大小
        text_rect = text_surface.get_rect(center=(self.cell_width // 2, self.cell_height // 2))

        # 将文本表面绘制到图片副本上
        image_copy.blit(text_surface, text_rect)

        return image_copy

    def reset(self, seed=None):
        # Initialize Robot's starting position
        self.robot_pos = [0, 0]

        # Random Target position
        random.seed(seed)
        self.target_pos = [
            random.randint(1, self.grid_rows - 1),
            random.randint(1, self.grid_cols - 1)
        ]

    def perform_action(self, player: int, action: ChipAction):
        #print('perform action:', player, action)
        self.chip.move(player,action)


    def render(self):
        self._process_events()

        # clear to white background, otherwise text with varying length will leave behind prior rendered portions
        self.window_surface.fill((255, 255, 255))

        # Draw floor
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                pos = (c * self.cell_width, r * self.cell_height)
                self.window_surface.blit(self.floor_img, pos)

        #render the qubits
        #print('position:', self.chip.position)
        for i in range(0, self.chip.num_qubits):
            r,c = self.chip.position[i+1]
            pos = (c * self.cell_width, r * self.cell_height)
            self.window_surface.blit(self.qubit_imgs[i], pos)

        self.current_frame += 1
        self._draw_progress_bar()
        #show bottom text
        # text_img = self.action_font.render(f'Action: {self.last_action}', True, (0, 0, 0), (255, 255, 255))
        # text_pos = (0, self.window_size[1] - self.action_info_height)
        # self.window_surface.blit(text_img, text_pos)

        pygame.display.update()



        # Limit frames per second
        self.clock.tick(self.fps)

    def _process_events(self):

        # Process user events, key presses
        for event in pygame.event.get():
            # User clicked on X at the top right corner of window
            if event.type == pygame.QUIT:
                # save window to img
                if args.save_render_img:
                    self.save_to_img()
                pygame.quit()
                sys.exit()

            if (event.type == pygame.KEYDOWN):
                # User hit escape
                if (event.key == pygame.K_ESCAPE):
                    # save window to img
                    if args.save_render_img:
                        self.save_to_img()
                    pygame.quit()
                    sys.exit()
    def save_to_img(self):
        path = Path(args.results_evaluate_path, (args.time_id + 'render.png'))
        print('saving to img:',path)
        pygame.image.save(self.window_surface, path)

    def render_trace(self):
        pass

def show_trace(trace:list):
    chip = Chip(num_qubits=len(trace),cols=args.chip_cols,rows=args.chip_rows)
    total_frames = len(trace[0])
    CV = ChipVisualizer(chip = chip, fps=8,total_frames=total_frames)
    CV.render()
    for j in range(total_frames):
         for i in range(len(trace)):
            if j >= len(trace[i]):
                continue
            act = trace[i][j]
            player = i+1
            CV.perform_action(player, act)
         CV.render()
    #CV.save_to_img()

#test code
if __name__ == "__main__":
    # chip = Chip(10, 5)
    # warehouseRobot = WarehouseRobot(chip,fps=3)
    # warehouseRobot.render()

    # while (True):
    #     rand_action = random.choice(list(ChipAction))
    #     player = random.randint(1, chip.num_qubits)
    #     warehouseRobot.perform_action(player,rand_action)
    #     warehouseRobot.render()
    trace =[[3, 0, 2, 3, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [3, 1, 1, 1, 1, 1, 0, 1, 3, 1, 3, 1, 3, 1, 3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [3, 3, 2, 3, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2]]

    show_trace(trace)