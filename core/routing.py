'''
routing algo for qubits
'''
from collections import deque

def bfs_find_target(grid, px, py):
    # 获取网格的行列大小
    rows, cols = len(grid), len(grid[0])

    # 用于标记访问过的点
    visited = set()

    # 方向数组，分别表示上下左右四个方向
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 队列用于存储当前点以及路径
    queue = deque([(px, py, [(px, py)])])  # (当前x, 当前y, 当前路径)

    # 标记起始点为已访问
    visited.add((px, py))

    while queue:
        x, y, path = queue.popleft()

        # 如果找到目标值 -1，返回路径长度和路径
        if grid[x][y] == -1:
            return len(path), path

        # 尝试向四个方向移动
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # 检查是否越界或已经访问过
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                visited.add((nx, ny))  # 标记为已访问
                queue.append((nx, ny, path + [(nx, ny)]))  # 将新点加入队列

    # 如果没有找到目标值 -1，返回 -1 和空路径
    return -1, []

#A* start
import heapq


def a_star_path(start, goal, grid,g_qubit):
    """
    使用A*算法计算二维网格中的最短路径

    参数:
        start (tuple): 起点坐标 (x, y)
        goal (tuple): 终点坐标 (x, y)
        grid (list[list[int]]): 二维网格地图，0表示可通过，-1表示障碍物

    返回:
        list[tuple]: 从起点到终点的路径坐标列表，如果不可达则返回空列表
    """
    # 检查起点和终点是否有效
    assert (0 <= start[0] < len(grid) and 0 <= start[1] < len(grid[0])) , "起点不在地图范围内"

    assert (0 <= goal[0] < len(grid) and 0 <= goal[1] < len(grid[0])), "终点不在地图范围内"


    # 定义可能的移动方向：上、下、左、右
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 初始化开放列表(优先队列)和关闭列表
    open_list = []
    heapq.heappush(open_list, (0, start))

    # 记录每个节点的来源，用于重建路径
    came_from = {}

    # 记录从起点到每个节点的实际代价
    g_score = {start: 0}

    # 记录每个节点的预估总代价(f = g + h)
    f_score = {start: heuristic(start, goal)}

    open_set = {start}

    while open_list:
        # 获取当前f值最小的节点
        current = heapq.heappop(open_list)[1]
        open_set.remove(current)

        # 如果到达目标点，重建路径
        if current == goal:
            return reconstruct_path(came_from, current)

        # 检查所有相邻节点
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            # 检查邻居是否在地图范围内且不是障碍物
            if (0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0])
                    and (grid[neighbor[0]][neighbor[1]] == 0 or grid[neighbor[0]][neighbor[1]] == g_qubit) ):

                # 计算从起点经过当前节点到邻居的代价
                tentative_g_score = g_score[current] + 1

                # 如果这是更优的路径
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                    # 如果邻居不在开放列表中，添加进去
                    if neighbor not in open_set:
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
                        open_set.add(neighbor)

    # 开放列表为空但未找到路径
    return []


def heuristic(a, b):
    """曼哈顿距离启发式函数"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from, current):
    """从终点回溯重建路径"""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()  # 反转路径，从起点到终点
    return path

if __name__ == '__main__':

    # 示例地图
    grid = [
        [0, 0, 0, 0, 0],
        [0, -1, -1, -1, 0],
        [0, -1, 0, 0, 0],
        [0, -1, 0, -1, -1],
        [0, 0, 0, -1, 0]
    ]

    start = (0, 0)  # 左上角
    goal = (0, 2)   # 右下角

    path = a_star_path(start, goal, grid)
    print("最短路径:", path)
