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
