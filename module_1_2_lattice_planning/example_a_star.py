import numpy as np
import matplotlib.pyplot as plt

from a_star import AStar



pos = tuple[int, int]
class Scene():
    
    def __init__(self):
        self.grid = np.zeros((20, 20))  # 20x20 grid
        self.grid[5:15, 10] = 1  # Vertical wall
        self.grid[5, 5:15] = 1   # Horizontal wall
    
    def neighbors(self, state: pos) -> list[tuple[pos, float]]:
        """ Get all valid neighboring positions.
        
            :param map: 2D numpy array where 0 represents walkable cells and 1 represents obstacles
        """
        x, y = state[0], state[1]
        rows, cols = self.grid.shape
        
        # All possible moves (including diagonals)
        possible_moves = [
            (x+1, y), (x-1, y),    # Right, Left
            (x, y+1), (x, y-1),    # Up, Down
            (x+1, y+1), (x-1, y-1),  # Diagonal moves
            (x+1, y-1), (x-1, y+1)
        ]
        
        allowed_moves = []
        for nx, ny in possible_moves:
            if 0 <= nx < rows and 0 <= ny < cols and self.grid[nx, ny] == 0:
                neighbor = (nx, ny)
                cost = np.hypot(x - nx, y - ny)
                allowed_moves.append((neighbor, cost))
        
        return allowed_moves
    
    def visualize_path(self, path: list[pos]):
        """
        Visualize the grid and found path.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap='binary')
        
        if path:
            path = np.array(path)
            plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=3, label='Path')
            plt.plot(path[0, 1], path[0, 0], 'go', markersize=15, label='Start')
            plt.plot(path[-1, 1], path[-1, 0], 'ro', markersize=15, label='Goal')
        
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.title("A* Pathfinding Result")
        plt.show()


if __name__ == "__main__":

    scene = Scene()
    start = (2, 2)
    goal = (12, 12)

    # run A*
    heuristic = lambda state, goal: np.hypot(state[0] - goal[0], state[1] - goal[1])
    neighbors = lambda state: scene.neighbors(state)
    done, path = AStar().find_path(start, goal, heuristic, neighbors)
    if done:
        print(f"Path found with {len(path)} steps!")
    else:
        print("No path found!")
    scene.visualize_path(path)
