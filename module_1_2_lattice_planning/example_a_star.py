from typing_extensions import override
import numpy as np
import matplotlib.pyplot as plt

from a_star import Node, AStar


class Grid2DNode(Node[np.ndarray, np.ndarray]):
    
    @override
    def __hash__(self):
        return hash(self.value.data.tobytes())
    
    @override
    def heuristic(self, goal: "Grid2DNode") -> float:
        """ Calculate the estimated distance between two points using Euclidean distance.
        """
        return np.linalg.norm(self.value - goal.value)
    
    def _cost_move(self, neighbor: "Grid2DNode"):
        return np.linalg.norm(self.value - neighbor.value)

    @override
    def neighbors(self, scene: np.ndarray) -> list[tuple["Grid2DNode", float]]:
        """ Get all valid neighboring positions.
        
            :param map: 2D numpy array where 0 represents walkable cells and 1 represents obstacles
        """
        x, y = self.value
        rows, cols = scene.shape
        
        # All possible moves (including diagonals)
        possible_moves = [
            (x+1, y), (x-1, y),    # Right, Left
            (x, y+1), (x, y-1),    # Up, Down
            (x+1, y+1), (x-1, y-1),  # Diagonal moves
            (x+1, y-1), (x-1, y+1)
        ]
        
        allowed_moves = []
        for nx, ny in possible_moves:
            if 0 <= nx < rows and 0 <= ny < cols and scene[nx, ny] == 0:
                neighbor = Grid2DNode(np.array([nx, ny]), parent=self, eps=self.eps)
                cost = self._cost_move(neighbor)
                allowed_moves.append((neighbor, cost))
        
        return allowed_moves
    
    @override
    def close_to_goal(self, goal: "Grid2DNode"):
        return np.linalg.norm(self.value - goal.value) < 0.2


def visualize_path(grid: np.ndarray, path: list[np.ndarray]):
    """
    Visualize the grid and found path.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='binary')
    
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
    
    # Create a sample grid
    grid = np.zeros((20, 20))  # 20x20 grid
    grid[5:15, 10] = 1  # Vertical wall
    grid[5, 5:15] = 1   # Horizontal wall

    start_pos = np.array([2, 2])
    goal_pos = np.array([12, 12])

    # run A*
    done, path = AStar(Grid2DNode).find_path(grid, start_pos, goal_pos, eps=2.5)
    if done:
        print(f"Path found with {len(path)} steps!")
    else:
        print("No path found!")
    visualize_path(grid, path)
