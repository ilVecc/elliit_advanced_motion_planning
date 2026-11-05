import numpy as np
import matplotlib.pyplot as plt
import shapely as sh
from shapely.plotting import plot_polygon

from a_star import AStar
from primitives_car import CarState


class GridScene:
    
    def __init__(self, size: tuple[int, int], n_obs: int = 10):
        super()
        self.w, self.h = size[0], size[1]
        avg_size = (self.w + self.h) / 2
        
        # generate obstacles        
        coords = np.random.random((2, n_obs))
        coords[0,:] *= self.w
        coords[1,:] *= self.h
        dims = np.random.normal(10, 10, size=(2, n_obs))
        dims = np.clip(dims, avg_size*0.05, avg_size*0.65)
        angles = np.random.uniform(0, 2*np.pi, size=n_obs)
        
        obs = []
        for i in range(n_obs):
            geom = sh.box(coords[0,i]-dims[0,i]/2, coords[1,i]-dims[1,i]/2, coords[0,i]+dims[0,i]/2, coords[1,i]+dims[1,i]/2)
            geom = sh.affinity.rotate(geom, angles[i], use_radians=True)
            obs.append(geom)
        self.obstacles = sh.GeometryCollection(obs)
        sh.prepare(self.obstacles)
        
        self._cache_valid_pos = {}
    
    def is_free_position(self, pos: np.ndarray) -> bool:
        return not sh.contains_xy(self.obstacles, pos[0], pos[1])

    def random_free_state(self) -> CarState:
        while True:
            x = np.random.randint(0, self.w)
            y = np.random.randint(0, self.h)
            th = (2*np.pi) * (np.random.randint(0, 4) / 4)
            if self.is_valid_position((x, y)):
                return CarState(x, y, th)

    def is_valid_position(self, pos: tuple[int, int]) -> bool:
        if pos not in self._cache_valid_pos:
            is_inside = 0 < pos[0] < self.w and 0 < pos[1] < self.h
            is_free = self.is_free_position(pos)
            self._cache_valid_pos[pos] = is_inside and is_free
        return self._cache_valid_pos[pos]
    
    def neighbors(self, state: CarState) -> list[tuple[CarState, float]]:
        """ Get all valid neighboring positions.
        """
        neighbors = state.next_states()
        
        allowed_neigh = []
        for neighbor in neighbors:
            x, y = neighbor.x, neighbor.y
            if scene.is_valid_position((x, y)) \
            and (neighbor.from_action != "fs_long" 
                 or (neighbor.th % (2*np.pi) == 0          and scene.is_valid_position((x-1, y)))
                 or (neighbor.th % (2*np.pi) == np.pi/2    and scene.is_valid_position((x, y-1)))
                 or (neighbor.th % (2*np.pi) == np.pi      and scene.is_valid_position((x+1, y)))
                 or (neighbor.th % (2*np.pi) == 3*np.pi/2  and scene.is_valid_position((x, y+1)))
            ):
                allowed_neigh.append((neighbor, neighbor.cost))
        
        return allowed_neigh
    
    def visualize(self, start: CarState, goal: CarState, path: list[CarState] = []):
        """ Visualize scene and found path.
        """
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
            
        for g in self.obstacles.geoms:
            plot_polygon(g, add_points=False, color="red", ax=ax)
        
        cmap = plt.colormaps.get_cmap("jet").reversed()
        tot = np.hypot(goal.x - start.x, goal.y - start.y)
        for i in range(len(path)):
            par = np.hypot(goal.x - path[i].x, goal.y - path[i].y)
            path[i].plot(ax, action_color=cmap(par / tot))
        
        ax.plot(start.x, start.y, marker="o", markersize=7, color=cmap(1.))
        ax.plot(goal.x, goal.y, marker="*", markersize=12, color=cmap(0.))
        start.plot(ax, state_color=cmap(1.), no_parent=True)
        goal.plot(ax, state_color=cmap(0.), no_parent=True)
        
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_aspect("equal")
        ax.set_title("A* Pathfinding Result")
        ax.set_xlim(0, self.w)
        ax.set_ylim(0, self.h)
        plt.show()
        
    

def load_seed(seed: int | None):
    if seed is None:
        # load a seed or try a new example
        seed = int(np.random.random() * 123456789)
    print(f"Seed: {seed}")
    np.random.seed(seed)


if __name__ == "__main__":
    
    seed = None
    # seed = 4565  # a cul-de-sac !
    # seed = 4563
    # seed = 921797
    # seed = 593598
    # seed = 691205
    # seed = 447080
    # seed = 1049930  # UNLUCKY ! also this 812222
    load_seed(seed)
    
    scene = GridScene((50, 50))
    start = scene.random_free_state()
    goal = scene.random_free_state()

    # run A*
    heuristic = lambda state, goal: np.hypot(state.x - goal.x, state.y - goal.y)
    neighbors = lambda state: scene.neighbors(state)
    done, path_or_tree = AStar().find_path(start, goal, heuristic, neighbors, max_expanded=100000)
    if done:
        print(f"Path found with {len(path_or_tree)} steps!")
    else:
        print(f"No path found!")
    scene.visualize(start, goal, path_or_tree)
