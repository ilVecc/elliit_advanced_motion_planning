from typing import Optional, Generic, TypeVar, Self
from abc import ABCMeta, abstractmethod
import heapq


V = TypeVar("T")  # value type
S = TypeVar("S")  # scene type

class Node(Generic[V, S], metaclass=ABCMeta):
    def __init__(self, 
        value: V, 
        g: Optional[float] = float('inf'), 
        h: Optional[float] = 0.0, 
        parent: Optional[Self] = None,
        eps : float = 1
    ):
        """ Node for the A* algorithm.
            :param position: coordinates of the node
            :param g: cost from start to this node
            :param h: estimated cost from this node to goal (heuristic)
            :param parent: parent node
            :param eps: inflation of the heuristic
        """
        super()
        self.value = value
        self.g = g
        self.h = h
        self.parent = parent
        self.eps = eps
    
    @property
    def f(self):
        return self.g + self.eps * self.h
    
    def __eq__(self, other: Self):
        return self.f == other.f
    
    def __ne__(self, other: Self):
        return self.f != other.f
    
    def __lt__(self, other: Self):
        return self.f < other.f
    
    def __gt__(self, other: Self):
        return self.f > other.f
    
    def __ge__(self, other: Self):
        return self.f >= other.f
    
    @abstractmethod
    def __hash__(self):
        raise NotImplementedError()

    @abstractmethod
    def heuristic(self, goal: Self) -> float:
        """ Calculate the estimated distance to the goal.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def neighbors(self, scene : S) -> list[tuple[Self, float]]:
        """ Get all valid neighboring positions.
        
            :param scene: the representation of the environment
        """
        raise NotImplementedError()

    @abstractmethod
    def close_to_goal(self, goal: Self) -> bool:
        raise NotImplementedError()


class AStar():
    
    def __init__(self, node_type: type[Node[V, S]]):
        self.node_type = node_type
    
    def find_path(self, scene: S, start: V,  goal: V, max_explored: int = 10000, eps: float = 1) -> tuple[bool, list[V]]:
        """ Find the optimal path using A* algorithm.
        
            :param grid: 2D numpy array (0 = free space, 1 = obstacle)
            :param start: Starting position (x, y)
            :param goal: Goal position (x, y)
            :param eps: inflation of the heuristic
        """
        # Initialize start node
        start_node = self.node_type(start, g=0, parent=None, eps=eps)
        goal_node = self.node_type(goal, parent=None, eps=eps)
        # setting h is useless since the start node is the only node ad the beginning, 
        # so there is no need for actual comparison of f-scores (requiring h) 
        # start_node.h = heuristic(start_node, goal_node)
        
        # Initialize open and closed sets
        open_heap = [start_node]                    # Priority queue
        open_dict = {hash(start_node): start_node}  # For quick node lookup
        closed_set = set()                          # Explored nodes
        
        while open_heap != []:
            if len(closed_set) > max_explored:
                break
            
            # Get node with lowest f value
            current_node = heapq.heappop(open_heap)
            
            # Check if we've reached the goal
            if current_node.close_to_goal(goal_node):
                return True, self.reconstruct_path(current_node)
                
            closed_set.add(current_node)
            
            # Explore neighbors
            for neighbor_node, cost in current_node.neighbors(scene):
                # Skip if already explored
                if neighbor_node in closed_set:
                    continue
                
                # Calculate new path cost
                tentative_g = current_node.g + cost
                
                # Create or update neighbor
                if hash(neighbor_node) not in open_dict:
                    # New neighbor has default g and h
                    neighbor_node.g = tentative_g
                    neighbor_node.h = neighbor_node.heuristic(goal_node)
                    heapq.heappush(open_heap, neighbor_node)
                    open_dict[hash(neighbor_node)] = neighbor_node
                    
                elif tentative_g < open_dict[hash(neighbor_node)].g:
                    # Found a better path to the (known) neighbor
                    cache_neighbor_node = open_dict[hash(neighbor_node)]
                    cache_neighbor_node.value = neighbor_node.value  # update the value
                    cache_neighbor_node.g = tentative_g
                    cache_neighbor_node.parent = current_node
        
        return False, list(set.union(*[set(self.reconstruct_path(node)) for node in closed_set]))  # No path found, giving all paths
    
    def reconstruct_path(self, goal_node: Node[V, S]) -> list[V]:
        """ Reconstruct the path from goal to start by following parents.
        """
        path = []
        current = goal_node
        while current is not None:
            path.append(current.value)
            current = current.parent
        return path[::-1]  # reverse for start to goal
