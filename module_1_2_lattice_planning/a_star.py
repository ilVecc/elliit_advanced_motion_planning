from typing import TypeVar, Self, Callable
import heapq
from collections import defaultdict


S = TypeVar("S")  # state type
class AStar():

    class Key():
        def __init__(self, score : int, value : S):
            super()
            self.score = score
            self.value = value
        
        def __lt__(self, other: Self):
            return self.score < other.score

    def find_path(self, 
        start: S,  goal: S,
        heuristic: Callable[[S, S], float], 
        neighbors: Callable[[S,], list[S]], 
        max_expanded: int = 10000
    ) -> tuple[bool, list[S]]:
        """ Find the optimal path using A* algorithm.
            `start` and `goal` must implement `__hash__` and `__eq__`

            :param start: starting state
            :param goal: goal state
            :param heuristic: function computing the estimated cost to reach the goal from a given node
            :param neighbors: function computing the neighbors of a given node
        """
        
        # Initialize open and closed sets
        gscore = defaultdict(lambda: float("inf"))
        gscore[start] = 0
        fscore = defaultdict(lambda: float("inf"))
        fscore[start] = heuristic(start, goal)
        parent_tree = dict()
        
        start_key = AStar.Key(fscore[start], start)
        open_heap = [start_key]                 # Priority queue (uses < operator of `Key`, which uses `f`)
        open_dict = {start: start_key}          # Reverse-lookup Key object and modify it in the heap, to then heapify again
        expanded = 0
        
        while open_heap != []:
            
            if expanded > max_expanded:
                break
            
            # Get node with lowest f value (using < operator)
            current = heapq.heappop(open_heap).value
            
            if current == goal:
                return True, self.reconstruct_path(current, parent_tree)
            
            open_dict.pop(current)
            
            # Explore neighbors
            for neighbor, cost in neighbors(current):
                
                # Calculate new path cost
                tentative_g = gscore[current] + cost
                
                if tentative_g < gscore[neighbor]:
                    # Found a better path to the neighbor
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + heuristic(neighbor, goal)
                    parent_tree[neighbor] = current
                    
                    if neighbor not in open_dict:
                        # add neighbor to the frontier
                        key = AStar.Key(fscore[neighbor], neighbor)
                        open_dict[neighbor] = key
                        heapq.heappush(open_heap, key)
                    else:
                        # better path to neighbor, update the heap
                        key = open_dict[neighbor]
                        key.score = fscore[neighbor]
                        # This has nothing to do with A* itself: this "new" neighbor object has the same hash, but 
                        # internally it might contain different info (e.g. for advanced plotting), so by changing 
                        # the value we make sure nothing is left behind.
                        key.value = neighbor
                        heapq.heapify(open_heap)
                
                expanded += 1
                if expanded > max_expanded:
                    break
                
        return False, list(set.union(*[set(self.reconstruct_path(state, parent_tree)) for state in parent_tree.keys()]))  # No path found, giving all paths
    
    def reconstruct_path(self, state: S, parent_tree: dict[S, S]) -> list[S]:
        """ Reconstruct the path from goal to start by following parents.
        """
        path = [state]
        while state in parent_tree.keys():
            state = parent_tree[state]
            path.append(state)
        return path[::-1]  # reverse for start to goal
