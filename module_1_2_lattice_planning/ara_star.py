from collections.abc import Generator
from typing import TypeVar, Self, Callable
import heapq
from collections import defaultdict


S = TypeVar("S")  # state type
class ARAStar():

    def __init__(self, 
        start: S,  goal: S, 
        heuristic: Callable[[S, S], float], 
        neighbors: Callable[[S,], list[S]], 
        max_expanded: int = 10000, 
        eps: float = 1
    ) -> None:
        """ `start` and `goal` must implement `__hash__` and `__eq__`

            :param start: starting state
            :param goal: goal state
            :param heuristic: function computing the estimated cost to reach the goal from a given node
            :param neighbors: function computing the neighbors of a given node
        """
        super()
        self.heuristic = heuristic
        self.neighbors = neighbors
        self.eps = eps
        self.max_expanded = max_expanded
        
        # Initialize open and closed sets
        self.gscore = defaultdict(lambda: float("inf"))
        self.gscore[start] = 0
        self.parent_tree = dict()
        
        self.start = start
        self.goal = goal
        self.open_heap : list[ARAStar.Key] = []     # Priority queue (uses < operator of `Key`, which uses `f`)
        self.open_dict : dict[S, ARAStar.Key] = {}  # Reverse-lookup Key object and modify it in the heap, to then heapify again
        self.closed_set = set()
        self.incons_set = set()
        self.expanded = 0

    class Key():
        def __init__(self, score : int, value : S):
            super()
            self.score = score
            self.value = value
        
        def __lt__(self, other: Self):
            return self.score < other.score


    def fvalue(self, state, goal):
        return self.gscore[state] + self.eps * self.heuristic(state, goal)

    def unweighted_fvalue(self, state, goal):
        return self.gscore[state] + self.heuristic(state, goal)

    def current_suboptimality(self):
        # could be improved by using another heap with the unweighted-fscore
        state = min(set(self.open_dict.keys()).union(self.incons_set), key=lambda x: self.unweighted_fvalue(x, self.goal))
        return self.gscore[self.goal] / self.unweighted_fvalue(state, self.goal)
    
    def update_open_set(self):
        # update keys in heap
        for key in self.open_heap:
            key.score = self.fvalue(key.value, self.goal)
        # add keys in heap
        for state in self.incons_set:
            key = ARAStar.Key(self.fvalue(state, self.goal), state)
            self.open_dict[state] = key
            heapq.heappush(self.open_heap, key)
        self.incons_set.clear()
        # refresh priority
        heapq.heapify(self.open_heap)


    def _improving_astar_path(self) -> tuple[bool, list[S]]:
        """ Find the suboptimal path using weighted-A* algorithm for the current `eps`
        """
        # modified A*
        while self.open_heap != []:
            if self.expanded > self.max_expanded:
                break
            
            # Get node with lowest f value (using < operator)
            key_current = self.open_heap[0]
            
            # # BREAK SAME-FSCORE TIES
            # if not self.open_heap.is_empty() and self.open_heap.top().f == current_node.f:
            #     tied_node = self.open_heap.top()
            #     if tied_node.g < current_node.g:
            #         tied_node = self.open_heap.extract()
            #         self.open_heap.insert(current_node)
            #         current_node = tied_node
            
            # no better state for this `eps`, returning path
            if key_current.score >= self.gscore[self.goal]:
                return True, self.reconstruct_path(key_current.value, self.parent_tree)
            
            # run A* on current node
            key_current = heapq.heappop(self.open_heap)
            current = key_current.value
            self.open_dict.pop(current)
            self.closed_set.add(current)
            
            # ignore if un-weighted fscore is worse than last solution cost
            # this is not part of the ARA* paper, but prunes the open set quite a lot
            if self.unweighted_fvalue(current, self.goal) >= self.gscore[self.goal]:
                continue
                
            # Explore neighbors
            for neighbor, cost in self.neighbors(current):
                
                # Calculate new path cost
                tentative_g = self.gscore[current] + cost
                
                if tentative_g < self.gscore[neighbor]:
                    # Found a better path to the neighbor
                    self.gscore[neighbor] = tentative_g
                    self.parent_tree[neighbor] = current
                    
                    if neighbor not in self.closed_set:
                        # neighbor never visited, but was it added?
                        if neighbor not in self.open_dict:
                            # no, add the neighbor in the heap
                            key = ARAStar.Key(self.fvalue(neighbor, self.goal), neighbor)
                            self.open_dict[neighbor] = key
                            heapq.heappush(self.open_heap, key)
                        else:
                            # yes, update the neighbor in the heap
                            key = self.open_dict[neighbor]
                            key.score = self.fvalue(neighbor, self.goal)
                            key.value = neighbor
                            heapq.heapify(self.open_heap)
                    else:
                        # neighbor already visited, ignore
                        self.incons_set.add(neighbor)
                        
                self.expanded += 1
                if self.expanded > self.max_expanded:
                    break
        
        return False, list(set.union(*[set(self.reconstruct_path(state, self.parent_tree)) for state in self.parent_tree.keys()]))  # No path found, giving all paths
    
    
    def find_path(self) -> Generator[list[S], list[S]]:
        discount = 0.75
        self.incons_set.add(self.start)
        curr_eps = self.eps / discount
        while curr_eps > 1:
            self.eps = max(1, self.eps * discount)
            self.update_open_set()
            self.closed_set.clear()
            
            success, path = self._improving_astar_path()
            if not success:
                yield "failed", path  # return all traces
                curr_eps = 0
                break
            
            # this could instead modify `self.eps`, but paper discourages it
            curr_eps = min(self.eps, self.current_suboptimality())
            if curr_eps == 1:
                yield "optimal", path  # reuse last path
            else:
                yield "success", path
    
    
    def reconstruct_path(self, state: S, parent_tree: dict[S, S]) -> list[S]:
        """ Reconstruct the path from goal to start by following parents.
        """
        path = [state]
        while state in parent_tree.keys():
            state = parent_tree[state]
            path.append(state)
        return path[::-1]  # reverse for start to goal
