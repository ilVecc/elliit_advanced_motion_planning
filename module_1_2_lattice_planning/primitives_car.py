import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.transforms as transforms
from matplotlib.collections import PatchCollection


class State():
    
    actions = {
        # "fl_long" : (-1, +2),
        "fl_short" : ((+1, +1, +np.pi/2),   1/(np.pi/2)),   # cost is time = 1/length
        "fs_short" : ((+1,  0, +0),         1),             # "longer" actions are ran "faster"
        "fs_long" :  ((+2,  0, +0),         1/2),
        "fr_short" : ((+1, -1, -np.pi/2),   1/(np.pi/2)),
        # "fr_long" : (+1, +2),
        "bl_short" : ((-1, +1, -np.pi/2),   2/(np.pi/2)),   # backwards costs double
        "bs_short" : ((-1,  0, +0),         2*1),
        "br_short" : ((-1, -1, +np.pi/2),   2/(np.pi/2)),
    }
    
    kwargs = {"linewidth": 3}
    _plot_patches = {
        "fl_short": patch.Arc((0, +1), 2, 2, theta1=180, theta2=270, **kwargs),
        "fr_short": patch.Arc((0, -1), 2, 2, theta1=90, theta2=180, **kwargs),
        "bl_short": patch.Arc((0, +1), 2, 2, theta1=270, theta2=0, **kwargs),
        "br_short": patch.Arc((0, -1), 2, 2, theta1=0, theta2=90, **kwargs),
        "fs_short": patch.FancyArrow(*(0, 0), *(-1, 0), head_length=0, **kwargs),
        "fs_long":  patch.FancyArrow(*(0, 0), *(-2, 0), head_length=0, **kwargs),
        "bs_short": patch.FancyArrow(*(0, 0), *(+1, 0), head_length=0, **kwargs),
        "heading" : patch.FancyArrow(0, 0, 0.25, 0, head_width=0.1, linewidth=3)
    }
    
    def __init__(self, x: int, y: int, th: float, from_action : str = ""):
        super()
        self.x = x
        self.y = y
        # self.h = heading
        self.th = th
        self.from_action = from_action
        self.cost = 0
        
    def plot(self, ax, action_color = "tab:blue", state_color = "black", no_parent = False):
        # ax.scatter(self.x, self.y, s=30, color="black")
        
        patches = []
        p_arrow = self._plot_patches["heading"]
        p_arrow.set_color(state_color)
        patches.append(p_arrow)
        if self.from_action != "" and not no_parent:
            p_action = self._plot_patches[self.from_action]
            p_action.set_color(action_color)
            patches.append(p_action)
        
        t = transforms.Affine2D().rotate(self.th).translate(self.x, self.y)
        
        coll = PatchCollection(patches, match_original=True)
        coll.set_transform(t+ax.transData)
        ax.add_collection(coll, autolim=True)
        
    @staticmethod
    def rotz(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, 0], 
                        [s,  c, 0], 
                        [0,  0, 1]])

    def next_states(self) -> list["State"]:
        neigh = []
        c, s = np.cos(self.th), np.sin(self.th)
        for name, (dx, cost) in self.actions.items():
            x = int(self.x + dx[0] * c - dx[1] * s)
            y = int(self.y + dx[0] * s + dx[1] * c)
            th = (self.th + dx[2]) % (2*np.pi)
            sn = State(x, y, th, name)
            sn.cost = cost
            neigh.append(sn)
        return neigh
    
            
if __name__ == "__main__":
    x0 = State(0, 0, 0)
    
    fig, ax = plt.subplots()
    ax.grid(True, "both")
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    
    x0.plot(ax)
    for xi in x0.next_states()[:2]:
        xi.plot(ax, "tab:blue")
        for xj in xi.next_states()[:2]:
            xj.plot(ax, "tab:red")
            for xk in xj.next_states()[:2]:
                xk.plot(ax, "tab:green")
    
    ax.autoscale_view()
    plt.show()
