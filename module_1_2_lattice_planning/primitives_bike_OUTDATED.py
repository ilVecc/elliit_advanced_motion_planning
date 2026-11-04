import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

LR = 0.4 
LF = 0.6

def model(u, s, lr=0.4, lf=0.6):
    """ Model of a bicycle in local frame.
    """
    L = lr + lf
    wz = u/L * np.tan(s)
    vy = lf * wz
    vx = u
    return np.array([vx, vy, wz])

def rotz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], 
                     [s,  c, 0], 
                     [0,  0, 1]])

def simulate(x0, u, s, T, dt):
    o = [x0]
    for _ in range(int(T/dt)):
        p, R = o[-1], rotz(o[-1][2])
        dx = model(u, s, LR, LF)  # local [vx vy wz]
        x = p + R @ dx * dt
        o.append(x)
    return np.vstack(o)


def generate_primitives():
    """ Generate local primitives (suppose x0 = [0, 0, 0])
    """
    cases = [
        (u, s)
        for u in [3.6, 7.2] # 10 km/h, 20 km/h
        for s in [-np.pi/6, -np.pi/12, -np.pi/24, 0, +np.pi/24, +np.pi/12, +np.pi/6]
    ]
    
    DT = 1  # seconds
    dt = 0.05  # seconds
    x0 = np.zeros(3)
    
    primitives_paths = [
        simulate(x0, u, s, DT, dt)
        for u, s in cases
    ]
    
    return cases, primitives_paths
    

def plot_primitives(primitive_cmd, primitive_paths):
    fig, ax = plt.subplots()
    for cmd, primitive in zip(primitive_cmd, primitive_paths):
        x = primitive[:,0]
        y = primitive[:,1]
        th = primitive[:,2]
        u = cmd[0]
        s = cmd[1]
        # for i in range(primitive.shape[0]):
        #     plot_bike([x[i], y[i], th[i]], [u, s], ax)
        ax.quiver(x, y, np.cos(th), np.sin(th))
        ax.plot(x, y)
    ax.set_aspect("equal")
    ax.grid(True, "both")
    plt.show()

def plot_bike(state, cmd, ax):
    """ state = [x, y, th]
        cmd = [u, s]
    """
    steer = cmd[1]
    xi, yi, thi = state[0], state[1], state[2]
    bike_rx, bike_ry = xi-LR*np.cos(thi), yi-LR*np.sin(thi)
    bike_fx, bike_fy = xi+LF*np.cos(thi), yi+LF*np.sin(thi)
    ax.plot((bike_rx,bike_fx), (bike_ry,bike_fy), linewidth=5, color="tab:blue")
    wheel = patch.Ellipse((bike_fx, bike_fy), (LF+LR)/10, (LF+LR)/20, angle=np.rad2deg(thi+steer), facecolor="tab:orange", linewidth=2.5, edgecolor="tab:red")
    ax.add_patch(wheel)



if __name__ == "__main__":
    primitive_cmd, primitive_paths = generate_primitives()
    plot_primitives(primitive_cmd, primitive_paths)
