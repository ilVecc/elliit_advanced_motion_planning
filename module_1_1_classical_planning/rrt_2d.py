import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import shapely as sh
from shapely.plotting import plot_polygon

REGION_W = 100
REGION_H = 100

OBSTACLES : sh.GeometryCollection

def create_scene(n=10):
    global OBSTACLES
    coords = np.random.random((2, n))
    coords[0,:] *= REGION_W
    coords[1,:] *= REGION_H
    dims = np.random.normal(10, 10, size=(2, n))
    dims[dims < 3] = 3
    dims[dims > 50] = 50
    angles = np.random.uniform(0, 2*np.pi, size=n)
    
    obs = []
    for i in range(n):
        geom = sh.box(coords[0,i]-dims[0,i]/2, coords[1,i]-dims[1,i]/2, coords[0,i]+dims[0,i]/2, coords[1,i]+dims[1,i]/2)
        geom = sh.affinity.rotate(geom, angles[i], use_radians=True)
        obs.append(geom)
    OBSTACLES = sh.GeometryCollection(obs)
    

def random_free_conf():
    global OBSTACLES
    ok = False
    while not ok:
        q = np.random.random(2)
        q[0] *= REGION_W
        q[1] *= REGION_H
        if is_free_conf(q):
            ok = True
    return q

def is_free_conf(q):
    return not OBSTACLES.contains(sh.Point(q))

def is_through_obstacle(q0, q1):
    return OBSTACLES.intersects(sh.LineString([q0, q1]))
    

def nearest_vertex(q : np.ndarray, G : list[np.ndarray]) -> int:
    dist = np.linalg.norm(G - q, axis=1)
    best = np.argmin(dist)
    return best

def new_conf(q0, q1, dq):
    diff = q1 - q0
    q = q0 + dq * diff / np.linalg.norm(diff)
    return q


def build_rrt(q0 : np.ndarray, K : int, dq : float) -> nx.Graph:
    """ Build the RRT graph.
        :param q0: initial configuration
        :param K: number of vertices
        :param Dq: incremental distance
    """
    G = nx.Graph()
    G.add_node(0)
    points = [q0]
    
    for k in range(1, K):
        ok = False
        while not ok:
            qrand = random_free_conf()
            qnear_idx = nearest_vertex(qrand, points)
            qnear = points[qnear_idx]
            ## Setting a dq forces steps to be limited, making exploration much slower
            # qnew = new_conf(qnear, qrand, dq)
            qnew = qrand
            if not is_through_obstacle(qnear, qnew):
                ok = True
        G.add_node(k)
        G.add_edge(qnear_idx, k)
        points.append(qnew)
    return G, points


def main():
    global OBSTACLES
    create_scene()
    
    q0 = random_free_conf()
    G, G_pos = build_rrt(q0, K=2000, dq=2)
    
    qf = random_free_conf()
    best = nearest_vertex(qf, G_pos)
    path = nx.algorithms.dijkstra_path(G, 0, best)
    
    
    ax = plt.axes()
    ax.set_aspect('equal')
    for g in OBSTACLES.geoms:
        plot_polygon(g, add_points=False, color="red", ax=ax)
    pos = {i: p for (i, p) in enumerate(G_pos)}
    nx.draw(G, pos, node_size=1.5, width=2, edge_color="k", node_color="k", ax=ax)
    nx.draw(G, pos, nodelist=path, edgelist=list(nx.utils.pairwise(path)), node_size=2, width=3, edge_color="y", node_color="y", ax=ax)
    ax.scatter(*q0, s=30, c="y", marker="o", zorder=100)
    ax.scatter(*qf, s=30, c="y", marker="*", zorder=100)
    plt.show()

if __name__ == "__main__":
    main()
