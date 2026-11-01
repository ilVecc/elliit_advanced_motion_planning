import time
import numpy as np
import networkx as nx
import trimesh as tm
from pyglet.gl import Config



REGION = np.array([100, 100, 100])

OBSTACLES : tm.Trimesh = None


def create_scene(n=5):
    global OBSTACLES
    coords = REGION[:,None] * np.random.random((3, n))
    dims = np.random.normal(30, 20, size=(3, n))
    dims[dims < 3] = 3
    dims[dims > 60] = 60
    
    obs = []
    for i in range(n):
        pos = tm.transformations.translation_matrix(coords[:,i])
        ori = tm.transformations.random_rotation_matrix()
        trans = pos @ ori
        
        if np.random.random() < 0.5:
            geom = tm.primitives.Sphere(radius=min(dims[:,i]), transform=pos)
        else:
            geom = tm.primitives.Box(dims[:,i], trans)
        # tm.primitives.Capsule
        # tm.primitives.Cylinder
        obs.append(geom)
    
    # improve computation speed ?
    OBSTACLES = tm.util.concatenate(obs)
    


def random_free_conf():
    global OBSTACLES
    ok = False
    while not ok:
        q = REGION * np.random.random(3)
        if is_free_conf(q):
            ok = True
    return q

def is_free_conf(q):
    if OBSTACLES.contains([q])[0]:
        return False
    return True

def is_through_obstacle(q0, q1):
    d = q1 - q0
    d_len = np.linalg.norm(d)

    # segment fully inside obstacle
    # or line does not exist (q0 = q1)
    if np.any(OBSTACLES.contains([q0, q1])):
        return True
    
    # look for ray hits 
    locs, _, _ = OBSTACLES.ray.intersects_location(
        ray_origins=[q0],
        ray_directions=[d],
        multiple_hits=False
    )
    if locs.size == 0:
        return False
    # keep hits before q1
    hit_len = (locs - q0) @ (d / d_len)
    if np.any(hit_len <= d_len):
        return True

    return False



def nearest_vertex(q : np.ndarray, G_pos : list[np.ndarray]) -> int:
    dist = np.linalg.norm(G_pos - q, axis=1)
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


def find_path(qf, G, G_pos):
    best = nearest_vertex(qf, G_pos)
    path = nx.algorithms.dijkstra_path(G, 0, best)
    points = [G_pos[i] for i in path] + [qf]
    return points

def simplify_path(path):
    simplified = [path[-1]]
    i = len(path)-2
    while i >= 0:
        q0 = simplified[-1]
        qf = path[i]
        if is_through_obstacle(q0, qf):
            simplified.append(path[i+1])
        i -= 1
    simplified.append(path[0])
    return simplified[::-1]
    

def main():
    global OBSTACLES
    create_scene(n=10)
    
    q0 = random_free_conf()
    time_start = time.perf_counter()
    G, G_pos = build_rrt(q0, K=2000, dq=2)
    time_total = time.perf_counter() - time_start
    print(f"RRT creation took {time_total:.4f} s.")
    
    qf = random_free_conf()
    time_start = time.perf_counter()
    path = find_path(qf, G, G_pos)
    time_total = time.perf_counter() - time_start
    print(f"Path search took {time_total:.4f} s.")
    time_start = time.perf_counter()
    path_short = simplify_path(path)
    time_total = time.perf_counter() - time_start
    print(f"Path simplification took {time_total:.4f} s.")
    
    
    scene = tm.Scene()
    # plot obstacles
    OBSTACLES.visual.face_colors = [0.9, 0, 0.1, 1.0]
    scene.add_geometry(OBSTACLES)
    # plot RRT
    edges_geom = tm.load_path([[G_pos[u], G_pos[v]] for u, v in G.edges()])
    edges_geom.colors = np.tile([100, 100, 100, 255], reps=(len(edges_geom.entities), 1))
    scene.add_geometry(edges_geom)
    # plot path
    path_segments = [[pi, pj] for pi, pj in nx.utils.pairwise(path)]
    path_geom = [tm.creation.cylinder(radius=0.65, segment=seg, sections=12) for seg in path_segments]
    path_geom = tm.util.concatenate(path_geom)
    path_geom.visual.face_colors = [0., 0., 0.95, 1.]
    scene.add_geometry(path_geom)
    # plot short path
    path_short_segments = [[pi, pj] for pi, pj in nx.utils.pairwise(path_short)]
    path_short_geom = [tm.creation.cylinder(radius=0.65, segment=seg, sections=12) for seg in path_short_segments]
    path_short_geom = tm.util.concatenate(path_short_geom)
    path_short_geom.visual.face_colors = [0., 0.95, 0., 1.]
    scene.add_geometry(path_short_geom)
    # plot start point
    start = tm.primitives.Sphere(radius=2, center=q0)
    start.visual.face_colors = [1., 1., 0., 1.]
    scene.add_geometry(start)
    # plot end point
    end = tm.primitives.Box(extents=[3,3,3], transform=tm.transformations.translation_matrix(qf))
    end.visual.face_colors = [1., 1., 0., 1.]
    scene.add_geometry(end)
    # show everything
    scene.show(
        window_conf=Config(sample_buffers=1, samples=8, depth_size=24, double_buffer=True),
        line_settings={"line_width": 3},
    )
    

if __name__ == "__main__":
    main()