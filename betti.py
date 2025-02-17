import networkx as nx 
import numpy as np
from typing import List
import multiprocessing as mp
from math import pow,sqrt
from numpy.linalg import eigvalsh


def edge_in_cycle(edge: tuple[int,int],cycle: List[int]):
    """
    Determines the orientation of an edge within a given cycle.

    Args:
        edge (tuple[int,int]): A tuple (u, v) representing the directed edge from u to v.
        cycle (List[int]): A list of vertices representing a cycle in the graph.

    Returns:
        int: 
        - 1 if the edge (u, v) appears in the given order in the cycle.
        - -1 if the reversed edge (v, u) appears in the cycle.
        - 0 if neither (u, v) nor (v, u) is traversed in the cycle.
    """
    (v1, v2) = edge
    for k in range(len(cycle)):
        if cycle[k] == v1 and cycle[(k+1) % len(cycle)] == v2:
            return 1
        elif cycle[k] == v2 and cycle[(k+1) % len(cycle)] == v1:
            return -1
    else:
        return 0
    
def _bakry_emery_vertex(x):
    """Compute Bakry-Émery of a given vertex x. We use the formula provided in 'Bakry-Émery curvature on graphs as an eigenvalue problem' by Cushing et al. (2021)

    Args:
        x (int): Vertex

    Returns:
        Tuple: A tuple containing the vertex and the Bakry-Émery curvature
    """

    # 1-Sphere
    S_1 = np.nonzero(_P[x])[0]
    if len(S_1)==0:
        return 0
    
    n = len(_P)
    # two-sphere
    S_2 = [y for y in range(n) if _P2[x,y] != 0 and _P[x,y] == 0 and y != x]

    A_inf = np.zeros((len(S_1), len(S_1)))
    
    for i in range(len(S_1)):
        y = S_1[i]
        A_inf[i,i] = (pow(_P[x,y],2) + 3/2 * _P[x,y] * _P[y,x]  - 1/2 * _d[x] / _m[x] * _P[x,y] + 3/2 * _P[x,y] * np.sum(_P[y][S_2]) + 3/2 * _P[x,y] * np.sum(_P[y][S_1]) + 1/2 * _P2[x,y] - 2 * pow(_P[x,y],2) * np.sum(np.square(_P[y][S_2]) * np.reciprocal(_P2[x][S_2]))) / _P[x,y]                    

    for i in range(len(S_1)):
        for j in range(i+1, len(S_1)):
            y_i, y_j = S_1[i], S_1[j]
            A_inf[i,j] = (_P[x,y_i] * _P[x,y_j] - _P[x,y_i] * _P[y_i, y_j] - _P[x,y_j] * _P[y_j,y_i] - 2 * _P[x,y_i] * _P[x,y_j] * np.sum(_P[y_i][S_2] * _P[y_j][S_2] * np.reciprocal(_P2[x][S_2]))) / sqrt(_P[x,y_i]*_P[x,y_j])
            A_inf[j,i] = A_inf[i,j]

    return x, round(eigvalsh(A_inf)[0],4)

def _compute_bakry_emery_edges(G: nx.Graph, edge_weight='weight', vertex_weight='weight'):
    """Compute the Bakry-Émery curvature of a Graph

    Args:
        G (nx.Graph): A NetworkX graph
        edge_weight (str, optional): The edge weight used for the computations. (Default = 'weight')
        vertex_weight (str, optional): The vertex weight used for the computations. (Defualt = 'weight')

    Returns:
        dict: Dictionary containing the Bakry-Émery curvature of the graph
    """

    # Adjacency matrix
    A = nx.to_numpy_array(G, dtype=float, weight=edge_weight)

    # --- Set global variables for multiprocessing ---
    global _P
    global _P2
    global _d
    global _m
    # ------------------------------------------------

    _P = np.array(A*[[1/G.nodes[v][vertex_weight]] for v in G.nodes()])
    _P2 = _P @ _P
    _d = [np.sum(A[x])for x in range(len(A))]
    _m = [G.nodes[v][vertex_weight] for v in range(len(A))]
     
    with mp.get_context('fork').Pool(mp.cpu_count()) as pool:

        chunksize, extra = divmod(len(A), mp.cpu_count() * 4)
        if extra:
            chunksize += 1

        # Compute Ricci curvature for edges
        result = pool.imap_unordered(_bakry_emery_vertex, range(len(A)), chunksize=chunksize)
        pool.close()
        pool.join()

    be_curv = {}
    for v in result:
        be_curv[v[0]] = v[1]

    return be_curv


class CellComplex:

    """
    A class to compute the first Betti number of a cell complex, arising from a weighted graph by gluing 2-cells to all cycles of length at most five.
    """

    def __init__(self, G: nx.Graph, edge_weight='weight', vertex_weight='weight', init_two_cells=True):
        """
        Initialize the CW-complex from a given weighted graph.

        Parameters:
        graph (nx.Graph): A NetworkX graph representing the 1-skeleton of the CW-complex.
        edge_weight (str, optional): The edge weight used for the computations. (Default = 'weight')
        vertex_weight (str, optional): The vertex weight used for the computations. (Defualt = 'weight')
        init_two_cells (bool, optional): Boolean value indicating whether two cells are initialized.
        """

        self.G = nx.relabel_nodes(G, {node: i for i, node in enumerate(G.nodes())})
        self.X0 = list(self.G.nodes)
        self.X1 = list(self.G.edges)
        self.edge_weight = edge_weight
        self.vertex_weight = vertex_weight
        self.init_two_cells = init_two_cells

        if init_two_cells:
            self.X2 = self._initialize_two_cells()
        self._initialize_weights()

    def _initialize_two_cells(self):
        """
        Initialize the set of 2-cells

        Returns:
        list: A list of cycles, where each cycle is represented as a list of nodes.
        """
        return [c for c in nx.simple_cycles(self.G, 5)]
    
    def _initialize_weights(self):
        """
        Initialize the edge and vertex weights of the graph
        """
        if not nx.get_edge_attributes(self.G, self.edge_weight):
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.edge_weight] = 1.0
        
        if not nx.get_node_attributes(self.G, self.vertex_weight):
            for v in self.G.nodes():
                d = self.G.degree(v,self.edge_weight)
                if d > 0:
                    self.G.nodes[v][self.vertex_weight] = d
                else:
                    raise ValueError(f"Vertex {v} has degree 0, which is not allowed.")      

    def _delta1(self):
        """
        Compute the coboundary operator $\delta_1:C(X_0) \to C(X_1)$ as a matrix.

        Returns:
        np.ndarray: A matrix representing the coboundary operator, where rows correspond to edges
                    and columns correspond to vertices.
        """
        delta1 = np.zeros((len(self.X1), len(self.X0)))
        for j, (v1, v2) in enumerate(self.X1):
            i1 = self.X0.index(v1)
            i2 = self.X0.index(v2)
            delta1[j, i1] = 1
            delta1[j, i2] = -1
        
        return delta1
    

    def _delta1_star(self):
        """
        Compute the coboundary operator $\delta_1^*:C(X_1) \to C(X_0)$ as a matrix.

        Returns:
        np.ndarray: A matrix representing the coboundary operator, where rows correspond to vertices
                    and columns correspond to edges.
        """

        delta1_star = np.zeros((len(self.X0), len(self.X1)))

        for j, (v1,v2) in enumerate(self.X1):
            i1 = self.X0.index(v1)
            i2 = self.X0.index(v2)
            delta1_star[i1,j] = self.G[v1][v2][self.edge_weight] / self.G.nodes[i1][self.vertex_weight]
            delta1_star[i2,j] = - self.G[v1][v2][self.edge_weight] / self.G.nodes[i1][self.vertex_weight]

        return delta1_star
    
    def _delta2(self):
        """
        Compute the coboundary operator $\delta_2: C(X_1) \to C(X_2)$ as a matrix.

        Returns:
        np.ndarray: A matrix representing the coboundary operator, where rows correspond to cycles
                    and columns correspond to edges.
        """

        delta2 = np.zeros((len(self.X2), len(self.X1)))
        for j, e in enumerate(self.X1):
            for k, c in enumerate(self.X2):
                delta2[k,j] = edge_in_cycle(edge=e,cycle=c)
        
        return delta2
    
    def _delta2_star(self):
        """
        Compute the coboundary operator $\delta_2^*: C(X_2) \to C(X_1)$ as a matrix.

        Returns:
        np.ndarray: A matrix representing the coboundary operator, where rows correspond to edges
                    and columns correspond to cycles.
        """
        delta2_star = np.zeros((len(self.X1), len(self.X2)))

        for j, (v1,v2) in enumerate(self.X1):
            for k, c in enumerate(self.X2):
                delta2_star[j,k] = 1/self.G[v1][v2][self.edge_weight] * edge_in_cycle(edge=(v1,v2),cycle=c)
        
        return delta2_star

    def hodge_laplacian(self):
        """
        Compute the Hodge-Laplacian of the graph

        Returns:
        np.ndarray: A matrix representing the Hodge-Laplacian operator
        """
        if self.init_two_cells:
            return self._delta1() @ self._delta1_star() + self._delta2_star() @ self._delta2()
        else:
            raise ValueError(f"Init_two_cells must be True for this function")
    
    def betti_num(self):
        """
        Compute the first Betti-number of the cell complex. The first Betti-number is the dimension of the kernel of the Hodge-Laplacian

        Returns:
            int: Betti-number
        """

        if self.init_two_cells:
            rank = np.linalg.matrix_rank(self.hodge_laplacian())
        else:
            raise ValueError(f"Init_two_cells must be True for this function")
         
        return len(self.X1) - rank
    
    
    def bakry_emery_curvature(self):
        """Compute the Bakry-Émery curvature of the graph

        Returns:
            dict: Dictionary containing the Bakry-Émery curvature of the graph
        """
        be_curv = _compute_bakry_emery_edges(G=self.G, edge_weight=self.edge_weight, vertex_weight=self.vertex_weight)
        return be_curv
            

    def __repr__(self):
        return f"CWComplex(num_vertices={len(self.X0)}, num_edges={len(self.X1)}, num_two_cells={len(self.X2)})"
