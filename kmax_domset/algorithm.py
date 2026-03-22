import heapq
from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt


class KMaxAlgorithm:
    """
    K-Max algorithm for computing a dominating set of an undirected graph.

    The algorithm uses Karci centrality — combining degree in the original
    graph, degree in a K-max spanning tree, and fundamental cut-set sizes —
    to greedily select the most "central" nodes into the dominating set.

    Parameters
    ----------
    adj_matrix : list[list[int]]
        Square symmetric adjacency matrix.  ``adj_matrix[i][j] == 1`` means
        there is an edge between node *i* and node *j*.  Diagonal must be 0.

    Examples
    --------
    >>> from kmax_domset import KMaxAlgorithm
    >>> adj = [
    ...     [0, 1, 0, 0, 1],
    ...     [1, 0, 1, 0, 0],
    ...     [0, 1, 0, 1, 0],
    ...     [0, 0, 1, 0, 1],
    ...     [1, 0, 0, 1, 0],
    ... ]
    >>> algo = KMaxAlgorithm(adj)
    >>> sorted(algo.dominating_set())
    [0, 2]
    """

    def __init__(self, adj_matrix):
        self.A = adj_matrix
        self.n = len(adj_matrix)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Internal graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        graph = defaultdict(list)
        for i in range(self.n):
            graph[i]  # ensure isolated nodes are always present
            for j in range(self.n):
                if self.A[i][j] == 1:
                    graph[i].append(j)
        return graph

    # ------------------------------------------------------------------
    # Degree in original graph
    # ------------------------------------------------------------------

    def _compute_deg_G(self):
        return {v: len(self.graph[v]) for v in self.graph}

    # ------------------------------------------------------------------
    # PHASE 1 — K-max spanning tree (forest for disconnected graphs)
    # ------------------------------------------------------------------

    def _construct_kmax_tree(self, deg_G):
        visited = set()
        T = []

        start = max(deg_G, key=deg_G.get) if deg_G else 0
        pq = [(-deg_G.get(start, 0), start, -1)]

        while len(visited) < self.n:
            if not pq:
                for node in range(self.n):
                    if node not in visited:
                        heapq.heappush(pq, (-deg_G.get(node, 0), node, -1))
                        break

            _, node, parent = heapq.heappop(pq)

            if node in visited:
                continue

            visited.add(node)

            if parent != -1:
                T.append((parent, node))

            for nei in self.graph[node]:
                if nei not in visited:
                    heapq.heappush(pq, (-deg_G.get(nei, 0), nei, node))

        return T

    # ------------------------------------------------------------------
    # Degree in tree
    # ------------------------------------------------------------------

    def _compute_deg_T(self, T):
        deg_T = defaultdict(int)
        for u, v in T:
            deg_T[u] += 1
            deg_T[v] += 1
        return deg_T

    # ------------------------------------------------------------------
    # PHASE 2 — Fundamental cut-sets
    # ------------------------------------------------------------------

    def _compute_cutsets(self, T):
        tree_adj = defaultdict(list)
        for u, v in T:
            tree_adj[u].append(v)
            tree_adj[v].append(u)

        branches = set(T)
        chords = set()

        for u in self.graph:
            for v in self.graph[u]:
                if (u, v) not in branches and (v, u) not in branches:
                    if u < v:
                        chords.add((u, v))

        cut_sizes = {}

        def bfs_components(remove_edge):
            seen = set()
            components = []
            for v in self.graph:
                if v not in seen:
                    queue = deque([v])
                    comp = []
                    seen.add(v)
                    while queue:
                        x = queue.popleft()
                        comp.append(x)
                        for nei in tree_adj[x]:
                            if (x, nei) == remove_edge or (nei, x) == remove_edge:
                                continue
                            if nei not in seen:
                                seen.add(nei)
                                queue.append(nei)
                    components.append(comp)
            return components

        for b in branches:
            comps = bfs_components(b)
            if len(comps) != 2:
                continue
            V1, V2 = set(comps[0]), set(comps[1])
            size = 1
            for x, y in chords:
                if (x in V1 and y in V2) or (x in V2 and y in V1):
                    size += 1
            cut_sizes[b] = size

        return cut_sizes

    # ------------------------------------------------------------------
    # Karci centrality
    # ------------------------------------------------------------------

    def _compute_karci(self, deg_G, deg_T, cut_sizes):
        K = {v: deg_G[v] + deg_T[v] for v in deg_G}
        for (u, v), size in cut_sizes.items():
            K[u] += size
            K[v] += size
        return K

    # ------------------------------------------------------------------
    # PHASE 3 — Iterative dominating-set selection
    # ------------------------------------------------------------------

    def dominating_set(self):
        """
        Compute a dominating set of the graph using the K-Max algorithm.

        Returns
        -------
        set[int]
            A set of node indices that form a dominating set.  Every node in
            the graph either belongs to this set or is adjacent to a node in
            this set.
        """
        dominated = set()
        D = set()
        all_nodes = set(range(self.n))

        while dominated != all_nodes:
            deg_G = self._compute_deg_G()
            T = self._construct_kmax_tree(deg_G)
            deg_T = self._compute_deg_T(T)
            cut_sizes = self._compute_cutsets(T)
            K = self._compute_karci(deg_G, deg_T, cut_sizes)

            undominated = all_nodes - dominated
            v = max(undominated, key=lambda x: K.get(x, 0))

            D.add(v)
            dominated.add(v)
            dominated.update(self.graph[v])

        return D

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(self, D=None, layout="spring"):
        """
        Draw the graph and highlight the dominating set.

        Parameters
        ----------
        D : set[int] | None
            Dominating set to highlight.  If *None*, ``dominating_set()`` is
            called automatically.
        layout : str
            One of ``"spring"`` (default), ``"circular"``, ``"kamada"``,
            ``"spectral"``.
        """
        if D is None:
            D = self.dominating_set()

        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        for u in range(self.n):
            for v in range(u + 1, self.n):
                if self.A[u][v] == 1:
                    G.add_edge(u, v)

        layouts = {
            "spring":   nx.spring_layout(G, seed=42),
            "circular": nx.circular_layout(G),
            "kamada":   nx.kamada_kawai_layout(G),
            "spectral": nx.spectral_layout(G),
        }
        pos = layouts.get(layout, nx.spring_layout(G, seed=42))
        colors = ["red" if node in D else "lightblue" for node in G.nodes()]

        plt.figure(figsize=(8, 6), layout="constrained")
        nx.draw(G, pos, with_labels=True, node_color=colors,
                node_size=800, font_size=11, font_weight="bold",
                edge_color="gray", width=1.5)
        plt.title(f"Dominating Set (red nodes): {sorted(D)}  |  size = {len(D)}")
        plt.show()
