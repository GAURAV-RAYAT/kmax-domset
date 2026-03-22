import heapq
from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt

class KMaxAlgorithm:

    def __init__(self, adj_matrix):
        self.A = adj_matrix
        self.n = len(adj_matrix)
        self.graph = self.build_graph()

    # -----------------------------------
    # Build adjacency list
    # -----------------------------------
    def build_graph(self):
        graph = defaultdict(list)
        for i in range(self.n):
            graph[i]  # ensure all nodes, including isolated ones, are present
            for j in range(self.n):
                if self.A[i][j] == 1:
                    graph[i].append(j)
        return graph

    # -----------------------------------
    # Compute degree in original graph
    # -----------------------------------
    def compute_deg_G(self):
        return {v: len(self.graph[v]) for v in self.graph}

    # -----------------------------------
    # PHASE 1: K-max spanning tree
    # -----------------------------------
    def construct_kmax_tree(self, deg_G):
        visited = set()
        T = []

        # Start from the highest-degree node instead of hardcoding node 0
        start = max(deg_G, key=deg_G.get) if deg_G else 0
        pq = [(-deg_G.get(start, 0), start, -1)]

        while len(visited) < self.n:
            # If the PQ is exhausted, the graph is disconnected — resume from
            # an unvisited node to build a spanning forest
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

    # -----------------------------------
    # Compute degree in tree
    # -----------------------------------
    def compute_deg_T(self, T):
        deg_T = defaultdict(int)
        for u, v in T:
            deg_T[u] += 1
            deg_T[v] += 1
        return deg_T

    # -----------------------------------
    # PHASE 2: Fundamental cut-sets
    # -----------------------------------
    def compute_cutsets(self, T):
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
            visited = set()
            components = []

            for v in self.graph:
                if v not in visited:
                    queue = deque([v])
                    comp = []
                    visited.add(v)

                    while queue:
                        x = queue.popleft()
                        comp.append(x)

                        for nei in tree_adj[x]:
                            if (x, nei) == remove_edge or (nei, x) == remove_edge:
                                continue
                            if nei not in visited:
                                visited.add(nei)
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

    # -----------------------------------
    # Compute Karci centrality
    # -----------------------------------
    def compute_karci(self, deg_G, deg_T, cut_sizes):
        K = {v: deg_G[v] + deg_T[v] for v in deg_G}

        for (u, v), size in cut_sizes.items():
            K[u] += size
            K[v] += size

        return K

    # -----------------------------------
    # PHASE 3: Iterative selection
    # -----------------------------------
    def dominating_set(self):
        dominated = set()
        D = set()
        all_nodes = set(range(self.n))

        while dominated != all_nodes:

            deg_G = self.compute_deg_G()
            T = self.construct_kmax_tree(deg_G)
            deg_T = self.compute_deg_T(T)
            cut_sizes = self.compute_cutsets(T)
            K = self.compute_karci(deg_G, deg_T, cut_sizes)

            # Select the highest-centrality node among those not yet dominated.
            # Restricting to undominated nodes ensures dominated grows every
            # iteration, guaranteeing termination.
            undominated = all_nodes - dominated
            v = max(undominated, key=lambda x: K.get(x, 0))

            D.add(v)
            dominated.add(v)
            dominated.update(self.graph[v])

        return D

    # -----------------------------------
    # Visualize graph + dominating set
    # -----------------------------------
    def visualize(self, D=None, layout="spring"):
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


# -----------------------------------
# Run directly: python algo.py
# -----------------------------------
if __name__ == "__main__":
    adj = [
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
    ]

    algo = KMaxAlgorithm(adj)
    D = algo.dominating_set()
    print("Dominating set:", sorted(D))
    algo.visualize(D)