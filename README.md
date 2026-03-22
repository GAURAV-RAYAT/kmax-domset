# kmax-domset

**K-Max algorithm for computing dominating sets in undirected graphs.**

A dominating set *D* of a graph *G = (V, E)* is a subset of vertices such
that every vertex in *V* either belongs to *D* or is adjacent to at least one
vertex in *D*.

This package implements the **K-Max algorithm**, which uses *Karci centrality*
— a measure that combines the node degree in the original graph, its degree in
a K-max spanning tree, and the sizes of fundamental cut-sets — to greedily
select the most structurally important nodes into the dominating set.

---

## Installation

```bash
pip install kmax-domset
```

---

## Quick Start

```python
from kmax_domset import KMaxAlgorithm

# Adjacency matrix: 1 = edge, 0 = no edge
adj = [
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0],
]

algo = KMaxAlgorithm(adj)
D = algo.dominating_set()
print("Dominating set:", sorted(D))   # e.g. [0, 2]
```

---

## Visualisation

```python
algo.visualize(D)                        # spring layout (default)
algo.visualize(D, layout="circular")     # circular layout
algo.visualize(D, layout="kamada")       # Kamada-Kawai layout
algo.visualize(D, layout="spectral")     # spectral layout
```

Red nodes are in the dominating set; blue nodes are dominated neighbours.

---

## Building the adjacency matrix

### Manual (small graphs)

```python
n = 6
adj = [[0] * n for _ in range(n)]

edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]
for u, v in edges:
    adj[u][v] = 1
    adj[v][u] = 1
```

### From NetworkX

```python
import networkx as nx
from kmax_domset import KMaxAlgorithm

G = nx.petersen_graph()
n = G.number_of_nodes()
adj = [[0] * n for _ in range(n)]
for u, v in G.edges():
    adj[u][v] = 1
    adj[v][u] = 1

D = KMaxAlgorithm(adj).dominating_set()
print("Dominating set:", sorted(D))
```

---

## API

### `KMaxAlgorithm(adj_matrix)`

| Parameter    | Type                | Description                              |
|-------------|---------------------|------------------------------------------|
| `adj_matrix` | `list[list[int]]`  | Square symmetric adjacency matrix (0/1). |

### `dominating_set() → set[int]`

Returns a set of node indices forming a dominating set.

### `visualize(D=None, layout="spring")`

| Parameter | Type           | Description                                       |
|-----------|----------------|---------------------------------------------------|
| `D`       | `set[int]`     | Dominating set to highlight. Auto-computed if `None`. |
| `layout`  | `str`          | `"spring"` · `"circular"` · `"kamada"` · `"spectral"` |

---

## Algorithm Overview

1. **Phase 1 — K-max spanning tree**: Build a spanning tree (forest for
   disconnected graphs) by greedily selecting the highest-degree unvisited
   node at each step using a max-priority queue.

2. **Phase 2 — Fundamental cut-sets**: For each tree branch (edge), compute
   the size of its fundamental cut-set — the number of graph edges (branches +
   chords) whose removal disconnects the two components formed by removing that
   branch.

3. **Karci centrality**: For each node *v*:  
   `K(v) = deg_G(v) + deg_T(v) + Σ cut_size(e)` for all tree branches *e*
   incident to *v*.

4. **Phase 3 — Greedy selection**: Repeatedly pick the undominated node with
   the highest Karci centrality and add it to the dominating set until all
   nodes are covered.

---

## Requirements

- Python ≥ 3.8
- `networkx >= 2.6`
- `matplotlib >= 3.4`

---

## License

MIT License — see [LICENSE](LICENSE).
