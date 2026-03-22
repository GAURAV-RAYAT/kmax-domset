"""
Comprehensive pytest suite for KMaxAlgorithm.dominating_set()

For every graph the returned set D must satisfy:
    ∀ v ∈ V,   v ∈ D   OR   ∃ u ∈ D : (u, v) ∈ E
"""

import random
import pytest
from algo import KMaxAlgorithm


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATOR & RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_ds(adj, D):
    n = len(adj)
    covered = set(D)
    for d in D:
        for u in range(n):
            if adj[d][u] == 1:
                covered.add(u)
    return covered == set(range(n))


def run(adj):
    n = len(adj)
    D = KMaxAlgorithm(adj).dominating_set()
    assert isinstance(D, set), "must return a set"
    assert D.issubset(set(range(n))), f"D has out-of-range nodes: {D - set(range(n))}"
    assert is_valid_ds(adj, D), f"D={sorted(D)} is NOT a valid dominating set (n={n})"
    return D


# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH FACTORY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _m(n):
    return [[0] * n for _ in range(n)]


def _e(m, u, v):
    m[u][v] = m[v][u] = 1


def path_graph(n):
    m = _m(n)
    for i in range(n - 1):
        _e(m, i, i + 1)
    return m


def cycle_graph(n):
    m = path_graph(n)
    _e(m, 0, n - 1)
    return m


def complete_graph(n):
    m = _m(n)
    for i in range(n):
        for j in range(i + 1, n):
            _e(m, i, j)
    return m


def star_graph(n):
    m = _m(n)
    for i in range(1, n):
        _e(m, 0, i)
    return m


def wheel_graph(spokes):
    n = spokes + 1
    m = _m(n)
    for i in range(1, spokes):
        _e(m, i, i + 1)
    _e(m, 1, spokes)
    for i in range(1, n):
        _e(m, 0, i)
    return m


def grid_graph(rows, cols):
    n = rows * cols
    m = _m(n)
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                _e(m, v, v + 1)
            if r + 1 < rows:
                _e(m, v, v + cols)
    return m


def hypercube_graph(d):
    n = 2 ** d
    m = _m(n)
    for i in range(n):
        for b in range(d):
            j = i ^ (1 << b)
            if i < j:
                _e(m, i, j)
    return m


def petersen_graph():
    m = _m(10)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                 (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
                 (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]:
        _e(m, u, v)
    return m


def balanced_binary_tree(depth):
    n = 2 ** (depth + 1) - 1
    m = _m(n)
    for i in range(n):
        l, r = 2 * i + 1, 2 * i + 2
        if l < n:
            _e(m, i, l)
        if r < n:
            _e(m, i, r)
    return m


def complete_bipartite(a, b):
    n = a + b
    m = _m(n)
    for i in range(a):
        for j in range(a, n):
            _e(m, i, j)
    return m


def lollipop(k, tail):
    n = k + tail
    m = _m(n)
    for i in range(k):
        for j in range(i + 1, k):
            _e(m, i, j)
    _e(m, k - 1, k)
    for i in range(k, k + tail - 1):
        _e(m, i, i + 1)
    return m


def barbell(k):
    n = 2 * k
    m = _m(n)
    for i in range(k):
        for j in range(i + 1, k):
            _e(m, i, j)
    for i in range(k, 2 * k):
        for j in range(i + 1, 2 * k):
            _e(m, i, j)
    _e(m, 0, k)
    return m


def friendship(k):
    n = 2 * k + 1
    m = _m(n)
    for i in range(k):
        a, b = 2 * i + 1, 2 * i + 2
        _e(m, 0, a)
        _e(m, 0, b)
        _e(m, a, b)
    return m


def caterpillar(spine, leaves):
    n = spine + spine * leaves
    m = _m(n)
    for i in range(spine - 1):
        _e(m, i, i + 1)
    idx = spine
    for i in range(spine):
        for _ in range(leaves):
            _e(m, i, idx)
            idx += 1
    return m


def ladder(n):
    total = 2 * n
    m = _m(total)
    for i in range(n - 1):
        _e(m, i, i + 1)
        _e(m, n + i, n + i + 1)
    for i in range(n):
        _e(m, i, n + i)
    return m


def circulant(n, *offsets):
    m = _m(n)
    for i in range(n):
        for off in offsets:
            j = (i + off) % n
            if i != j:
                _e(m, i, j)
    return m


def ring_of_cliques(k, cs):
    n = k * cs
    m = _m(n)
    for c in range(k):
        base = c * cs
        for i in range(cs):
            for j in range(i + 1, cs):
                _e(m, base + i, base + j)
    for c in range(k):
        _e(m, c * cs, ((c + 1) % k) * cs)
    return m


def compose(*mats):
    total = sum(len(c) for c in mats)
    m = _m(total)
    off = 0
    for comp in mats:
        nc = len(comp)
        for i in range(nc):
            for j in range(nc):
                m[off + i][off + j] = comp[i][j]
        off += nc
    return m


def rng_graph(n, p, seed):
    """Random graph with a spanning path to avoid full isolation."""
    rng = random.Random(seed)
    m = _m(n)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                _e(m, i, j)
    for i in range(n - 1):
        _e(m, i, i + 1)
    return m


def rng_graph_sparse_isolated(n, p, seed):
    """Random graph that may contain isolated nodes."""
    rng = random.Random(seed)
    m = _m(n)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                _e(m, i, j)
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD THE PARAMETRISED TEST MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def _collect():
    cases = []

    # ── Trivial / degenerate ──────────────────────────────────────────────────
    cases += [
        ("single_node",          [[0]]),
        ("2_connected",          [[0, 1], [1, 0]]),
        ("2_disconnected",       [[0, 0], [0, 0]]),
        ("3_triangle",           complete_graph(3)),
        ("3_path",               path_graph(3)),
        ("3_all_isolated",       _m(3)),
        ("3_one_edge_01",        [[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        ("3_star",               star_graph(3)),
        ("4_complete",           complete_graph(4)),
        ("4_path",               path_graph(4)),
        ("4_cycle",              cycle_graph(4)),
        ("4_star",               star_graph(4)),
        ("4_all_isolated",       _m(4)),
        ("5_all_isolated",       _m(5)),
        ("10_all_isolated",      _m(10)),
    ]

    # ── Paths P_n ─────────────────────────────────────────────────────────────
    for n in list(range(2, 31)) + [35, 40, 45, 50, 55, 60]:
        cases.append((f"path_{n}", path_graph(n)))

    # ── Cycles C_n ────────────────────────────────────────────────────────────
    for n in list(range(3, 31)) + [35, 40, 45, 50, 55, 60]:
        cases.append((f"cycle_{n}", cycle_graph(n)))

    # ── Complete K_n ──────────────────────────────────────────────────────────
    for n in list(range(1, 21)) + [25, 30, 40, 50]:
        cases.append((f"complete_{n}", complete_graph(n)))

    # ── Stars S_n ─────────────────────────────────────────────────────────────
    for n in list(range(2, 21)) + [25, 30, 40, 50, 51]:
        cases.append((f"star_{n}", star_graph(n)))

    # ── Wheels W_n ────────────────────────────────────────────────────────────
    for spokes in list(range(3, 21)) + [25, 30, 49]:
        cases.append((f"wheel_{spokes}", wheel_graph(spokes)))

    # ── Grids ─────────────────────────────────────────────────────────────────
    for r, c in [
        (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),
        (2, 15), (2, 20), (2, 25),
        (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 10),
        (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 10),
        (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
        (6, 6), (6, 7), (6, 8),
        (7, 7), (7, 8),
        (8, 8),
    ]:
        cases.append((f"grid_{r}x{c}", grid_graph(r, c)))

    # ── Hypercubes Q_d ────────────────────────────────────────────────────────
    for d in range(1, 7):
        cases.append((f"hypercube_Q{d}", hypercube_graph(d)))

    # ── Petersen ──────────────────────────────────────────────────────────────
    cases.append(("petersen", petersen_graph()))

    # ── Balanced binary trees ─────────────────────────────────────────────────
    for depth in range(1, 7):
        cases.append((f"bin_tree_d{depth}", balanced_binary_tree(depth)))

    # ── Complete bipartite K_{a,b} ────────────────────────────────────────────
    for a, b in [
        (1, 1), (1, 2), (1, 5), (1, 10), (1, 20),
        (2, 2), (2, 3), (2, 5), (2, 8),
        (3, 3), (3, 4), (3, 5), (3, 7),
        (4, 4), (4, 5), (4, 6),
        (5, 5), (5, 6), (5, 7),
        (6, 6), (6, 7),
        (7, 7), (7, 8),
        (8, 8), (9, 9),
        (10, 10), (10, 15), (10, 20),
        (15, 15), (15, 20),
        (20, 20), (20, 30),
        (25, 25),
    ]:
        cases.append((f"bipartite_{a}x{b}", complete_bipartite(a, b)))

    # ── Lollipop  ─────────────────────────────────────────────────────────────
    for k, t in [
        (3, 3), (3, 5), (3, 10), (3, 20),
        (4, 4), (4, 8), (4, 15),
        (5, 5), (5, 10), (5, 20),
        (8, 8), (8, 12),
        (10, 5), (10, 10), (10, 20),
        (15, 15), (20, 3),
    ]:
        cases.append((f"lollipop_k{k}_t{t}", lollipop(k, t)))

    # ── Barbell ───────────────────────────────────────────────────────────────
    for k in list(range(2, 16)) + [18, 20, 25]:
        cases.append((f"barbell_{k}", barbell(k)))

    # ── Friendship ────────────────────────────────────────────────────────────
    for k in list(range(1, 16)) + [18, 20, 24]:
        cases.append((f"friendship_{k}", friendship(k)))

    # ── Caterpillar ───────────────────────────────────────────────────────────
    for s, l in [
        (3, 1), (3, 2), (3, 3),
        (4, 2), (4, 3), (4, 4),
        (5, 2), (5, 3), (5, 4),
        (6, 2), (6, 3),
        (8, 3), (8, 4),
        (10, 3), (10, 4),
        (12, 3),
    ]:
        cases.append((f"caterpillar_s{s}_l{l}", caterpillar(s, l)))

    # ── Ladder ────────────────────────────────────────────────────────────────
    for n in list(range(2, 16)) + [18, 20, 25]:
        cases.append((f"ladder_{n}", ladder(n)))

    # ── Circulant ─────────────────────────────────────────────────────────────
    for n, offs in [
        (6,  (1, 2)), (8,  (1, 3)), (10, (1, 2)), (10, (1, 4)),
        (12, (1, 4)), (12, (1, 5)),
        (15, (1, 2)), (15, (1, 5)), (15, (1, 6)),
        (20, (1, 2)), (20, (1, 3)), (20, (1, 7)), (20, (2, 5)),
        (25, (1, 4)), (25, (2, 7)),
        (30, (1, 5)), (30, (2, 7)), (30, (3, 9)),
        (40, (1, 6)), (40, (2, 11)),
        (50, (1, 7)), (50, (3, 13)), (50, (5, 17)),
    ]:
        cases.append((f"circulant_{n}_off{offs}", circulant(n, *offs)))

    # ── Ring of cliques ───────────────────────────────────────────────────────
    for k, cs in [
        (3, 3), (4, 3), (5, 3),
        (3, 4), (4, 4), (5, 4), (6, 4),
        (4, 5), (5, 5), (6, 5),
        (4, 6), (5, 6),
        (3, 8), (4, 8),
        (3, 10), (5, 10),
    ]:
        cases.append((f"ring_cliques_{k}x{cs}", ring_of_cliques(k, cs)))

    # ── Disconnected graphs ───────────────────────────────────────────────────
    cases += [
        ("dis_2_paths",          compose(path_graph(5), path_graph(5))),
        ("dis_3_paths",          compose(path_graph(4), path_graph(5), path_graph(6))),
        ("dis_clique_path",      compose(complete_graph(5), path_graph(7))),
        ("dis_3_cycles",         compose(cycle_graph(4), cycle_graph(5), cycle_graph(6))),
        ("dis_star_grid",        compose(star_graph(6), grid_graph(3, 3))),
        ("dis_two_k10",          compose(complete_graph(10), complete_graph(10))),
        ("dis_k5_k10_k5",        compose(complete_graph(5), complete_graph(10), complete_graph(5))),
        ("dis_3_trees",          compose(balanced_binary_tree(2), balanced_binary_tree(3), balanced_binary_tree(2))),
        ("dis_single_k5",        compose([[0]], complete_graph(5))),
        ("dis_5_singles",        _m(5)),
        ("dis_path5_iso5",       compose(path_graph(5), _m(5))),
        ("dis_wheel_star",       compose(wheel_graph(5), star_graph(6))),
        ("dis_bipartite_cycle",  compose(complete_bipartite(4, 4), cycle_graph(7))),
        ("dis_grid_hypercube",   compose(grid_graph(3, 3), hypercube_graph(3))),
        ("dis_two_grids",        compose(grid_graph(4, 4), grid_graph(3, 3))),
        ("dis_friends_barbell",  compose(friendship(3), barbell(4))),
        ("dis_lollipop_ladder",  compose(lollipop(4, 5), ladder(5))),
        ("dis_cat_ring",         compose(caterpillar(5, 2), ring_of_cliques(3, 3))),
    ]
    for i in range(1, 6):
        cases.append((f"dis_{i}singles_path10",
                      compose(_m(i), path_graph(10))))

    # ── Random sparse (p = 0.10) ──────────────────────────────────────────────
    for n in [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]:
        for seed in range(10):
            cases.append((f"rand_sparse_n{n}_s{seed}", rng_graph(n, 0.10, seed)))

    # ── Random medium (p = 0.30) ──────────────────────────────────────────────
    for n in [5, 8, 10, 15, 20, 25, 30, 40, 50]:
        for seed in range(10):
            cases.append((f"rand_medium_n{n}_s{seed}", rng_graph(n, 0.30, seed)))

    # ── Random dense (p = 0.60) ───────────────────────────────────────────────
    for n in [5, 10, 15, 20, 30, 40, 50]:
        for seed in range(10):
            cases.append((f"rand_dense_n{n}_s{seed}", rng_graph(n, 0.60, seed)))

    # ── Random very dense (p = 0.85) ─────────────────────────────────────────
    for n in [10, 20, 30, 50]:
        for seed in range(8):
            cases.append((f"rand_vdense_n{n}_s{seed}", rng_graph(n, 0.85, seed)))

    # ── Random with possible isolated nodes ───────────────────────────────────
    for n in [10, 15, 20, 25, 30]:
        for seed in range(8):
            cases.append((f"rand_iso_n{n}_s{seed}",
                          rng_graph_sparse_isolated(n, 0.12, seed + 500)))

    # ── Specifically 50-node / high-degree graphs ─────────────────────────────
    cases += [
        ("50_complete",           complete_graph(50)),
        ("50_path",               path_graph(50)),
        ("50_cycle",              cycle_graph(50)),
        ("50_star_51",            star_graph(51)),
        ("50_grid_5x10",          grid_graph(5, 10)),
        ("50_grid_2x25",          grid_graph(2, 25)),
        ("50_bipartite_25x25",    complete_bipartite(25, 25)),
        ("50_circulant_1_7",      circulant(50, 1, 7)),
        ("50_circulant_1_13",     circulant(50, 1, 13)),
        ("50_circulant_5_17",     circulant(50, 5, 17)),
        ("50_ladder_25",          ladder(25)),
        ("50_friendship_24",      friendship(24)),
        ("50_caterpillar_10x4",   caterpillar(10, 4)),
        ("50_barbell_25",         barbell(25)),
        ("50_lollipop_25_25",     lollipop(25, 25)),
        ("50_ring_cliques_10x5",  ring_of_cliques(10, 5)),
        ("50_ring_cliques_5x10",  ring_of_cliques(5, 10)),
        ("55_path",               path_graph(55)),
        ("55_cycle",              cycle_graph(55)),
        ("60_complete",           complete_graph(60)),
        ("60_grid_6x10",          grid_graph(6, 10)),
        ("64_hypercube_Q6",       hypercube_graph(6)),
        ("63_bin_tree_d5",        balanced_binary_tree(5)),
        ("60_ring_cliques_6x10",  ring_of_cliques(6, 10)),
        ("60_circulant_1_11",     circulant(60, 1, 11)),
        ("60_bipartite_30x30",    complete_bipartite(30, 30)),
        ("50_dis_two_25paths",    compose(path_graph(25), path_graph(25))),
        ("50_dis_k25_k25",        compose(complete_graph(25), complete_graph(25))),
    ]

    for seed in range(20):
        cases.append((f"50_rand_sparse_s{seed}",  rng_graph(50, 0.10, seed + 1000)))
        cases.append((f"50_rand_medium_s{seed}",  rng_graph(50, 0.30, seed + 2000)))
        cases.append((f"50_rand_dense_s{seed}",   rng_graph(50, 0.60, seed + 3000)))
        cases.append((f"50_rand_vdense_s{seed}",  rng_graph(50, 0.85, seed + 4000)))

    for seed in range(10):
        cases.append((f"60_rand_medium_s{seed}",  rng_graph(60, 0.30, seed + 5000)))
        cases.append((f"60_rand_dense_s{seed}",   rng_graph(60, 0.60, seed + 6000)))

    return cases


GRAPHS = _collect()


# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETRISED BULK TEST
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,adj", GRAPHS, ids=[c[0] for c in GRAPHS])
def test_dominating_set_valid(name, adj):
    run(adj)


# ─────────────────────────────────────────────────────────────────────────────
#  KNOWN-OPTIMUM TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestKnownOptima:

    def test_single_node_ds_is_itself(self):
        assert run([[0]]) == {0}

    def test_two_connected_ds_size_1(self):
        D = run([[0, 1], [1, 0]])
        assert len(D) == 1

    def test_two_isolated_both_required(self):
        D = run([[0, 0], [0, 0]])
        assert D == {0, 1}

    def test_triangle_ds_size_1(self):
        assert len(run(complete_graph(3))) == 1

    def test_complete_k5_ds_size_1(self):
        assert len(run(complete_graph(5))) == 1

    def test_complete_k10_ds_size_1(self):
        assert len(run(complete_graph(10))) == 1

    def test_complete_k20_ds_size_1(self):
        assert len(run(complete_graph(20))) == 1

    def test_complete_k50_ds_size_1(self):
        assert len(run(complete_graph(50))) == 1

    def test_star_center_alone_dominates(self):
        # Only the center (node 0) can be a size-1 DS
        D = run(star_graph(10))
        assert len(D) == 1

    def test_star_50_leaves_ds_size_1(self):
        D = run(star_graph(51))
        assert len(D) == 1

    def test_wheel_hub_alone_dominates(self):
        D = run(wheel_graph(6))
        assert len(D) == 1

    def test_wheel_large_hub_alone_dominates(self):
        D = run(wheel_graph(30))
        assert len(D) == 1

    def test_all_isolated_3_requires_all(self):
        D = run(_m(3))
        assert D == {0, 1, 2}

    def test_all_isolated_7_requires_all(self):
        D = run(_m(7))
        assert D == set(range(7))

    def test_friendship_center_dominates(self):
        # Center (node 0) is adjacent to every other node
        D = run(friendship(5))
        assert len(D) == 1

    def test_friendship_large_center_dominates(self):
        D = run(friendship(20))
        assert len(D) == 1

    def test_k_bipartite_1_n_size_1(self):
        for n in [2, 5, 10, 20]:
            D = run(complete_bipartite(1, n))
            assert len(D) == 1, f"K_1,{n} should have DS of size 1"

    def test_path_3_ds_size_1(self):
        # Middle node (1) dominates all
        D = run(path_graph(3))
        assert len(D) == 1

    def test_complete_bipartite_25x25_valid(self):
        run(complete_bipartite(25, 25))

    def test_barbell_bridge_role(self):
        adj = barbell(5)
        D = run(adj)
        assert is_valid_ds(adj, D)


# ─────────────────────────────────────────────────────────────────────────────
#  RETURN-TYPE / CONTRACT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestContract:

    def test_returns_set(self):
        assert isinstance(KMaxAlgorithm(complete_graph(5)).dominating_set(), set)

    def test_deterministic_same_input(self):
        adj = rng_graph(20, 0.4, 42)
        D1 = KMaxAlgorithm(adj).dominating_set()
        D2 = KMaxAlgorithm(adj).dominating_set()
        assert D1 == D2

    def test_all_nodes_in_valid_range_small(self):
        n = 8
        adj = rng_graph(n, 0.3, 7)
        D = KMaxAlgorithm(adj).dominating_set()
        assert all(0 <= v < n for v in D)

    def test_all_nodes_in_valid_range_large(self):
        n = 50
        adj = rng_graph(n, 0.3, 77)
        D = KMaxAlgorithm(adj).dominating_set()
        assert all(0 <= v < n for v in D)

    def test_non_empty_for_nonempty_graph(self):
        for n in [1, 2, 5, 10, 20]:
            D = KMaxAlgorithm(complete_graph(n)).dominating_set()
            assert len(D) >= 1


# ─────────────────────────────────────────────────────────────────────────────
#  SPECIFIC GRAPH STRUCTURE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSpecificStructures:

    def test_petersen(self):
        run(petersen_graph())

    def test_hypercube_q3(self):
        run(hypercube_graph(3))

    def test_hypercube_q4(self):
        run(hypercube_graph(4))

    def test_hypercube_q5(self):
        run(hypercube_graph(5))

    def test_hypercube_q6(self):
        run(hypercube_graph(6))

    def test_grid_5x10(self):
        run(grid_graph(5, 10))

    def test_grid_7x7(self):
        run(grid_graph(7, 7))

    def test_grid_8x8(self):
        run(grid_graph(8, 8))

    def test_caterpillar_heavy(self):
        run(caterpillar(12, 4))

    def test_ring_of_cliques_large(self):
        run(ring_of_cliques(5, 10))

    def test_lollipop_long_tail(self):
        run(lollipop(5, 40))

    def test_barbell_large(self):
        run(barbell(25))

    def test_circulant_50_wide(self):
        run(circulant(50, 1, 5, 11))

    def test_binary_tree_depth6(self):
        run(balanced_binary_tree(6))

    def test_disconnected_many_components(self):
        adj = compose(
            complete_graph(5),
            cycle_graph(6),
            path_graph(7),
            star_graph(8),
            _m(3),
        )
        run(adj)

    def test_disconnected_all_isolated_20(self):
        D = run(_m(20))
        assert D == set(range(20))

    def test_disconnected_k10_plus_k10(self):
        run(compose(complete_graph(10), complete_graph(10)))

    def test_path_50_valid(self):
        run(path_graph(50))

    def test_cycle_50_valid(self):
        run(cycle_graph(50))

    def test_complete_50_valid(self):
        run(complete_graph(50))

    # Exhaustive check over all paths P_2 .. P_20
    @pytest.mark.parametrize("n", range(2, 21))
    def test_all_paths(self, n):
        run(path_graph(n))

    # Exhaustive check over all cycles C_3 .. C_20
    @pytest.mark.parametrize("n", range(3, 21))
    def test_all_cycles(self, n):
        run(cycle_graph(n))

    # Exhaustive check over all complete graphs K_1 .. K_15
    @pytest.mark.parametrize("n", range(1, 16))
    def test_all_complete(self, n):
        run(complete_graph(n))

    # Exhaustive check over all stars S_2 .. S_20
    @pytest.mark.parametrize("n", range(2, 21))
    def test_all_stars(self, n):
        run(star_graph(n))

    # Exhaustive check over wheels W_3 .. W_15
    @pytest.mark.parametrize("spokes", range(3, 16))
    def test_all_wheels(self, spokes):
        run(wheel_graph(spokes))
