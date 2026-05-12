import random
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix


PACKAGE_ROOT = Path(__file__).resolve().parent
INSTANCE_ROOT = PACKAGE_ROOT / "instance"


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_in_package(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PACKAGE_ROOT / path


def parse_gset(ins_id: int | str, gset_dir: Optional[Path] = None) -> nx.Graph:
    token = str(ins_id).strip()
    if token.lower().endswith(".txt"):
        path = Path(token)
        if not path.is_absolute():
            path = (gset_dir or INSTANCE_ROOT / "Gset") / path.name
    else:
        token = token[1:] if token.upper().startswith("G") else token
        path = (gset_dir or INSTANCE_ROOT / "Gset") / f"G{token}.txt"

    nx_graph = nx.Graph()
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    n, m = map(int, lines[0].split())
    nx_graph.add_nodes_from(range(n))
    if len(lines) != m + 1:
        raise ValueError(f"Invalid Gset file {path}: expected {m} edges, found {len(lines) - 1}.")

    for line in lines[1:]:
        u, v, w = map(int, line.split())
        nx_graph.add_edge(u - 1, v - 1, weight=w)
    return nx_graph


def generate_Max_cut(graph: nx.Graph) -> dict[str, Any]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    Q_values, Q_indices = [], []
    for i, j, w in graph.edges(data=True):
        Q_values.append(w["weight"])
        Q_values.append(w["weight"])
        Q_indices.append([i, j])
        Q_indices.append([j, i])

    Q_values = np.array(Q_values, dtype=np.float32)
    Q_indices = np.array(Q_indices, dtype=np.int32).T

    Q_sparse = coo_matrix((Q_values, Q_indices), shape=(n, n))
    c = -np.array(Q_sparse.sum(axis=1), dtype=np.float32).flatten()
    return {
        "num_nodes": n,
        "num_edges": m,
        "num_vars": n,
        "Q_indices": Q_indices,
        "Q_values": Q_values,
        "c": c,
        "Q_sparse": Q_sparse,
    }


def generate_uBQP_sparse_mc(path: str | Path) -> dict[str, Any]:
    """
    Parse Instances_uBQP *.sparse.mc files into a ShapeaK-compatible min-form QUBO.

    The last header node is treated as an anchored node, matching the conversion
    already used by the existing ShapeaK UBQP script.
    """
    instance_path = Path(path)
    with instance_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty uBQP file: {instance_path}")

    header_tokens = lines[0].split()
    if len(header_tokens) < 2:
        raise ValueError(f"Invalid uBQP header in {instance_path}: {lines[0]}")

    n_header, m = map(int, header_tokens[:2])
    if n_header <= 1:
        raise ValueError(f"Invalid n_header={n_header} in {instance_path}. Expect >= 2.")

    n_vars = n_header - 1
    degree = np.zeros(n_vars, dtype=np.float32)
    q_dict: dict[tuple[int, int], np.float32] = {}
    quad_terms = 0
    anchor_terms = 0

    for ln in lines[1:]:
        toks = ln.split()
        if len(toks) < 3:
            raise ValueError(f"Invalid uBQP term in {instance_path}: {ln}")

        i_raw, j_raw = int(toks[0]), int(toks[1])
        w = float(toks[2])

        if i_raw < 1 or i_raw > n_vars:
            raise ValueError(f"Index i={i_raw} out of range [1, {n_vars}] in {instance_path}.")
        if j_raw < 1 or j_raw > n_header:
            raise ValueError(f"Index j={j_raw} out of range [1, {n_header}] in {instance_path}.")

        i = i_raw - 1
        if j_raw == n_header:
            degree[i] += np.float32(w)
            anchor_terms += 1
            continue

        j = j_raw - 1
        if i == j:
            degree[i] += np.float32(w)
            anchor_terms += 1
            continue

        q_w = np.float32(w)
        q_dict[(i, j)] = q_dict.get((i, j), np.float32(0.0)) + q_w
        q_dict[(j, i)] = q_dict.get((j, i), np.float32(0.0)) + q_w

        degree[i] += q_w
        degree[j] += q_w
        quad_terms += 1

    if len(lines) - 1 != m:
        raise ValueError(
            f"Header mismatch in {instance_path}: header m={m}, but found {len(lines) - 1} terms."
        )

    if q_dict:
        pairs = np.array(list(q_dict.keys()), dtype=np.int32)
        Q_indices = pairs.T
        Q_values = np.array([q_dict[tuple(p)] for p in pairs], dtype=np.float32)
    else:
        Q_indices = np.zeros((2, 0), dtype=np.int32)
        Q_values = np.zeros((0,), dtype=np.float32)

    Q_sparse = coo_matrix((Q_values, Q_indices), shape=(n_vars, n_vars))
    c = -degree

    return {
        "num_nodes": n_vars,
        "num_edges": quad_terms,
        "num_vars": n_vars,
        "Q_indices": Q_indices,
        "Q_values": Q_values,
        "c": c,
        "Q_sparse": Q_sparse,
        "raw_terms": m,
        "anchor_terms": anchor_terms,
        "quadratic_terms": quad_terms,
        "header_n": n_header,
    }


def instance_tag(path: Path) -> str:
    name = path.name
    if name.endswith(".sparse.mc"):
        return name[:-len(".sparse.mc")]
    return path.stem


def project_to_psd(Q_dense: np.ndarray, eps: float = 0.0) -> np.ndarray:
    Q_sym = 0.5 * (Q_dense + Q_dense.T)
    evals, evecs = np.linalg.eigh(Q_sym)
    evals = np.clip(evals, eps, None)
    return (evecs * evals) @ evecs.T


def shapeak_qp_cache_path(dataset: str, name: str, n_vars: int) -> Path:
    cache_dir = PACKAGE_ROOT / "cache" / "shapeak_qp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace("/", "_").replace("\\", "_")
    return cache_dir / f"{dataset}_{safe_name}_n{n_vars}.npy"


def build_preconditioner(
        preQ: int,
        data: dict[str, Any],
        dataset: str,
        name: str,
        cache_override: str = "",
        rebuild: int = 0,
        eps: float = 0.0,
        verbose: bool = True,
):
    if preQ not in {1, 2}:
        return None

    cache_path = resolve_in_package(cache_override.strip()) if cache_override.strip() else shapeak_qp_cache_path(
        dataset, name, data["num_vars"]
    )
    need_build = bool(rebuild) or not cache_path.exists()
    h_precond = None

    if not need_build:
        h_precond = np.load(cache_path, allow_pickle=False).astype(np.float32)
        if h_precond.shape != (data["num_vars"], data["num_vars"]):
            need_build = True
            h_precond = None

    if need_build:
        if verbose:
            print(f"[shapeakadmm] building PSD preconditioner: {cache_path}")
        q_dense = np.asarray(data["Q_sparse"].todense(), dtype=np.float64)
        q_proj = project_to_psd(q_dense, eps=float(eps))
        h_precond = (2.0 * q_proj).astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, h_precond)

    if verbose:
        print(f"[shapeakadmm] using preconditioner cache: {cache_path}")
    return h_precond

