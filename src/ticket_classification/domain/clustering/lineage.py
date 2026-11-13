"""Track stable taxonomy identities independently of local cluster indices."""

from uuid import uuid4

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize


def build_lineage(
    previous,
    current,
    old_ids: dict[int, str],
    compatible: bool,
    threshold: float = 0.75,
) -> tuple[dict[int, str], dict]:
    new_ids = {i: uuid4().hex for i in range(len(current))}
    edges = []
    retained = set()
    if compatible and previous is not None and len(previous) and len(current):
        similarities = normalize(previous) @ normalize(current).T
        rows, cols = linear_sum_assignment(1 - similarities)
        for old, new in zip(rows, cols):
            if similarities[old, new] >= threshold and int(old) in old_ids:
                new_ids[int(new)] = old_ids[int(old)]
                retained.add(old_ids[int(old)])
        for old, new in np.argwhere(similarities >= threshold):
            edges.append(
                {
                    "from": old_ids.get(int(old)),
                    "to": new_ids[int(new)],
                    "cosine_similarity": float(similarities[old, new]),
                }
            )
    retired = sorted(set(old_ids.values()) - retained)
    births = sorted(set(new_ids.values()) - retained)
    splits = {
        sid: [e["to"] for e in edges if e["from"] == sid] for sid in old_ids.values()
    }
    merges = {
        sid: [e["from"] for e in edges if e["to"] == sid] for sid in new_ids.values()
    }
    return new_ids, {
        "retained": sorted(retained),
        "births": births,
        "retired": retired,
        "splits": {k: v for k, v in splits.items() if len(v) > 1},
        "merges": {k: v for k, v in merges.items() if len(v) > 1},
        "edges": edges,
        "embedding_space_compatible": compatible,
    }
