#!/usr/bin/env python3
"""Generates binary trace files (little-endian uint32 keys) for problem 3.

insert.bin : keys to insert (a fraction are duplicates of earlier keys)
search.bin : keys to search (a fraction are in the inserted set, the rest are not)
delete.bin : keys to delete (taken from the inserted set; optionally a fraction
             of never-inserted keys, and a fraction of duplicates)

Example:
  python3 gen_traces.py --n-insert 1000000 --n-search 1000000 --n-delete 300000 \
      --out traces
"""
import argparse
import os

import numpy as np


def distinct_keys(rng, count, exclude=None):
    """Returns `count` distinct random uint32 keys, none in `exclude`."""
    out = np.empty(0, dtype=np.uint32)
    while out.size < count:
        need = count - out.size
        cand = rng.integers(0, 2**32, size=int(need * 1.2) + 16, dtype=np.uint64)
        cand = np.unique(cand.astype(np.uint32))
        if exclude is not None and exclude.size:
            cand = cand[~np.isin(cand, exclude)]
        if out.size:
            cand = cand[~np.isin(cand, out)]
        rng.shuffle(cand)
        out = np.concatenate([out, cand[:need]])
    return out


def add_duplicates(rng, uniq, total):
    """Extends `uniq` to `total` entries by repeating random members, then shuffles."""
    extra = total - uniq.size
    if extra > 0:
        uniq = np.concatenate([uniq, rng.choice(uniq, size=extra, replace=True)])
    rng.shuffle(uniq)
    return uniq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-insert", type=int, default=1_000_000)
    p.add_argument("--n-search", type=int, default=1_000_000)
    p.add_argument("--n-delete", type=int, default=300_000)
    p.add_argument("--hit-frac", type=float, default=0.5,
                   help="fraction of search keys that were inserted")
    p.add_argument("--dup-frac", type=float, default=0.05,
                   help="fraction of insert/delete entries that repeat an earlier key")
    p.add_argument("--absent-del-frac", type=float, default=0.0,
                   help="fraction of delete keys that were never inserted")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="traces")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)

    n_ins_u = max(1, int(round(a.n_insert * (1 - a.dup_frac))))
    n_hit = int(round(a.n_search * a.hit_frac))
    n_miss = a.n_search - n_hit
    n_del_absent = int(round(a.n_delete * a.absent_del_frac))
    n_del_valid_total = a.n_delete - n_del_absent
    n_del_valid_u = min(n_ins_u, max(0, int(round(n_del_valid_total * (1 - a.dup_frac)))))
    n_del_absent_u = n_del_absent

    # keys used for inserts, then keys disjoint from them for misses and absent deletes
    ins_u = distinct_keys(rng, n_ins_u)
    absent = distinct_keys(rng, n_miss + n_del_absent_u, exclude=ins_u)

    # insert trace
    ins = add_duplicates(rng, ins_u.copy(), a.n_insert)

    # search trace
    hits = rng.choice(ins_u, size=n_hit, replace=True)
    search = np.concatenate([hits, absent[:n_miss]])
    rng.shuffle(search)

    # delete trace
    del_valid_u = rng.choice(ins_u, size=n_del_valid_u, replace=False)
    del_valid = add_duplicates(rng, del_valid_u.copy(), n_del_valid_total) \
        if n_del_valid_u > 0 else np.empty(0, dtype=np.uint32)
    delete = np.concatenate([del_valid, absent[n_miss:n_miss + n_del_absent_u]])
    rng.shuffle(delete)

    os.makedirs(a.out, exist_ok=True)
    for name, arr in (("insert.bin", ins), ("search.bin", search), ("delete.bin", delete)):
        arr.astype("<u4").tofile(os.path.join(a.out, name))

    print(f"wrote to {a.out}/")
    print(f"  insert.bin: {ins.size} keys ({np.unique(ins).size} distinct)")
    print(f"  search.bin: {search.size} keys ({n_hit} inserted, {n_miss} not inserted)")
    print(f"  delete.bin: {delete.size} keys ({np.unique(delete).size} distinct, "
          f"{n_del_absent} never inserted)")


if __name__ == "__main__":
    main()