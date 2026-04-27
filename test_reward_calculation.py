#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import pandas as pd
import snowflake.connector

# =============================================================================
# HARD-CODE HERE
# =============================================================================

SQL = """
SELECT                                                                                                                                                                     
                                 ( (SELECT COUNT(*) FROM CMS_DATA.CMS_SYNTHETIC_PATIENT_DATA_OMOP.PERSON) -                                                                                             
                                   (SELECT COUNT(DISTINCT d."person_id")                                                                                                                                
                                    FROM CMS_DATA.CMS_SYNTHETIC_PATIENT_DATA_OMOP.DRUG_EXPOSURE d                                                                                                       
                                    JOIN CMS_DATA.CMS_SYNTHETIC_PATIENT_DATA_OMOP.CONCEPT_ANCESTOR ca                                                                                                   
                                        ON d."drug_concept_id" = ca."descendant_concept_id"                                                                                                             
                                    JOIN CMS_DATA.CMS_SYNTHETIC_PATIENT_DATA_OMOP.CONCEPT c                                                                                                             
                                        ON ca."ancestor_concept_id" = c."concept_id"                                                                                                                    
                                    WHERE c."concept_code" = '35208')                                                                                                                                   
                                 ) * 100.0 / (SELECT COUNT(*) FROM CMS_DATA.CMS_SYNTHETIC_PATIENT_DATA_OMOP.PERSON)                                                                                     
                             AS percentage_not_using_quinapril_related;  
"""

GOLD_CSVS = [
    "/efs/open_source_sql_agentic_rl/agent-lightning/examples/spider2/Spider2/spider2-snow/evaluation_suite/gold/exec_result/sf_bq355_a.csv"
]

# per-gold condition cols (aligned to GOLD_CSVS order). Examples:
#   []                 -> use full gold rows for row-subset fallback
#   [4,5,6]            -> broadcast to all gold csvs
#   [[4,5,6],[1]]      -> per-gold
CONDITION_COLS = []   # e.g. [] or [4,5,6] or [[1],[3],[1]]

IGNORE_ORDER = True
TOLERANCE = 1e-2
MAX_SHOW = 10

ROW_DEBUG_SHOW_UNMATCHED = 10   # how many unmatched gold rows to print
ROW_DEBUG_TOPK_CLOSEST = 3      # show top-k closest pred rows for each unmatched gold row

CRED_PATH = "/efs/sunkexua/src/verl/data/spider2-snow/credentials/snowflake_credential.json"
TIMEOUT_S = 120


# =============================================================================
# EXEC SQL
# =============================================================================

def exec_sql(sql: str) -> pd.DataFrame:
    with open(CRED_PATH, "r", encoding="utf-8") as f:
        cred = json.load(f)

    conn = snowflake.connector.connect(**cred)
    cur = conn.cursor()
    cur.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {int(TIMEOUT_S)}")
    cur.execute(sql)

    if cur.description:
        headers = [d[0] for d in cur.description]
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=headers)
    else:
        conn.commit()
        df = pd.DataFrame()

    conn.close()
    return df


def normalize_condition_cols(condition_cols, num_gold: int):
    """
    Normalize CONDITION_COLS to List[List[int]] length == num_gold.
    """
    if not condition_cols:
        return [[] for _ in range(num_gold)]

    # list-of-lists
    if isinstance(condition_cols, list) and all(isinstance(x, list) for x in condition_cols):
        if len(condition_cols) == num_gold:
            return condition_cols
        if len(condition_cols) == 1:
            return [condition_cols[0] for _ in range(num_gold)]
        if len(condition_cols) < num_gold:
            last = condition_cols[-1]
            return condition_cols + [last for _ in range(num_gold - len(condition_cols))]
        return condition_cols[:num_gold]

    # single list[int] -> broadcast
    if isinstance(condition_cols, list):
        return [condition_cols for _ in range(num_gold)]

    raise ValueError(f"Invalid CONDITION_COLS: {condition_cols}")


# =============================================================================
# COLUMN-VECTOR COMPARE (debug)
# =============================================================================

def col_compare_by_vectors_debug(pred: pd.DataFrame, gold: pd.DataFrame, condition_cols_for_gold) -> float:
    condition_cols = condition_cols_for_gold or []

    def _norm(x):
        return 0 if pd.isna(x) else x

    def _to_float_or_none(x):
        x = _norm(x)
        if pd.isna(x):
            return None
        try:
            return float(x)
        except Exception:
            return None

    def _cell_eq(a, b):
        fa, fb = _to_float_or_none(a), _to_float_or_none(b)
        if fa is not None and fb is not None:
            return math.isclose(fa, fb, abs_tol=TOLERANCE)
        return str(_norm(a)).strip() == str(_norm(b)).strip()

    def _sort_key(x):
        fx = _to_float_or_none(x)
        if fx is not None:
            return (0, fx)
        return (1, str(_norm(x)).strip())

    def _debug_vec_pair(gold_vec, pred_vec, gold_name, pred_name) -> bool:
        gv = [_norm(x) for x in gold_vec]
        pv = [_norm(x) for x in pred_vec]

        if len(gv) != len(pv):
            print(f"      - FAIL len: gold_len={len(gv)} pred_len={len(pv)}")
            return False

        if IGNORE_ORDER:
            gv_sorted = sorted(gv, key=_sort_key)
            pv_sorted = sorted(pv, key=_sort_key)
        else:
            gv_sorted = gv
            pv_sorted = pv

        for i, (a, b) in enumerate(zip(gv_sorted, pv_sorted)):
            if not _cell_eq(a, b):
                print(f"      - FAIL first_mismatch@{i}: gold={repr(a)} pred={repr(b)}")
                print(f"      - gold({gold_name}) head: {gv[:MAX_SHOW]}")
                print(f"      - pred({pred_name}) head: {pv[:MAX_SHOW]}")
                if IGNORE_ORDER:
                    print(f"      - gold_sorted head: {gv_sorted[:MAX_SHOW]}")
                    print(f"      - pred_sorted head: {pv_sorted[:MAX_SHOW]}")
                return False

        print("      - PASS (vectors match)")
        return True

    # select gold columns to score
    if condition_cols:
        if gold.shape[1] <= max(condition_cols):
            print(f"[COL] gold has {gold.shape[1]} cols but condition_cols={condition_cols} -> col_ratio=0.000")
            return 0.0
        gold_use = gold.iloc[:, condition_cols]
        gold_colnames = list(gold_use.columns)
        print(f"[COL] scoring GOLD condition_cols={condition_cols} -> {gold_colnames}")
    else:
        gold_use = gold
        gold_colnames = list(gold.columns)
        print(f"[COL] scoring ALL GOLD columns -> {gold_colnames}")

    pred_colnames = list(pred.columns)
    print(f"[COL] PRED columns -> {pred_colnames}")

    gold_vectors = gold_use.transpose().values.tolist()
    pred_vectors = pred.transpose().values.tolist()

    total = len(gold_vectors)
    if total == 0:
        print("[COL] empty gold vectors -> col_ratio=0.000")
        return 0.0

    matched = 0
    mapping = {}

    for gi, gv in enumerate(gold_vectors):
        gname = gold_colnames[gi]
        print(f"\n[COL] GOLD col {gi}/{total-1}: {gname}")

        found = False
        for pj, pv in enumerate(pred_vectors):
            pname = pred_colnames[pj]
            print(f"    trying PRED col {pj}: {pname}")
            if _debug_vec_pair(gv, pv, gname, pname):
                found = True
                mapping[gname] = pname
                break

        if found:
            matched += 1
            print(f"  => MATCHED: {gname} -> {mapping[gname]}")
        else:
            print(f"  => UNMATCHED: {gname}")

    ratio = matched / total
    print("\n[COL SUMMARY]")
    print(f"matched_cols={matched}/{total} col_ratio={ratio:.3f}")
    if mapping:
        print("matched_map (gold -> pred):")
        for k, v in mapping.items():
            print(f"  - {k} -> {v}")
    unmatched = [c for c in gold_colnames if c not in mapping]
    if unmatched:
        print("unmatched_gold_cols:", unmatched)

    return ratio


# =============================================================================
# ROW-SUBSET FALLBACK (value-based, no column names)
# =============================================================================

def _norm_cell_value_based(x, tol=TOLERANCE):
    """
    Normalize a cell into a hashable token.
    - numeric: quantized by tolerance
    - string: stripped
    """
    if pd.isna(x):
        return None
    try:
        fx = float(x)
        q = round(fx / tol) * tol
        return ("num", q)
    except Exception:
        return ("str", str(x).strip())


def row_subset_fallback_debug(pred: pd.DataFrame, gold: pd.DataFrame, condition_cols_for_gold) -> float:
    """
    Gold row token-set (condition cols or full row) must be subset of some pred row token-set (full pred row).
    Returns recall over gold rows: matched_gold / total_gold.
    """
    cond = condition_cols_for_gold or []

    # build gold row sets
    gold_sets = []
    if cond:
        # validate indices
        if gold.shape[1] <= max(cond):
            print(f"[ROW] gold has {gold.shape[1]} cols but condition_cols={cond} -> row_recall=0.000")
            return 0.0
        print(f"[ROW] using gold condition_cols={cond} (row tokens from those cols only)")
        for r in gold.itertuples(index=False, name=None):
            tokens = []
            for c in cond:
                tokens.append(_norm_cell_value_based(r[c]))
            gold_sets.append(frozenset(t for t in tokens if t is not None))
    else:
        print(f"[ROW] condition_cols empty -> using FULL gold row (all cols) as token set")
        for r in gold.itertuples(index=False, name=None):
            tokens = [_norm_cell_value_based(v) for v in r]
            gold_sets.append(frozenset(t for t in tokens if t is not None))

    # build pred row sets (always full pred row)
    pred_sets = []
    for r in pred.itertuples(index=False, name=None):
        tokens = [_norm_cell_value_based(v) for v in r]
        pred_sets.append(frozenset(t for t in tokens if t is not None))

    if not gold_sets or not pred_sets:
        print("[ROW] empty gold or pred -> row_recall=0.000")
        return 0.0

    matched = 0
    unmatched_printed = 0

    for i, gset in enumerate(gold_sets):
        found = False

        # track closest rows by intersection size
        best = []  # list of (inter, j, missing_count, missing_tokens)

        for j, pset in enumerate(pred_sets):
            inter = len(gset & pset)
            missing = gset - pset
            best.append((inter, j, len(missing), missing))
            if gset.issubset(pset):
                found = True
                break

        if found:
            matched += 1
            continue

        # debug unmatched
        if unmatched_printed < ROW_DEBUG_SHOW_UNMATCHED:
            unmatched_printed += 1
            best.sort(key=lambda x: (x[0], -x[2]), reverse=True)
            print(f"\n[ROW] UNMATCHED gold_row#{i}: gset_size={len(gset)} tokens={sorted(list(gset))[:20]}")
            for rank, (inter, j, missn, missing) in enumerate(best[:ROW_DEBUG_TOPK_CLOSEST]):
                print(f"  closest#{rank+1}: pred_row#{j} inter={inter}/{len(gset)} missing={list(missing)[:10]}")
                try:
                    print("    pred raw:", pred.iloc[j].to_dict())
                except Exception:
                    pass

    recall = matched / len(gold_sets)
    print(f"\n[ROW SUMMARY] matched_gold_rows={matched}/{len(gold_sets)} row_recall={recall:.3f}")
    return recall


# =============================================================================
# MAIN (Policy B: row fallback only if col_ratio==0)
# =============================================================================

def main():
    pred_df = exec_sql(SQL)
    print(f"[PRED] rows={len(pred_df)} cols={len(pred_df.columns)}")
    print(pred_df.head(5).to_string(index=False))
    print("[PRED] dtypes:\n" + pred_df.dtypes.to_string())

    conds = normalize_condition_cols(CONDITION_COLS, len(GOLD_CSVS))

    best_ratio = -1.0
    best_gold = None

    for i, path in enumerate(GOLD_CSVS):
        print("\n" + "=" * 80)
        if not os.path.exists(path):
            print(f"[GOLD] missing: {path}")
            continue

        gold_df = pd.read_csv(path)
        print(f"[GOLD] {path} rows={len(gold_df)} cols={len(gold_df.columns)}")
        print(gold_df.head(5).to_string(index=False))
        print("[GOLD] dtypes:\n" + gold_df.dtypes.to_string())

        cc = conds[i]
        print(f"[META] condition_cols_for_this_gold={cc} ignore_order={IGNORE_ORDER}")

        col_ratio = col_compare_by_vectors_debug(pred_df, gold_df, cc)

        final_ratio = col_ratio
        if IGNORE_ORDER and col_ratio == 0.0:
            print("\n[POLICY B] col_ratio==0 and IGNORE_ORDER=True -> try ROW-SUBSET fallback")
            row_recall = row_subset_fallback_debug(pred_df, gold_df, cc)
            final_ratio = row_recall
            print(f"[POLICY B] final_ratio=row_recall={final_ratio:.3f}")

        print(f"\n[FINAL for this gold] ratio={final_ratio:.3f}")

        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_gold = path

    print("\n" + "=" * 80)
    print("[FINAL]")
    print(f"best_match_ratio={best_ratio:.3f}")
    if best_gold:
        print(f"best_gold={best_gold}")




#### wrapped up here 
def compute_result_match_ratio(
    pred_df,
    gold_dfs,
    expected_csvs,
    condition_cols_meta,
    ignore_order: bool,
    tolerance: float = 1e-2,
    policy_b_row_fallback: bool = True,
    logger=None,
    debug: bool = True,
    row_debug_show_unmatched: int = 10,
    row_debug_topk_closest: int = 3,
):
    """
    Returns:
      best_ratio (float in [0,1]),
      best_ok (int: 1 if best_ratio==1 else 0),
      best_gold_idx (int or None),
      best_gold_path (str or None)
    """

    import math
    import pandas as pd

    def _log(msg):
        if not debug:
            return
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    def _warn(msg):
        if logger is not None:
            logger.warning(msg)
        else:
            print(msg)

    # -----------------------------
    # normalize condition_cols to per-gold list
    # -----------------------------
    num_gold = len(gold_dfs)

    def _normalize_condition_cols(condition_cols, n):
        if not condition_cols:
            return [[] for _ in range(n)]

        # list-of-lists
        if isinstance(condition_cols, list) and all(isinstance(x, list) for x in condition_cols):
            if len(condition_cols) == n:
                return condition_cols
            if len(condition_cols) == 1:
                return [condition_cols[0] for _ in range(n)]
            if len(condition_cols) < n:
                last = condition_cols[-1]
                return condition_cols + [last for _ in range(n - len(condition_cols))]
            return condition_cols[:n]

        # single list[int] -> broadcast
        if isinstance(condition_cols, list):
            return [condition_cols for _ in range(n)]

        raise ValueError(f"Invalid condition_cols_meta: {condition_cols}")

    conds_per_gold = _normalize_condition_cols(condition_cols_meta, num_gold)

    # -----------------------------
    # column-vector compare ratio
    # -----------------------------
    def col_compare_ratio(pred: pd.DataFrame, gold: pd.DataFrame, condition_cols_for_gold):
        cond = condition_cols_for_gold or []

        def _norm(x):
            return 0 if pd.isna(x) else x

        def _to_float_or_none(x):
            x = _norm(x)
            if pd.isna(x):
                return None
            try:
                return float(x)
            except Exception:
                return None

        def _cell_eq(a, b):
            fa, fb = _to_float_or_none(a), _to_float_or_none(b)
            if fa is not None and fb is not None:
                return math.isclose(fa, fb, abs_tol=tolerance)
            return str(_norm(a)).strip() == str(_norm(b)).strip()

        def _sort_key(x):
            fx = _to_float_or_none(x)
            if fx is not None:
                return (0, fx)
            return (1, str(_norm(x)).strip())

        def _vec_match(gold_vec, pred_vec):
            gv = [_norm(x) for x in gold_vec]
            pv = [_norm(x) for x in pred_vec]
            if len(gv) != len(pv):
                return False
            if ignore_order:
                gv = sorted(gv, key=_sort_key)
                pv = sorted(pv, key=_sort_key)
            return all(_cell_eq(a, b) for a, b in zip(gv, pv))

        # select gold columns to score
        if cond:
            if gold.shape[1] <= max(cond):
                return 0.0
            gold_use = gold.iloc[:, cond]
        else:
            gold_use = gold

        gold_vectors = gold_use.transpose().values.tolist()
        pred_vectors = pred.transpose().values.tolist()

        if not gold_vectors:
            return 0.0

        matched = 0
        for gv in gold_vectors:
            if any(_vec_match(gv, pv) for pv in pred_vectors):
                matched += 1
        return matched / len(gold_vectors)

    # -----------------------------
    # row-subset fallback ratio (recall over gold rows)
    # condition_cols empty => FULL gold row token set
    # -----------------------------
    def _norm_cell_value_based(x):
        if pd.isna(x):
            return None
        try:
            fx = float(x)
            q = round(fx / tolerance) * tolerance
            return ("num", q)
        except Exception:
            return ("str", str(x).strip())

    def row_subset_recall(pred: pd.DataFrame, gold: pd.DataFrame, condition_cols_for_gold):
        cond = condition_cols_for_gold or []

        # build gold row sets
        gold_sets = []
        if cond:
            if gold.shape[1] <= max(cond):
                return 0.0
            for r in gold.itertuples(index=False, name=None):
                tokens = [_norm_cell_value_based(r[c]) for c in cond]
                gold_sets.append(frozenset(t for t in tokens if t is not None))
        else:
            for r in gold.itertuples(index=False, name=None):
                tokens = [_norm_cell_value_based(v) for v in r]
                gold_sets.append(frozenset(t for t in tokens if t is not None))

        # build pred row sets (always full pred row)
        pred_sets = []
        for r in pred.itertuples(index=False, name=None):
            tokens = [_norm_cell_value_based(v) for v in r]
            pred_sets.append(frozenset(t for t in tokens if t is not None))

        if not gold_sets or not pred_sets:
            return 0.0

        matched = 0
        unmatched_printed = 0

        for i, gset in enumerate(gold_sets):
            found = False
            best = []  # (inter, j, missing)

            for j, pset in enumerate(pred_sets):
                inter = len(gset & pset)
                missing = gset - pset
                best.append((inter, j, missing))
                if gset.issubset(pset):
                    found = True
                    break

            if found:
                matched += 1
                continue

            if debug and unmatched_printed < row_debug_show_unmatched:
                unmatched_printed += 1
                best.sort(key=lambda x: x[0], reverse=True)
                _log(f"[ROW] UNMATCHED gold_row#{i} gset_size={len(gset)} tokens={sorted(list(gset))[:20]}")
                for rank, (inter, j, missing) in enumerate(best[:row_debug_topk_closest]):
                    _log(f"  closest#{rank+1}: pred_row#{j} inter={inter}/{len(gset)} missing={list(missing)[:10]}")
                    try:
                        _log(f"    pred raw: {pred.iloc[j].to_dict()}")
                    except Exception:
                        pass

        return matched / len(gold_sets)

    # -----------------------------
    # evaluate each gold, take best
    # -----------------------------
    best_ratio = -1.0
    best_idx = None

    for gi, gold in enumerate(gold_dfs):
        cc = conds_per_gold[gi] if gi < len(conds_per_gold) else []
        gold_path = expected_csvs[gi] if gi < len(expected_csvs) else f"<gold#{gi}>"

        _log("=" * 80)
        _log(f"[GOLD#{gi}] {gold_path} rows={len(gold)} cols={len(gold.columns)}")
        _log(f"[META] condition_cols={cc} ignore_order={ignore_order}")
        _log("[GOLD] head():\n" + gold.head(5).to_string(index=False))
        _log("[PRED] head():\n" + pred_df.head(5).to_string(index=False))

        col_ratio = col_compare_ratio(pred_df, gold, cc)
        final_ratio = col_ratio
        _log(f"[COL] ratio={col_ratio:.3f}")

        if policy_b_row_fallback and ignore_order and col_ratio == 0.0:
            _log("[POLICY B] col_ratio==0 and ignore_order=True -> ROW-SUBSET fallback")
            row_r = row_subset_recall(pred_df, gold, cc)
            final_ratio = row_r
            _log(f"[ROW] row_recall={row_r:.3f}")
            _log(f"[POLICY B] final_ratio={final_ratio:.3f}")

        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_idx = gi

    best_path = expected_csvs[best_idx] if (best_idx is not None and best_idx < len(expected_csvs)) else None
    best_ok = 1 if best_ratio == 1.0 else 0
    return best_ratio, best_ok, best_idx, best_path








if __name__ == "__main__":
    main()