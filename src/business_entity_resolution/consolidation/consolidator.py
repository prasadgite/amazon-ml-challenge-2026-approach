from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ConsolidationConfig:
    min_edge_score: float = 0.0
    reject_country_conflict: bool = True
    reject_explicit_negative: bool = True
    explicit_negative_score_threshold: float = 0.99
    reject_same_source_duplicate: bool = False
    max_cluster_size: int = 1000


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: str, b: str) -> str:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


def _node(source: str, entity_id: str) -> str:
    # Source is part of the key so identical IDs in different datasets never collide.
    return f"{source}:{entity_id}"


def _split_node(node: str) -> tuple[str, str]:
    source, entity_id = node.split(":", 1)
    return source, entity_id


def _stable_cluster_id(nodes: Iterable[str]) -> str:
    canonical = "|".join(sorted(nodes)).encode("utf-8")
    digest = hashlib.blake2b(canonical, digest_size=10).hexdigest()
    return f"ENT-{digest}"


def _read_tsv(path: Path) -> Iterable[dict[str, str]]:
    try:
        csv.field_size_limit(2147483647)
    except Exception:
        pass
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


class EntityConsolidator:
    """Consolidate accepted M7 pair matches into deterministic entity clusters."""

    def __init__(self, config: ConsolidationConfig | None = None):
        self.config = config or ConsolidationConfig()

    def _edge_accepted(self, row: dict[str, str]) -> tuple[bool, str]:
        if row.get("decision") != "MATCH":
            return False, "not_match"
        score = _float(row, "calibrated_score")
        if score < self.config.min_edge_score:
            return False, "below_m8_score"
        if self.config.reject_country_conflict and _float(row, "country_conflict") >= 1:
            return False, "country_conflict"
        return True, "accepted"

    def _load_edges(self, decision_path: Path):
        edges = []
        nodes: set[str] = set()
        rejected: dict[str, int] = {}
        for row in _read_tsv(decision_path):
            source1 = row.get("source1_id", "")
            target_source = row.get("target_source", "")
            target_id = row.get("target_id", "")
            if not source1 or not target_source or not target_id:
                rejected["missing_node_id"] = rejected.get("missing_node_id", 0) + 1
                continue
            accepted, reason = self._edge_accepted(row)
            if not accepted:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            left = _node("S1", source1)
            right = _node(target_source, target_id)
            edges.append({
                "left": left,
                "right": right,
                "source1_id": source1,
                "target_source": target_source,
                "target_id": target_id,
                "score": _float(row, "calibrated_score"),
                "reason": row.get("decision_reason", ""),
            })
            nodes.add(left)
            nodes.add(right)
        return edges, nodes, rejected

    def build_clusters(self, decision_path: Path) -> dict:
        edges, nodes, rejected = self._load_edges(decision_path)

        # Keep explicit pair decisions available for a conservative transitivity
        # check. Absence of a pair is deliberately treated as unknown, not negative.
        decision_lookup: dict[tuple[str, str, str], dict[str, str]] = {}
        for row in _read_tsv(decision_path):
            s1, ts, tid = row.get("source1_id", ""), row.get("target_source", ""), row.get("target_id", "")
            if s1 and ts and tid:
                decision_lookup[(s1, ts, tid)] = row

        uf = UnionFind()
        for node in nodes:
            uf.add(node)

        def component_members(root: str) -> list[str]:
            return [n for n in uf.parent if uf.find(n) == root]

        def component_sources(members: list[str]):
            result: dict[str, list[str]] = {}
            for n in members:
                src, eid = _split_node(n)
                result.setdefault(src, []).append(eid)
            return result

        def has_explicit_conflict(left_members: list[str], right_members: list[str]) -> bool:
            if not self.config.reject_explicit_negative:
                return False
            a, b = component_sources(left_members), component_sources(right_members)
            pairs = []
            for s1 in a.get("S1", []):
                for target_source in ("S2", "S3"):
                    for target_id in b.get(target_source, []):
                        pairs.append((s1, target_source, target_id))
            for s1 in b.get("S1", []):
                for target_source in ("S2", "S3"):
                    for target_id in a.get(target_source, []):
                        pairs.append((s1, target_source, target_id))
            for key in pairs:
                row = decision_lookup.get(key)
                if row and row.get("decision") == "NON_MATCH" and _float(row, "calibrated_score") >= self.config.explicit_negative_score_threshold:
                    return True
            return False

        rejected_merge_conflicts = 0
        accepted_edges = []
        for edge in edges:
            ra, rb = uf.find(edge["left"]), uf.find(edge["right"])
            if ra == rb:
                accepted_edges.append(edge)
                continue
            left_members = component_members(ra)
            right_members = component_members(rb)
            if has_explicit_conflict(left_members, right_members):
                rejected_merge_conflicts += 1
                rejected["explicit_negative_transitive_conflict"] = rejected.get("explicit_negative_transitive_conflict", 0) + 1
                continue
            uf.union(ra, rb)
            accepted_edges.append(edge)

        # Rebuild components after conflict-aware unions.
        components: dict[str, list[str]] = {}
        for node in nodes:
            components.setdefault(uf.find(node), []).append(node)

        clusters: dict[str, dict] = {}
        node_to_cluster: dict[str, str] = {}
        oversized = 0
        for members in components.values():
            members.sort()
            cluster_id = _stable_cluster_id(members)
            sources = {}
            for node in members:
                source, entity_id = _split_node(node)
                sources.setdefault(source, []).append(entity_id)
                node_to_cluster[node] = cluster_id
            if len(members) > self.config.max_cluster_size:
                oversized += 1
            clusters[cluster_id] = {
                "cluster_id": cluster_id,
                "node_count": len(members),
                "members": members,
                "sources": {k: sorted(v) for k, v in sorted(sources.items())},
                "s1_count": len(sources.get("S1", [])),
                "target_count": sum(len(v) for k, v in sources.items() if k != "S1"),
                "oversized": len(members) > self.config.max_cluster_size,
            }

        return {
            "edges": accepted_edges,
            "nodes": nodes,
            "clusters": clusters,
            "node_to_cluster": node_to_cluster,
            "rejected": rejected,
            "oversized_clusters": oversized,
            "rejected_merge_conflicts": rejected_merge_conflicts,
        }

    def write_assignments(self, state: dict, output_path: Path) -> dict:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = 0
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "cluster_id", "source", "entity_id", "node_count", "s1_count", "target_count", "oversized",
            ], delimiter="\t")
            writer.writeheader()
            for cluster_id in sorted(state["clusters"]):
                cluster = state["clusters"][cluster_id]
                for node in cluster["members"]:
                    source, entity_id = _split_node(node)
                    writer.writerow({
                        "cluster_id": cluster_id,
                        "source": source,
                        "entity_id": entity_id,
                        "node_count": cluster["node_count"],
                        "s1_count": cluster["s1_count"],
                        "target_count": cluster["target_count"],
                        "oversized": int(cluster["oversized"]),
                    })
                    rows += 1
        return {"rows": rows, "output": str(output_path)}

    def write_s1_resolution(self, state: dict, output_path: Path) -> dict:
        """Write one canonical resolution row per S1 record represented in a cluster."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = 0
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "source1_id", "cluster_id", "s2_ids", "s3_ids", "cluster_size", "s1_count", "target_count",
            ], delimiter="\t")
            writer.writeheader()
            for cluster_id in sorted(state["clusters"]):
                cluster = state["clusters"][cluster_id]
                s2_ids = sorted(cluster["sources"].get("S2", []))
                s3_ids = sorted(cluster["sources"].get("S3", []))
                for source1_id in sorted(cluster["sources"].get("S1", [])):
                    writer.writerow({
                        "source1_id": source1_id,
                        "cluster_id": cluster_id,
                        "s2_ids": ",".join(s2_ids),
                        "s3_ids": ",".join(s3_ids),
                        "cluster_size": cluster["node_count"],
                        "s1_count": cluster["s1_count"],
                        "target_count": cluster["target_count"],
                    })
                    rows += 1
        return {"rows": rows, "output": str(output_path)}

    def write_edges(self, state: dict, output_path: Path) -> dict:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "source1_id", "target_source", "target_id", "calibrated_score", "decision_reason", "cluster_id",
            ], delimiter="\t")
            writer.writeheader()
            for edge in sorted(state["edges"], key=lambda x: (x["source1_id"], x["target_source"], x["target_id"])):
                writer.writerow({
                    "source1_id": edge["source1_id"],
                    "target_source": edge["target_source"],
                    "target_id": edge["target_id"],
                    "calibrated_score": f'{edge["score"]:.10f}',
                    "decision_reason": edge["reason"],
                    "cluster_id": state["node_to_cluster"][_node("S1", edge["source1_id"])],
                })
        return {"rows": len(state["edges"]), "output": str(output_path)}

    def report(self, state: dict) -> dict:
        clusters = list(state["clusters"].values())
        sizes = sorted(c["node_count"] for c in clusters)
        target_sizes = sorted(c["target_count"] for c in clusters)
        s1_sizes = sorted(c["s1_count"] for c in clusters)

        def quantile(values: list[int], q: float) -> float:
            if not values:
                return 0.0
            if len(values) == 1:
                return float(values[0])
            pos = (len(values) - 1) * q
            lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
            frac = pos - lo
            return values[lo] + frac * (values[hi] - values[lo])

        return {
            "config": asdict(self.config),
            "nodes": len(state["nodes"]),
            "accepted_edges": len(state["edges"]),
            "rejected_rows": sum(state["rejected"].values()),
            "rejected_reasons": state["rejected"],
            "clusters": len(clusters),
            "clusters_with_multiple_s1": sum(c["s1_count"] > 1 for c in clusters),
            "clusters_with_multiple_targets": sum(c["target_count"] > 1 for c in clusters),
            "oversized_clusters": state["oversized_clusters"],
            "cluster_size": {
                "min": min(sizes) if sizes else 0,
                "median": quantile(sizes, 0.5),
                "p95": quantile(sizes, 0.95),
                "p99": quantile(sizes, 0.99),
                "max": max(sizes) if sizes else 0,
            },
            "s1_per_cluster": {
                "max": max(s1_sizes) if s1_sizes else 0,
                "median": quantile(s1_sizes, 0.5),
            },
            "targets_per_cluster": {
                "max": max(target_sizes) if target_sizes else 0,
                "median": quantile(target_sizes, 0.5),
            },
        }

    def run(self, decision_path: Path, assignments_path: Path, edges_path: Path, report_path: Path) -> dict:
        state = self.build_clusters(decision_path)
        self.write_assignments(state, assignments_path)
        self.write_edges(state, edges_path)
        report = self.report(state)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report


class ConsolidationEvaluator:
    """Evaluate M8 clusters against the S1 -> target ground-truth links."""

    def _ground_truth(self, path: Path) -> set[tuple[str, str, str]]:
        positives: set[tuple[str, str, str]] = set()
        for row in _read_tsv(path):
            s1 = row.get("source1_entity_id", "")
            raw = row.get("matched_entity_ids", "")
            if not s1 or not raw:
                continue
            for target in raw.split(","):
                target = target.strip()
                if not target:
                    continue
                # Dataset IDs are source-qualified by their prefixes in the Amazon
                # competition data. Keep the resolver centralized here so the
                # evaluation logic never depends on cluster ordering.
                source = "S2" if target.startswith("S2") else "S3" if target.startswith("S3") else "UNKNOWN"
                positives.add((s1, source, target))
        return positives

    @staticmethod
    def _metrics(tp: int, fp: int, fn: int) -> dict:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
                "precision": precision, "recall": recall, "f1": f1}

    def evaluate(self, state: dict, ground_truth_path: Path) -> dict:
        truth = self._ground_truth(ground_truth_path)
        predicted: set[tuple[str, str, str]] = set()
        direct_edges: set[tuple[str, str, str]] = set()

        for edge in state["edges"]:
            direct_edges.add((edge["source1_id"], edge["target_source"], edge["target_id"]))

        # A cluster implies every S1 member is associated with every target
        # member in that cluster. This measures the effect of transitive closure.
        for cluster in state["clusters"].values():
            s1s = cluster["sources"].get("S1", [])
            for source in ("S2", "S3"):
                for target in cluster["sources"].get(source, []):
                    for s1 in s1s:
                        predicted.add((s1, source, target))

        direct_metrics = self._metrics(
            len(direct_edges & truth),
            len(direct_edges - truth),
            len(truth - direct_edges),
        )
        cluster_metrics = self._metrics(
            len(predicted & truth),
            len(predicted - truth),
            len(truth - predicted),
        )
        inferred = predicted - direct_edges
        inferred_metrics = self._metrics(
            len(inferred & truth),
            len(inferred - truth),
            len(truth - predicted),
        )

        return {
            "ground_truth_pairs": len(truth),
            "direct_edge_metrics": direct_metrics,
            "cluster_closure_metrics": cluster_metrics,
            "transitive_inference": {
                "inferred_pairs": len(inferred),
                "correct_inferred_pairs": len(inferred & truth),
                "incorrect_inferred_pairs": len(inferred - truth),
                "precision": inferred_metrics["precision"],
            },
            "cluster_count": len(state["clusters"]),
        }
