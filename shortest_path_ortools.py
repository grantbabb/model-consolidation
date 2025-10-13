#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    # Preferred (older) OR-Tools import path
    from ortools.graph import pywrapgraph as _pywrapgraph  # type: ignore
    SimpleMinCostFlow = _pywrapgraph.SimpleMinCostFlow  # type: ignore[attr-defined]
except Exception:
    try:
        # Newer OR-Tools wheels expose the solver under this module
        from ortools.graph.python import min_cost_flow as _min_cost_flow  # type: ignore
        SimpleMinCostFlow = _min_cost_flow.SimpleMinCostFlow  # type: ignore[attr-defined]
    except Exception:
        SimpleMinCostFlow = None  # type: ignore


class ShortestPathError(Exception):
    pass


def read_edges_from_csv(
    csv_path: Path,
    source_col: str,
    target_col: str,
    cost_col: str,
    capacity_col: Optional[str],
    delimiter: str,
) -> List[Tuple[str, str, float, int]]:
    if not csv_path.exists():
        raise ShortestPathError(f"Input file not found: {csv_path}")

    edges: List[Tuple[str, str, float, int]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        required = {source_col, target_col, cost_col}
        missing = [c for c in required if c not in reader.fieldnames]  # type: ignore[arg-type]
        if missing:
            raise ShortestPathError(
                "CSV is missing required columns: " + ", ".join(missing)
            )

        for row in reader:
            u = str(row[source_col]).strip()
            v = str(row[target_col]).strip()
            if u == "" or v == "":
                raise ShortestPathError("Empty node label encountered in CSV")

            try:
                cost_value = float(str(row[cost_col]).strip())
            except ValueError as e:
                raise ShortestPathError(
                    f"Unable to parse cost '{row[cost_col]}' as number"
                ) from e

            capacity_value: int = 1
            if capacity_col is not None:
                raw_capacity = str(row.get(capacity_col, "")).strip()
                if raw_capacity != "":
                    try:
                        capacity_value = int(float(raw_capacity))
                    except ValueError as e:
                        raise ShortestPathError(
                            f"Unable to parse capacity '{raw_capacity}' as integer"
                        ) from e
                else:
                    capacity_value = 1

            edges.append((u, v, cost_value, capacity_value))

    return edges


def compute_shortest_path(
    edges: List[Tuple[str, str, float, int]],
    source: str,
    sink: str,
    cost_scale: int = 1,
    undirected: bool = False,
) -> Tuple[List[str], float]:
    if SimpleMinCostFlow is None:
        raise ShortestPathError(
            "ortools not available or incompatible. Install with 'pip install ortools'."
        )

    if source == sink:
        return [source], 0.0

    if cost_scale <= 0:
        raise ShortestPathError("cost_scale must be a positive integer")

    # Map node names to integer ids and preserve reverse mapping for output
    node_name_to_id: Dict[str, int] = {}
    node_id_to_name: Dict[int, str] = {}

    def get_node_id(name: str) -> int:
        if name not in node_name_to_id:
            new_id = len(node_name_to_id)
            node_name_to_id[name] = new_id
            node_id_to_name[new_id] = name
        return node_name_to_id[name]

    start_nodes: List[int] = []
    end_nodes: List[int] = []
    capacities: List[int] = []
    unit_costs: List[int] = []

    for u_name, v_name, cost_value, capacity_value in edges:
        u_id = get_node_id(u_name)
        v_id = get_node_id(v_name)

        # Scale cost to integer for OR-Tools
        scaled_cost = int(round(cost_value * cost_scale))

        start_nodes.append(u_id)
        end_nodes.append(v_id)
        capacities.append(max(1, int(capacity_value)))
        unit_costs.append(scaled_cost)

        if undirected:
            start_nodes.append(v_id)
            end_nodes.append(u_id)
            capacities.append(max(1, int(capacity_value)))
            unit_costs.append(scaled_cost)

    # Ensure source and sink are in the mapping even if isolated in the edge list
    source_id = get_node_id(source)
    sink_id = get_node_id(sink)

    flow_solver = SimpleMinCostFlow()

    # Compatibility helpers across pywrapgraph (CamelCase) and python (snake_case) APIs
    def add_arc_with_capacity_and_unit_cost(tail: int, head: int, capacity: int, unit_cost: int) -> None:
        if hasattr(flow_solver, "AddArcWithCapacityAndUnitCost"):
            flow_solver.AddArcWithCapacityAndUnitCost(tail, head, capacity, unit_cost)
        else:
            flow_solver.add_arc_with_capacity_and_unit_cost(tail, head, capacity, unit_cost)

    def set_node_supply(node_id: int, supply: int) -> None:
        if hasattr(flow_solver, "SetNodeSupply"):
            flow_solver.SetNodeSupply(node_id, supply)
        else:
            flow_solver.set_node_supply(node_id, supply)

    def solve() -> object:
        if hasattr(flow_solver, "Solve"):
            return flow_solver.Solve()
        return flow_solver.solve()

    def num_arcs() -> int:
        return flow_solver.NumArcs() if hasattr(flow_solver, "NumArcs") else flow_solver.num_arcs()

    def flow(i: int) -> int:
        return flow_solver.Flow(i) if hasattr(flow_solver, "Flow") else flow_solver.flow(i)

    def tail(i: int) -> int:
        return flow_solver.Tail(i) if hasattr(flow_solver, "Tail") else flow_solver.tail(i)

    def head(i: int) -> int:
        return flow_solver.Head(i) if hasattr(flow_solver, "Head") else flow_solver.head(i)

    def optimal_cost() -> int:
        return (
            flow_solver.OptimalCost() if hasattr(flow_solver, "OptimalCost") else flow_solver.optimal_cost()
        )

    for i in range(len(start_nodes)):
        add_arc_with_capacity_and_unit_cost(
            start_nodes[i], end_nodes[i], capacities[i], unit_costs[i]
        )

    all_node_ids = list(node_id_to_name.keys())
    for node_id in all_node_ids:
        set_node_supply(node_id, 0)

    set_node_supply(source_id, 1)
    set_node_supply(sink_id, -1)

    status = solve()

    # Determine the OPTIMAL status constant in both APIs
    optimal_status = getattr(flow_solver, "OPTIMAL", None)
    if optimal_status is None and hasattr(flow_solver, "Status"):
        optimal_status = flow_solver.Status.OPTIMAL

    if status != optimal_status:
        raise ShortestPathError(
            f"Min-cost flow did not find a solution (status={status})."
        )

    # Extract the unique unit-flow path from source to sink
    next_by_node: Dict[int, int] = {}
    for i in range(num_arcs()):
        if flow(i) > 0:
            t = tail(i)
            h = head(i)
            next_by_node[t] = h

    if source_id not in next_by_node:
        raise ShortestPathError("No path found carrying unit flow from source to sink")

    ordered_path_ids: List[int] = [source_id]
    visited: set[int] = set([source_id])

    while ordered_path_ids[-1] != sink_id:
        current = ordered_path_ids[-1]
        if current not in next_by_node:
            raise ShortestPathError(
                "Disconnected flow: could not reconstruct a full path to sink"
            )
        nxt = next_by_node[current]
        if nxt in visited:
            raise ShortestPathError("Cycle encountered while reconstructing path")
        ordered_path_ids.append(nxt)
        visited.add(nxt)

    ordered_path_names = [node_id_to_name[nid] for nid in ordered_path_ids]

    total_cost_scaled = optimal_cost()
    total_cost = float(total_cost_scaled) / float(cost_scale)

    return ordered_path_names, total_cost


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a shortest path using OR-Tools Min-Cost Flow. "
            "Provide a CSV of edges or use --demo."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to CSV file with edges. Columns default to u,v,cost. "
            "Use --source-col/--target-col/--cost-col to override."
        ),
    )
    parser.add_argument("--source", type=str, default=None, help="Source node label")
    parser.add_argument("--sink", type=str, default=None, help="Sink node label")
    parser.add_argument(
        "--source-col", type=str, default="u", help="CSV column for source node"
    )
    parser.add_argument(
        "--target-col", type=str, default="v", help="CSV column for target node"
    )
    parser.add_argument(
        "--cost-col", type=str, default="cost", help="CSV column for edge cost"
    )
    parser.add_argument(
        "--capacity-col",
        type=str,
        default=None,
        help="Optional CSV column for edge capacity (default 1)",
    )
    parser.add_argument(
        "--delimiter", type=str, default=",", help="CSV delimiter (default ',')"
    )
    parser.add_argument(
        "--cost-scale",
        type=int,
        default=1,
        help=(
            "Multiply costs by this integer before solving (default 1). "
            "Use e.g. 100 or 1000 if your costs have decimals."
        ),
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Treat edges as undirected (adds reverse arcs)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a built-in demo graph (ignores --input)",
    )
    return parser


def run_demo(cost_scale: int, undirected: bool) -> None:
    demo_edges = [
        ("A", "B", 4.0, 1),
        ("A", "C", 2.0, 1),
        ("C", "B", 1.0, 1),
        ("B", "D", 2.0, 1),
        ("C", "D", 5.0, 1),
    ]
    path, cost = compute_shortest_path(
        demo_edges, source="A", sink="D", cost_scale=cost_scale, undirected=undirected
    )
    print("Shortest path:", " -> ".join(path))
    print("Total cost:", cost)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if SimpleMinCostFlow is None:
        sys.stderr.write(
            "ERROR: ortools not installed or incompatible. Install it with: pip install ortools\n"
        )
        return 2

    if args.demo:
        try:
            run_demo(cost_scale=args.cost_scale, undirected=bool(args.undirected))
            return 0
        except ShortestPathError as e:
            sys.stderr.write(f"ERROR: {e}\n")
            return 1

    if not args.input or not args.source or not args.sink:
        parser.print_help(sys.stderr)
        sys.stderr.write(
            "\nERROR: --input, --source, and --sink are required unless using --demo.\n"
        )
        return 2

    try:
        edges = read_edges_from_csv(
            csv_path=Path(args.input),
            source_col=args.source_col,
            target_col=args.target_col,
            cost_col=args.cost_col,
            capacity_col=args.capacity_col,
            delimiter=args.delimiter,
        )
        path, cost = compute_shortest_path(
            edges=edges,
            source=str(args.source),
            sink=str(args.sink),
            cost_scale=int(args.cost_scale),
            undirected=bool(args.undirected),
        )
        print("Shortest path:", " -> ".join(path))
        print("Total cost:", cost)
        return 0
    except ShortestPathError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
