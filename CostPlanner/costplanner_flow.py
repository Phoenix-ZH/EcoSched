"""Integral minimum-cost flow with residual rerouting and reduced costs.

Successive shortest augmenting paths maximize the amount of source-to-sink
flow, up to a supplied limit, and minimize its cost. Forward input costs are
nonnegative integers. Reverse residual arcs carry the negated input cost so
later paths can revise earlier assignments. Node potentials keep every
reachable residual arc's reduced cost nonnegative for Dijkstra's algorithm.

Capacities count flow units. Physical resources consumed by one assignment
remain the caller's responsibility when their demand differs between tasks.
"""

from heapq import heappop, heappush
from typing import List, Optional, Tuple


__all__ = ["MinCostFlow"]
EdgeHandle = Tuple[int, int]


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("{} must be an integer".format(name))
    if value < minimum:
        raise ValueError("{} must be >= {}".format(name, minimum))
    return value


class _Edge:
    __slots__ = ("to", "reverse", "residual", "cost", "initial")

    def __init__(self, to, reverse, residual, cost, initial):
        self.to = to
        self.reverse = reverse
        self.residual = residual
        self.cost = cost
        # None distinguishes a reverse arc from an original zero-capacity arc.
        self.initial = initial


class MinCostFlow:
    """Build a graph, solve it once, then inspect original-edge flow.

    ``add_edge`` returns an opaque ``(node, adjacency_index)`` handle. Parallel
    arcs and self-loops are supported. A graph becomes immutable when a valid
    ``solve`` call starts; construct another instance for a changed problem.
    This prevents reusing zero initial potentials on an existing residual
    network whose reverse arcs may already have positive capacity.
    """

    def __init__(self, node_count):
        self.node_count = _integer(node_count, "node_count", 1)
        self._graph = [[] for _ in range(self.node_count)]
        self._solved = False

    def _node(self, node):
        _integer(node, "node")
        if node >= self.node_count:
            raise ValueError("node is outside the graph")
        return node

    def add_edge(self, source, destination, capacity, cost):
        # type: (int, int, int, int) -> EdgeHandle
        """Add one original directed arc and its reverse residual arc."""
        if self._solved:
            raise RuntimeError("Cannot add edges after solving has started")
        source = self._node(source)
        destination = self._node(destination)
        capacity = _integer(capacity, "capacity")
        cost = _integer(cost, "cost")
        forward_index = len(self._graph[source])
        # Both arcs share one adjacency list for a self-loop.
        reverse_index = len(self._graph[destination]) + int(source == destination)
        self._graph[source].append(
            _Edge(destination, reverse_index, capacity, cost, capacity)
        )
        self._graph[destination].append(
            _Edge(source, forward_index, 0, -cost, None)
        )
        return source, forward_index

    def flow_on(self, handle):
        # type: (EdgeHandle) -> int
        """Return final net flow on an original arc, including any rerouting."""
        if not isinstance(handle, tuple) or len(handle) != 2:
            raise TypeError("edge handle must be a (node, adjacency_index) tuple")
        node = self._node(handle[0])
        index = _integer(handle[1], "adjacency_index")
        if index >= len(self._graph[node]):
            raise ValueError("edge handle is outside the adjacency list")
        edge = self._graph[node][index]
        if edge.initial is None:
            raise ValueError("edge handle must identify an original arc")
        return edge.initial - edge.residual

    def solve(self, source, sink, flow_limit):
        # type: (int, int, int) -> Tuple[int, int]
        """Return ``(maximum_flow_up_to_limit, minimum_total_integer_cost)``.

        All arithmetic remains integral. Dijkstra runs to completion before
        potentials are updated for every reachable vertex. Stopping at the
        sink would leave some tentative distances unsuitable as potentials.
        """
        if self._solved:
            raise RuntimeError("Each graph can be solved only once")
        source = self._node(source)
        sink = self._node(sink)
        if source == sink:
            raise ValueError("source and sink must be distinct")
        flow_limit = _integer(flow_limit, "flow_limit")
        self._solved = True
        graph = self._graph
        potential = [0] * self.node_count
        total_flow = 0
        total_cost = 0

        while total_flow < flow_limit:
            distance = [None] * self.node_count  # type: List[Optional[int]]
            previous = [None] * self.node_count  # type: List[Optional[EdgeHandle]]
            distance[source] = 0
            queue = [(0, source)]
            while queue:
                current_distance, node = heappop(queue)
                if current_distance != distance[node]:
                    continue
                for index, edge in enumerate(graph[node]):
                    if edge.residual == 0:
                        continue
                    reduced_cost = edge.cost + potential[node] - potential[edge.to]
                    if reduced_cost < 0:
                        raise RuntimeError("Residual reduced-cost invariant was violated")
                    candidate_distance = current_distance + reduced_cost
                    known = distance[edge.to]
                    if known is None or candidate_distance < known:
                        distance[edge.to] = candidate_distance
                        previous[edge.to] = node, index
                        heappush(queue, (candidate_distance, edge.to))

            if distance[sink] is None:
                break
            for node, value in enumerate(distance):
                if value is not None:
                    potential[node] += value

            amount = flow_limit - total_flow
            path_cost = 0
            node = sink
            while node != source:
                step = previous[node]
                if step is None:
                    raise RuntimeError("An augmenting path has no predecessor")
                parent, index = step
                edge = graph[parent][index]
                amount = min(amount, edge.residual)
                path_cost += edge.cost
                node = parent

            # Update both directions. Reading the original arcs after all
            # augmentations yields the allocation after canceled flow is gone.
            node = sink
            while node != source:
                parent, index = previous[node]
                edge = graph[parent][index]
                edge.residual -= amount
                graph[edge.to][edge.reverse].residual += amount
                node = parent
            total_flow += amount
            total_cost += amount * path_cost

        return total_flow, total_cost
