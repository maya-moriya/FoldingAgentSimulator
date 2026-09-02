"""Initial DCEL setup and layer-graph pruning shared by all origami operations."""

from __future__ import annotations

import numpy as np
import networkx as nx
from networkx import DiGraph
from .config import DEFAULT_PAPER_BACK_COLOR, DEFAULT_PAPER_FRONT_COLOR
from .colors import resolve_paper_color
from .DCEL import Face, HalfEdge, Plane, Vertex

class OrigamiBase:
    """Builds the initial unit-square DCEL and prunes redundant layer relations."""

    vertices: dict[int, Vertex]
    faces: dict[int, Face]
    planes: dict[int, Plane]
    half_edges: dict[tuple[int, int], HalfEdge]
    layers: DiGraph

    def __init__(
        self,
        *,
        front_color: str = DEFAULT_PAPER_FRONT_COLOR,
        back_color: str = DEFAULT_PAPER_BACK_COLOR,
    ) -> None:
        self.paper_front_color: str = resolve_paper_color(front_color)
        self.paper_back_color: str = resolve_paper_color(back_color)
        self.vertices = {
            1: Vertex(id=1, pos=np.array([0, 1])),
            2: Vertex(id=2, pos=np.array([1, 1])),
            3: Vertex(id=3, pos=np.array([1, 0])),
            4: Vertex(id=4, pos=np.array([0, 0]))
        }
        self.faces = {
            1: Face(id=1, orientation=0, plane_id=1)
        }
        self.planes = {
            1: Plane(id=1, face_ids=[1])
        }
        self.layers = DiGraph() # edge (A, B) means A is above B
        self.layers.add_node(1)
        self._view_matrix: np.ndarray = np.eye(2, dtype=float)
        self._view_offset: np.ndarray = np.zeros(2, dtype=float)
        self._build_initial_edges()

    def _build_initial_edges(self) -> None:
        self.half_edges = {}
        internals = []
        boundaries = []
        for i in range(4):
            v_curr = i + 1
            v_next = ((i + 1) % 4) + 1

            internal = HalfEdge('B')
            boundary = HalfEdge('B')

            self.half_edges[(v_curr, v_next)] = internal
            self.half_edges[(v_next, v_curr)] = boundary

            internals.append(internal)
            boundaries.append(boundary)

        for i in range(4):
            v_curr = self.vertices[i + 1]
            v_next = self.vertices[((i + 1) % 4) + 1]

            internals[i].origin = v_curr
            internals[i].face = self.faces[1]
            internals[i].next = internals[(i + 1) % 4]
            internals[i].twin = boundaries[i]

            boundaries[i].origin = v_next
            boundaries[i].face = None
            boundaries[i].next = boundaries[(i + 1) % 4]
            boundaries[i].twin = internals[i]

            v_curr.he = internals[i]

        self.faces[1].edge = internals[0] 

    def _prune(self) -> list[tuple[int, int]]:
        """Remove redundant layer edges while preserving the same reachability."""
        redundant_edges = self._get_redundant_layer_edges()

        if redundant_edges:
            self.layers.remove_edges_from(redundant_edges)

        return redundant_edges

    def _get_redundant_layer_edges(self) -> list[tuple[int, int]]:
        """Return edges implied by longer paths in the layering DAG."""
        if self.layers.number_of_edges() < 2:
            return []

        if not nx.is_directed_acyclic_graph(self.layers):
            raise ValueError("Cannot prune layer relations because the layering graph contains a cycle.")

        reduced_layers = nx.transitive_reduction(self.layers)
        return [
            (pid_above, pid_below)
            for pid_above, pid_below in list(self.layers.edges())
            if not reduced_layers.has_edge(pid_above, pid_below)
        ]

