"""Origami paper folding simulation module.

This module provides functionality to simulate paper folding operations
including creasing, splitting, and complex fold operations.
"""

import logging
import math
from typing import Tuple, Optional, Union

import networkx as nx
import numpy as np

from .base import OrigamiBase
from .validation_mixin import OrigamiValidationMixin
from .geometry_mixin import OrigamiGeometryMixin
from .splitting_mixin import OrigamiSplittingMixin
from .fold_traversal_mixin import OrigamiFoldTraversalMixin
from .layering_mixin import OrigamiLayeringMixin
from .unfolding_mixin import OrigamiUnfoldingMixin
from .visualizer import OrigamiVisualizer
from .crease_pattern import CreasePatternValidationResult
from .DCEL import HalfEdge, Vertex
from .config import DEFAULT_PAPER_BACK_COLOR, DEFAULT_PAPER_FRONT_COLOR, EPS

# Don't touch the root logger; let the host application configure logging.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class OrigamiCore(
    OrigamiValidationMixin,
    OrigamiGeometryMixin,
    OrigamiSplittingMixin,
    OrigamiFoldTraversalMixin,
    OrigamiLayeringMixin,
    OrigamiUnfoldingMixin,
    OrigamiBase,
):
    """Core origami class for paper folding simulation.

    Provides functionality for creating, folding, and manipulating
    virtual origami paper with support for complex folding operations.
    """

    def __init__(
        self,
        *,
        front_color: str = DEFAULT_PAPER_FRONT_COLOR,
        back_color: str = DEFAULT_PAPER_BACK_COLOR,
    ) -> None:
        super().__init__(front_color=front_color, back_color=back_color)
        self.visualizer = OrigamiVisualizer(self)
        self.root_logger = logger
        self.folds_information: dict[str | int, dict] = {}

    def _compose_view_transform(self, matrix: np.ndarray, offset: np.ndarray) -> None:
        self._view_matrix = matrix @ self._view_matrix
        self._view_offset = matrix @ self._view_offset + offset

    def add_vertex(self, edge: Tuple[int, int], position: float) -> int:
        """Add a vertex on an edge at a given position and split the half-edges."""
        return self._insert_vertex_on_edge(edge, position)

    def _insert_vertex_on_edge(self, edge: Tuple[int, int], position: float) -> int:
        # Skips Origami's crease-pattern validation, which would reject the
        # pattern while a fold is still mid-traversal.
        self.root_logger.debug(f"Adding vertex on edge {edge} at position {position}")
        self._check_position_valid(position)
        v1_id, v2_id = edge

        if v1_id not in self.vertices or v2_id not in self.vertices:
            missing = [vid for vid in (v1_id, v2_id) if vid not in self.vertices]
            raise ValueError(f"edge references unknown vertex id(s): {missing}")

        if edge not in self.half_edges:
            raise ValueError(f"edge {edge} does not exist.")

        # New vertex
        p1, p2 = self.vertices[v1_id].pos, self.vertices[v2_id].pos
        new_pos = p1 + position * (p2 - p1)
        new_vid = max(self.vertices.keys()) + 1
        new_v = Vertex(id=new_vid, pos=new_pos)
        self.vertices[new_vid] = new_v

        # extracting the half-edges corresponding to the edge being split
        he = self.half_edges.pop((v1_id, v2_id))
        twin = self.half_edges.pop((v2_id, v1_id))

        # New half-edges for the split edge
        he_second_half = HalfEdge(he.type)
        twin_second_half = HalfEdge(twin.type)

        # Rewire the original half-edge (v1 -> v2) to (v1 -> new_vid -> v2)
        self.half_edges[(v1_id, new_vid)] = he
        self.half_edges[(new_vid, v2_id)] = he_second_half

        original_he_next = he.next
        he.next = he_second_half
        he_second_half.origin = new_v
        he_second_half.face = he.face
        he_second_half.next = original_he_next

        # Rewire the original half-edge (v2 -> v1) to (v2 -> new_vid -> v1)
        self.half_edges[(v2_id, new_vid)] = twin
        self.half_edges[(new_vid, v1_id)] = twin_second_half

        original_twin_next = twin.next
        twin.next = twin_second_half
        twin_second_half.origin = new_v
        twin_second_half.face = twin.face
        twin_second_half.next = original_twin_next

        # Link the twins
        he.twin = twin_second_half
        twin_second_half.twin = he

        he_second_half.twin = twin
        twin.twin = he_second_half

        # Linking the new vertex to one of its outgoing half-edges
        new_v.he = he_second_half
        self.root_logger.debug(f"Created new vertex {new_vid} at position {new_pos}")

        return new_vid

    def fold(
        self,
        edge: Tuple[int, int],
        side: int,
        fold_id: str | int | None = None,
    ) -> set[int]:

        vid1, vid2 = edge
        self._check_vid_exists(vid1)
        self._check_vid_exists(vid2)

        # If the edge already exists, and there is a face on the folding side
        if edge in self.half_edges:
            face_on_side = self._is_there_face_on_side_of_edge(edge, side)
            if face_on_side:
                return self._fold_along_edge(edge, side, fold_id)
        # If the vertices of the edge are in the same face
        elif self._get_face_containing_the_two_vertices(edge[0], edge[1]) is not None:
            return self._fold_along_edge(edge, side, fold_id)
        # If the vertices of the edge are in the same plane
        local_edge = self._get_plane_containing_the_two_vertices(edge[0], edge[1])
        if local_edge is not None:
            return self._fold_along_edge(local_edge, side, fold_id)

        # Otherwise, fold against the top face that the line crosses
        fid, cut_edge = self._find_highest_face_that_intersects_line(edge)
        if fid is None or cut_edge is None:
            raise ValueError(f"No face intersects with the fold line {edge}")
        return self._fold_along_edge(cut_edge, side, fold_id)

    def flip(self, axis: str = 'x') -> None:
        if axis == 'x':
            transform_matrix = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=float)
            transform_offset = np.array([1.0, 0.0], dtype=float)
        elif axis == 'y':
            transform_matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
            transform_offset = np.array([0.0, 1.0], dtype=float)
        elif axis == 'y=x':
            transform_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
            transform_offset = np.array([0.0, 0.0], dtype=float)
        elif axis == 'y=-x':
            transform_matrix = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=float)
            transform_offset = np.array([0.0, 0.0], dtype=float)
        else:
            raise ValueError(f"Invalid axis {axis}, expected 'x', 'y', 'y=x', or 'y=-x'")

        for vertex in self.vertices.values():
            transformed_xy = transform_matrix @ np.asarray(vertex.pos[:2], dtype=float) + transform_offset
            vertex.pos[:2] = transformed_xy
        new_layers = nx.DiGraph()
        for layer in self.layers.edges:
            new_layers.add_edge(layer[1], layer[0])
        for node in self.layers.nodes:
            if node not in new_layers:
                new_layers.add_node(node)
        self.layers = new_layers
        for face in self.faces.values():
            face.orientation = 1 - face.orientation
        self._swap_mountain_valley_edges()
        self._compose_view_transform(transform_matrix, transform_offset)

    def rotate(self, degrees: float, center: Optional[Union[Tuple[float, float], np.ndarray]] = None) -> None:
        """Rotate all vertices in the XY plane.

        Positive angles are clockwise.
        If center is None, vertices are rotated around the current centroid.
        """
        if len(self.vertices) == 0:
            return

        if center is None:
            center_xy = np.mean([v.pos[:2] for v in self.vertices.values()], axis=0)
        else:
            center_xy = np.asarray(center, dtype=float)
            if center_xy.shape[0] < 2:
                raise ValueError("center must contain at least 2 coordinates")
            center_xy = center_xy[:2]

        # Standard rotation matrix is CCW-positive, so negate for clockwise-positive.
        theta = math.radians(-degrees)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
        translation = center_xy - rotation @ center_xy

        for vertex in self.vertices.values():
            pos = np.asarray(vertex.pos, dtype=float)
            if pos.shape[0] < 2:
                raise ValueError(f"vertex {vertex.id} position must contain at least 2 coordinates")
            rotated_xy = rotation @ pos[:2] + translation
            new_pos = pos.copy()
            new_pos[:2] = rotated_xy
            vertex.pos = new_pos
        self._compose_view_transform(rotation, translation)

    def unfold(self, edge: Tuple[int, int]) -> set[int]:
        # If edge isn't a direct half-edge key, resolve it from its two vertex ids
        if edge not in self.half_edges:
            edge = self._find_edge_on_line_between_two_vertices(edge)
        self._check_edge_exists(edge)
        self._check_edge_is_folded(edge)

        initial_face_to_unfold = self._get_higher_face_of_edge(edge)

        side = self._get_unfold_side(initial_face_to_unfold, edge)

        # Traverse and collect the folded-side faces; no splitting needed
        # since unfolding never introduces new creases.
        fids_to_unfold_vs_unfold_edge = self._get_fids_to_fold(
            initial_face_to_unfold,
            edge,
            side,
            split_intersected_faces=False,
        )
        fids_to_unfold = list(fids_to_unfold_vs_unfold_edge.keys())
        self.root_logger.debug(f"Faces to unfold: {fids_to_unfold}")
        self.root_logger.debug(f"Edges to unfold: {fids_to_unfold_vs_unfold_edge.values()}")

        # Collect planes
        planes_to_unfold = self._get_planes_to_fold(fids_to_unfold).keys()
        self.root_logger.debug(f"Planes to unfold: {planes_to_unfold}")

        # Flip the relations between the planes that are folded
        self._flip_relations_between_folded_planes(planes_to_unfold)

        # Place the folded plane together with its original plane
        self._flatten_planes(planes_to_unfold, fids_to_unfold_vs_unfold_edge)
        self._update_fold_edges_type(fids_to_unfold_vs_unfold_edge)

        # Update the faces orientations
        self._flip_face_orientation(fids_to_unfold)

        # Reflect the vertices of the faces that are folded
        self._reflect_vertex_across_line(edge, fids_to_unfold, side)
        self._prune()

        self.root_logger.debug("Unfold complete.")

        return set(fids_to_unfold)

    def plot(
        self,
        show_vertices_indices: bool = False,
        show_faces_indices: bool = False,
        show: bool = True,
        save_path: str | None = None,
    ) -> None:
        """Plot the current state of the origami using the visualizer."""
        return self.visualizer.plot(
            show_vertices_indices=show_vertices_indices,
            show_faces_indices=show_faces_indices,
            show=show,
            save_path=save_path,
        )

    def plot_cp(
        self,
        show: bool = True,
        save_path: str | None = None,
    ) -> None:
        """Plot the crease pattern directly from the current origami state."""
        crease_pattern = self.to_crease_pattern(unfolded=True)
        return crease_pattern.plot(show=show, save_path=save_path)

    def validate_crease_pattern(
        self,
        unfolded: bool = True,
        angle_tolerance: float = 1.5e-2,
        point_tolerance: float = EPS,
        require_mv_assignment: bool = False,
    ) -> CreasePatternValidationResult:
        """Validate the current crease pattern with practical flat-foldability checks."""
        crease_pattern = self.to_crease_pattern(unfolded=unfolded)
        return crease_pattern.validate_flat_foldability(
            angle_tolerance=angle_tolerance,
            point_tolerance=point_tolerance,
            require_mv_assignment=require_mv_assignment,
        )
