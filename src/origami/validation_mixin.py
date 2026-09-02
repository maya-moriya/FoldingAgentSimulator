from __future__ import annotations


class OrigamiValidationMixin:
    """Small precondition checks shared across origami operations."""

    def _check_vid_exists(self, vid: int) -> None:
        if vid not in self.vertices:
            raise ValueError(f"vertex with id {vid} does not exist.")

    def _check_fid_exists(self, fid: int) -> None:
        if fid not in self.faces:
            raise ValueError(f"face with id {fid} does not exist.")

    def _is_fold_type(self, edge_type: str) -> bool:
        return edge_type in {"M", "V"}

    def _check_position_valid(self, position: float) -> None:
        if not (0 <= position <= 1):
            raise ValueError("position must be between 0 and 1")

    def _check_edge_exists(self, edge_tuple: tuple[int, int]) -> None:
        if edge_tuple not in self.half_edges:
            raise ValueError(f"Edge {edge_tuple} does not exist.")

    def _check_edge_is_folded(self, edge_tuple: tuple[int, int]) -> None:
        if edge_tuple not in self.half_edges:
            raise ValueError(f"Edge {edge_tuple} does not exist.")
        edge = self.half_edges[edge_tuple]
        if edge.type == 'F':
            raise ValueError(f"Edge {edge_tuple} is flat and cannot be unfolded.")
        elif edge.type == 'B':
            raise ValueError(f"Edge {edge_tuple} is a boundary edge and cannot be unfolded.")
