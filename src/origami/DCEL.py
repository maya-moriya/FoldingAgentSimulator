from __future__ import annotations

import numpy as np


class HalfEdge:
    def __init__(self, type: str = 'B') -> None:
        self.origin: Vertex | None = None
        self.twin: HalfEdge | None = None
        self.face: Face | None = None
        self.next: HalfEdge | None = None
        self.type: str = type    # 'B'=boundary, 'M'=mountain, 'V'=valley, 'F'=flat

class Vertex:
    def __init__(self, id: int, pos: np.ndarray) -> None:
        self.id: int = id
        self.pos: np.ndarray = pos
        self.edge: HalfEdge | None = None    # One outgoing HalfEdge
        self.he: HalfEdge | None = None

class Face:
    def __init__(self, id: int, orientation: int, plane_id: int | None = None) -> None:
        self.id: int = id
        self.orientation: int = orientation
        self.plane_id: int | None = plane_id
        self.edge: HalfEdge | None = None    # One HalfEdge on this face's own loop

class Plane:
    def __init__(self, id: int, face_ids: list[int] | None = None) -> None:
        self.id: int = id
        self.face_ids: list[int] = face_ids if face_ids is not None else []
