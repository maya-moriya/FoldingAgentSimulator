from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx
import numpy as np

from .config import DEFAULT_PAPER_BACK_COLOR, DEFAULT_PAPER_FRONT_COLOR
from .DCEL import Face, HalfEdge, Plane, Vertex
from .core import OrigamiCore


class OrigamiRepresentation:
    CREASE_PATTERN_GRID_SIZE = 0.02
    CREASE_PATTERN_ASSIGNMENT_MISMATCH_TOLERANCE = 0.3
    EXPORT_INDENT = 2

    @staticmethod
    def _format_scalar(value: float) -> str:
        return f"{float(value):.6f}"

    @staticmethod
    def _normalize_number(value: float) -> int | float:
        normalized = float(OrigamiRepresentation._format_scalar(value))
        if normalized.is_integer():
            return int(normalized)
        return normalized

    @staticmethod
    def _is_face_above(origami: OrigamiCore, upper_face: Face, lower_face: Face) -> bool:
        upper_pid = upper_face.plane_id
        lower_pid = lower_face.plane_id

        if upper_pid == lower_pid:
            return False
        if nx.has_path(origami.layers, upper_pid, lower_pid):
            return True
        if nx.has_path(origami.layers, lower_pid, upper_pid):
            return False

        raise ValueError(
            f"Could not determine the stacking order between faces {upper_face.id} and {lower_face.id}."
        )

    @staticmethod
    def _get_edge_type(origami: OrigamiCore, face1: Face, face2: Face) -> str:
        if face1.plane_id == face2.plane_id:
            return "F"

        if OrigamiRepresentation._is_face_above(origami, face1, face2):
            lower_face = face2
        else:
            lower_face = face1

        return "V" if lower_face.orientation == 0 else "M"

    @staticmethod
    def get_origami_layers(origami: OrigamiCore) -> list[list[int]]:
        if not nx.is_directed_acyclic_graph(origami.layers):
            raise ValueError("The layers are cyclic, cannot determine layering.")

        reversed_graph = origami.layers.reverse()
        layers = list(nx.topological_generations(reversed_graph))
        return [list(layer) for layer in layers]

    @staticmethod
    def _parse_numeric_suffix(value: str, prefix: str | None = None) -> int:
        token = (value or "").strip()
        if not token:
            raise ValueError("Expected a non-empty identifier.")

        if prefix and token.startswith(prefix):
            token = token[len(prefix):]

        match = re.search(r"(-?\d+)$", token)
        if match is None:
            raise ValueError(f"Could not parse identifier from {value!r}.")
        return int(match.group(1))

    @staticmethod
    def _coerce_id(value: object, *, context: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{context} must be an integer identifier.")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise ValueError(f"{context} must be an integer identifier.")
        if isinstance(value, str):
            return OrigamiRepresentation._parse_numeric_suffix(value)
        raise ValueError(f"{context} must be an integer identifier.")

    @staticmethod
    def _parse_position(value: object) -> np.ndarray:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                f"Invalid vertex position {value!r}. Expected a two-item array [x, y]."
            )
        return np.array([float(value[0]), float(value[1])], dtype=float)

    @staticmethod
    def _parse_id_list(value: object, *, context: str) -> list[int]:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{context} must be an array of identifiers.")
        return [OrigamiRepresentation._coerce_id(item, context=context) for item in value]

    @staticmethod
    def _normalize_edge_key(vid1: int, vid2: int) -> tuple[int, int]:
        if vid1 == vid2:
            raise ValueError("Edges must connect two distinct vertices.")
        return (vid1, vid2) if vid1 < vid2 else (vid2, vid1)

    @staticmethod
    def _parse_edge_type(value: object) -> str:
        edge_type = str(value).strip()
        if edge_type not in {"V", "M", "B", "F"}:
            raise ValueError(
                f"Invalid edge type {value!r}. Expected one of V, M, B, or F."
            )
        return edge_type

    @staticmethod
    def _serialize_unique_edges(origami: OrigamiCore) -> list[list[int | str]]:
        seen_edges = set()
        edges = []
        for fid in sorted(origami.faces):
            face_vids = origami._face_vids(fid)
            for index, origin_vid in enumerate(face_vids):
                next_vid = face_vids[(index + 1) % len(face_vids)]
                normalized_edge = OrigamiRepresentation._normalize_edge_key(origin_vid, next_vid)
                if normalized_edge in seen_edges:
                    continue
                seen_edges.add(normalized_edge)
                half_edge = origami.half_edges[(origin_vid, next_vid)]
                edge_type = half_edge.type
                if edge_type is None and half_edge.twin is not None:
                    edge_type = half_edge.twin.type
                if edge_type is None:
                    raise ValueError(f"Could not determine edge type for edge {normalized_edge}.")
                edges.append([origin_vid, next_vid, edge_type])
        return edges

    @staticmethod
    def _build_boundary_next_map(origami: OrigamiCore, prev_by_half_edge: dict[HalfEdge, HalfEdge]) -> None:
        for half_edge in list(origami.half_edges.values()):
            if half_edge.face is not None or half_edge.twin is None or half_edge.twin.face is None:
                continue

            walker = prev_by_half_edge[half_edge.twin]
            visited = set()
            while True:
                twin = walker.twin
                if twin is None:
                    raise ValueError("Malformed DCEL: half-edge is missing its twin.")
                if twin.face is None:
                    half_edge.next = twin
                    break

                twin_id = id(twin)
                if twin_id in visited:
                    raise ValueError("Malformed representation: could not close a boundary cycle.")
                visited.add(twin_id)
                walker = prev_by_half_edge[twin]

    @staticmethod
    def _assign_vertex_half_edges(origami: OrigamiCore) -> None:
        for vertex in origami.vertices.values():
            vertex.edge = None
            vertex.he = None

        for half_edge in origami.half_edges.values():
            if half_edge.origin is None:
                continue

            vertex = half_edge.origin
            if vertex.edge is None or half_edge.face is not None:
                vertex.edge = half_edge
                vertex.he = half_edge

    @staticmethod
    def _to_dict(origami: OrigamiCore) -> dict:
        payload = {
            "vertices": {
                str(vid): [
                    OrigamiRepresentation._normalize_number(vertex.pos[0]),
                    OrigamiRepresentation._normalize_number(vertex.pos[1]),
                ]
                for vid, vertex in sorted(origami.vertices.items())
            },
            "faces": {
                str(fid): [int(vid) for vid in origami._face_vids(fid)]
                for fid in sorted(origami.faces)
            },
            "faces_orientations": {
                str(fid): int(face.orientation)
                for fid, face in sorted(origami.faces.items())
            },
            "edges": OrigamiRepresentation._serialize_unique_edges(origami),
            "layers": [
                [[int(fid) for fid in origami.planes[pid].face_ids] for pid in layer]
                for layer in OrigamiRepresentation.get_origami_layers(origami)
            ],
        }
        return payload

    @staticmethod
    def _is_scalar(value: object) -> bool:
        return value is None or isinstance(value, (str, int, float, bool))

    @staticmethod
    def _format_inline(value: object) -> str:
        if OrigamiRepresentation._is_scalar(value):
            return json.dumps(value)

        if isinstance(value, list):
            items = [OrigamiRepresentation._format_inline(item) for item in value]
            if all(OrigamiRepresentation._is_scalar(item) for item in value):
                return f"[{', '.join(items)}]"
            return f"[ {', '.join(items)} ]"

        if isinstance(value, dict):
            items = [
                f"{json.dumps(key)}: {OrigamiRepresentation._format_inline(item)}"
                for key, item in value.items()
            ]
            return f"{{ {', '.join(items)} }}"

        raise TypeError(f"Unsupported representation value: {type(value)!r}")

    @staticmethod
    def _format_pretty(value: object, *, level: int = 0) -> str:
        if OrigamiRepresentation._is_scalar(value):
            return json.dumps(value)

        indent = OrigamiRepresentation.EXPORT_INDENT
        current_indent = ' ' * (indent * level)
        child_indent = ' ' * (indent * (level + 1))

        if isinstance(value, dict):
            if not value:
                return '{}'
            separator = ',\n\n' if level == 0 else ',\n'
            items = [
                f"{child_indent}{json.dumps(key)}: {OrigamiRepresentation._format_pretty(item, level=level + 1)}"
                for key, item in value.items()
            ]
            return '{\n' + separator.join(items) + f'\n{current_indent}' + '}'

        if isinstance(value, list):
            if not value:
                return '[]'

            inline = OrigamiRepresentation._format_inline(value)
            if level != 1 and len(inline) <= 40:
                return inline

            items = [
                f"{child_indent}{OrigamiRepresentation._format_pretty(item, level=level + 1)}"
                for item in value
            ]
            return '[\n' + ',\n'.join(items) + f'\n{current_indent}' + ']'

        raise TypeError(f"Unsupported representation value: {type(value)!r}")

    @staticmethod
    def export(
        origami: OrigamiCore,
        *,
        save_path: str | Path | None = None,
    ) -> dict:
        payload = OrigamiRepresentation._to_dict(origami)

        if save_path is not None:
            serialized_text = OrigamiRepresentation._format_pretty(payload)
            Path(save_path).write_text(serialized_text + "\n", encoding="utf-8")

        return payload

    @staticmethod
    def _get_origami_from_dict(
        data: dict,
        origami_factory: type[OrigamiCore] = OrigamiCore,
    ) -> OrigamiCore:
        if not isinstance(data, dict):
            raise ValueError("Representation root must be an object.")

        vertices_data = data.get("vertices")
        faces_data = data.get("faces")
        if not isinstance(vertices_data, dict) or not isinstance(faces_data, dict):
            raise ValueError("Representation must contain object-valued 'vertices' and 'faces' sections.")

        orientations_data = data.get("faces_orientations")
        layers_data = data.get("layers")
        edges_data = data.get("edges")
        paper_colors_data = data.get("paper_colors")

        vertices = {}
        for raw_vid, raw_position in sorted(
            vertices_data.items(),
            key=lambda item: OrigamiRepresentation._coerce_id(item[0], context="Vertex id"),
        ):
            vid = OrigamiRepresentation._coerce_id(raw_vid, context="Vertex id")
            vertices[vid] = Vertex(id=vid, pos=OrigamiRepresentation._parse_position(raw_position))

        if not vertices:
            raise ValueError("Representation does not define any vertices.")

        face_cycles = {}
        for raw_fid, raw_face_vertices in sorted(
            faces_data.items(),
            key=lambda item: OrigamiRepresentation._coerce_id(item[0], context="Face id"),
        ):
            fid = OrigamiRepresentation._coerce_id(raw_fid, context="Face id")
            face_vertices = OrigamiRepresentation._parse_id_list(
                raw_face_vertices,
                context=f"Face {fid}",
            )
            if len(face_vertices) < 3:
                raise ValueError(f"Face {fid} must contain at least three vertices.")
            for vid in face_vertices:
                if vid not in vertices:
                    raise ValueError(f"Face {fid} references unknown vertex {vid}.")
            face_cycles[fid] = face_vertices

        if not face_cycles:
            raise ValueError("Representation does not define any faces.")

        if not isinstance(orientations_data, dict):
            raise ValueError("Representation must contain object-valued 'faces_orientations'.")

        orientations = {}
        for raw_fid, raw_orientation in orientations_data.items():
            fid = OrigamiRepresentation._coerce_id(raw_fid, context="Face orientation id")
            orientations[fid] = int(raw_orientation)

        missing_orientations = set(face_cycles) - set(orientations)
        if missing_orientations:
            raise ValueError(
                f"Missing orientation entries for face ids: {sorted(missing_orientations)}"
            )

        faces = {
            fid: Face(id=fid, orientation=orientations[fid])
            for fid in sorted(face_cycles)
        }

        if not isinstance(edges_data, list):
            raise ValueError("Representation must contain an array-valued 'edges' section.")
        serialized_edges = {}
        for edge in edges_data:
            if not isinstance(edge, (list, tuple)) or len(edge) not in {2, 3}:
                raise ValueError("Each edge must be [v1, v2] or [v1, v2, edge_type].")
            v1 = OrigamiRepresentation._coerce_id(edge[0], context="Edge vertex")
            v2 = OrigamiRepresentation._coerce_id(edge[1], context="Edge vertex")
            if v1 not in vertices or v2 not in vertices:
                raise ValueError(f"'edges' references unknown vertices ({v1}, {v2}).")
            normalized_edge = OrigamiRepresentation._normalize_edge_key(v1, v2)
            edge_type = (
                OrigamiRepresentation._parse_edge_type(edge[2])
                if len(edge) == 3
                else None
            )
            existing_edge_type = serialized_edges.get(normalized_edge)
            if (
                existing_edge_type is not None
                and edge_type is not None
                and existing_edge_type != edge_type
            ):
                raise ValueError(
                    f"Edge {normalized_edge} is assigned multiple edge types: "
                    f"{existing_edge_type!r} and {edge_type!r}."
                )
            serialized_edges[normalized_edge] = edge_type or existing_edge_type

        planes = {}
        ordered_plane_layers = []
        face_to_plane = {}
        if not isinstance(layers_data, list):
            raise ValueError("Representation must contain an array-valued 'layers' section.")
        next_plane_id = 1
        for layer in layers_data:
            if not isinstance(layer, list):
                raise ValueError("Each layer must be an array of planes.")
            plane_ids = []
            for plane_faces in layer:
                plane_face_ids = OrigamiRepresentation._parse_id_list(
                    plane_faces,
                    context=f"Plane {next_plane_id}",
                )
                if not plane_face_ids:
                    raise ValueError(f"Plane {next_plane_id} must contain at least one face.")

                for fid in plane_face_ids:
                    if fid not in faces:
                        raise ValueError(f"Plane {next_plane_id} references unknown face {fid}.")
                    if fid in face_to_plane:
                        raise ValueError(
                            f"Face {fid} is assigned to multiple planes: "
                            f"{face_to_plane[fid]} and {next_plane_id}."
                        )
                    face_to_plane[fid] = next_plane_id
                    faces[fid].plane_id = next_plane_id

                planes[next_plane_id] = Plane(id=next_plane_id, face_ids=list(plane_face_ids))
                plane_ids.append(next_plane_id)
                next_plane_id += 1

            if plane_ids:
                ordered_plane_layers.append(plane_ids)

        if not ordered_plane_layers:
            raise ValueError("Representation does not define any layers.")

        unassigned_faces = set(faces) - set(face_to_plane)
        if unassigned_faces:
            raise ValueError(f"Faces missing from 'layers': {sorted(unassigned_faces)}")

        front_color = DEFAULT_PAPER_FRONT_COLOR
        back_color = DEFAULT_PAPER_BACK_COLOR
        if paper_colors_data is not None:
            if not isinstance(paper_colors_data, dict):
                raise ValueError("Representation field 'paper_colors' must be an object when provided.")

            front_color = paper_colors_data.get("front", front_color)
            back_color = paper_colors_data.get("back", back_color)
            if not isinstance(front_color, str) or not isinstance(back_color, str):
                raise ValueError("Representation field 'paper_colors' must contain string 'front' and 'back' values.")

        origami = origami_factory(front_color=front_color, back_color=back_color)
        origami.vertices = vertices
        origami.faces = faces
        origami.planes = planes
        origami.half_edges = {}
        origami.layers = nx.DiGraph()

        for pid in sorted(planes):
            origami.layers.add_node(pid)

        for layer_index in range(1, len(ordered_plane_layers)):
            upper_layer = ordered_plane_layers[layer_index]
            lower_layer = ordered_plane_layers[layer_index - 1]
            for upper_pid in upper_layer:
                for lower_pid in lower_layer:
                    origami.layers.add_edge(upper_pid, lower_pid)

        prev_by_half_edge = {}
        for fid in sorted(face_cycles):
            face = origami.faces[fid]
            cycle = face_cycles[fid]
            cycle_half_edges = []

            for index, origin_vid in enumerate(cycle):
                next_vid = cycle[(index + 1) % len(cycle)]
                edge_key = (origin_vid, next_vid)
                if edge_key in origami.half_edges:
                    raise ValueError(
                        f"Malformed representation: directed edge {edge_key} is duplicated across face cycles."
                    )

                half_edge = HalfEdge()
                half_edge.origin = origami.vertices[origin_vid]
                half_edge.face = face
                cycle_half_edges.append(half_edge)
                origami.half_edges[edge_key] = half_edge

            for index, half_edge in enumerate(cycle_half_edges):
                half_edge.next = cycle_half_edges[(index + 1) % len(cycle_half_edges)]
                prev_by_half_edge[half_edge] = cycle_half_edges[(index - 1) % len(cycle_half_edges)]

            face.edge = cycle_half_edges[0]

        boundary_half_edges = {}
        used_serialized_edges = set()
        for edge_key, half_edge in list(origami.half_edges.items()):
            if half_edge.twin is not None:
                continue

            normalized_edge = OrigamiRepresentation._normalize_edge_key(*edge_key)
            serialized_edge_type = serialized_edges.get(normalized_edge)
            if normalized_edge in serialized_edges:
                used_serialized_edges.add(normalized_edge)

            twin_key = (edge_key[1], edge_key[0])
            twin = origami.half_edges.get(twin_key)
            if twin is not None:
                half_edge.twin = twin
                twin.twin = half_edge
                if serialized_edge_type == "B":
                    raise ValueError(
                        f"Internal edge {normalized_edge} cannot be serialized as boundary."
                    )
                edge_type = serialized_edge_type or OrigamiRepresentation._get_edge_type(
                    origami, half_edge.face, twin.face
                )
                half_edge.type = edge_type
                twin.type = edge_type
                continue

            boundary_edge_type = serialized_edge_type or "B"
            if boundary_edge_type != "B":
                raise ValueError(
                    f"Boundary edge {normalized_edge} must be serialized as type 'B'."
                )

            boundary_half_edge = HalfEdge(boundary_edge_type)
            boundary_half_edge.origin = origami.vertices[edge_key[1]]
            boundary_half_edge.face = None
            boundary_half_edge.twin = half_edge
            half_edge.twin = boundary_half_edge
            half_edge.type = boundary_edge_type
            boundary_half_edges[twin_key] = boundary_half_edge

        origami.half_edges.update(boundary_half_edges)
        OrigamiRepresentation._build_boundary_next_map(origami, prev_by_half_edge)
        OrigamiRepresentation._assign_vertex_half_edges(origami)

        actual_edges = {
            OrigamiRepresentation._normalize_edge_key(*edge_key) for edge_key in origami.half_edges
        }
        serialized_edge_keys = set(serialized_edges)
        if serialized_edge_keys != actual_edges:
            missing_edges = sorted(actual_edges - serialized_edge_keys)
            extra_edges = sorted(serialized_edge_keys - actual_edges)
            raise ValueError(
                f"'edges' does not match the faces. Missing edges: {missing_edges}; extra edges: {extra_edges}."
            )

        unused_serialized_edges = serialized_edge_keys - used_serialized_edges
        if unused_serialized_edges:
            raise ValueError(
                f"'edges' references unknown edges: {sorted(unused_serialized_edges)}"
            )

        return origami

    @staticmethod
    def to_origami(
        representation: str | dict,
        origami_factory: type[OrigamiCore] = OrigamiCore,
    ) -> OrigamiCore:
        if isinstance(representation, str):
            serialized_text = representation.strip()
            if not serialized_text:
                raise ValueError("Serialized origami text is empty.")
            data = json.loads(serialized_text)
        elif isinstance(representation, dict):
            data = representation
        else:
            raise TypeError("Representation must be a JSON string or a dictionary.")
        return OrigamiRepresentation._get_origami_from_dict(data, origami_factory=origami_factory)

