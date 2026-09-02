from __future__ import annotations

from typing import Iterable

import networkx as nx

from .DCEL import Plane
from .utils import (
    do_faces_overlap,
)


class OrigamiLayeringMixin:
    """Maintaining the plane-stacking DAG as folds add, split, and reorder layers."""

    def _pid(self, pid: int) -> list[int]:
        # Expand a plane id to its face ids for more readable debug logs.
        return self.planes[pid].face_ids

    def _get_pids_above_pid(self, pid: int) -> set[int]:
        # Get the planes that are above the given pid in the layering graph
        if pid not in self.planes:
            raise ValueError(f"Plane id {pid} does not exist.")
        # Layer edges are directed as (A -> B) meaning A is above B.
        # So planes above pid are graph ancestors, not descendants.
        return set(nx.ancestors(self.layers, pid))

    def _get_pids_above_pids_on_correct_side(
        self,
        pids: Iterable[int],
        edge: tuple[int, int],
        side: int,
    ) -> set[int]:
        side_by_vid = self._side_by_vid(self._get_line_equation(edge))
        pids_on_correct_side = set()
        for pid in pids:
            for fid in self.planes[pid].face_ids:
                face = self.faces[fid]
                for half_edge in self._walk_face_half_edges(face):
                    vid = half_edge.origin.id
                    if side_by_vid[vid] == side:
                        pids_on_correct_side.add(pid)
                        break
        return pids_on_correct_side

    def _remap_relations_to_extensions(
        self,
        planes_to_fold_vs_origin_plane: dict[int, int],
        layers_snapshot: nx.DiGraph,
    ) -> None:
        # Give each newly split-off plane the layer relations its origin plane
        # had before this fold, based on layers_snapshot.
        folded_items = list(planes_to_fold_vs_origin_plane.items())
        for index, (folded_pid, origin_pid) in enumerate(folded_items):
            for other_folded_pid, other_origin_pid in folded_items[index + 1:]:
                if origin_pid == other_origin_pid:
                    continue

                if nx.has_path(layers_snapshot, origin_pid, other_origin_pid):
                    self.layers.add_edge(folded_pid, other_folded_pid)
                elif nx.has_path(layers_snapshot, other_origin_pid, origin_pid):
                    self.layers.add_edge(other_folded_pid, folded_pid)

        split_origin_pids = {
            origin_pid
            for folded_pid, origin_pid in folded_items
            if folded_pid != origin_pid
        }

        for split_origin_pid in split_origin_pids:
            for folded_pid, origin_pid in folded_items:
                if origin_pid == split_origin_pid:
                    continue
                if nx.has_path(layers_snapshot, origin_pid, split_origin_pid):
                    self.layers.add_edge(folded_pid, split_origin_pid)

            for current_pid in list(self.planes.keys()):
                if current_pid == split_origin_pid:
                    continue
                if current_pid in planes_to_fold_vs_origin_plane:
                    continue
                if not layers_snapshot.has_node(current_pid):
                    continue
                if nx.has_path(layers_snapshot, split_origin_pid, current_pid):
                    if self._are_planes_overlapping(split_origin_pid, current_pid):
                        self.layers.add_edge(split_origin_pid, current_pid)

    def _get_planes_to_fold(self, fids_to_fold: Iterable[int]) -> dict[int, int]:
        planes_to_fold_vs_origin_plane = {}
        layers_snapshot = self.layers.copy()

        for pid, plane in list(self.planes.items()):
            new_plane_face_ids = []

            # Collect the faces that are folded and remove them from the original plane
            for fid in list(plane.face_ids):
                if fid in fids_to_fold:
                    new_plane_face_ids.append(fid)
                    self.planes[pid].face_ids.remove(fid)

            # Update the new plane
            if len(new_plane_face_ids) > 0:
                if len(self.planes[pid].face_ids) > 0:
                    new_pid = max(self.planes.keys()) + 1
                    self.root_logger.debug(f"Created new plane {new_pid} for {len(new_plane_face_ids)} folded faces from plane {pid}")
                    self.planes[new_pid] = Plane(id=new_pid, face_ids=new_plane_face_ids)
                    self.layers.add_node(new_pid)
                    planes_to_fold_vs_origin_plane[new_pid] = pid

                    # Update the plane id for the folded faces
                    for fid in new_plane_face_ids:
                        self.faces[fid].plane_id = new_pid

                # The plane is not split, but some of its faces are folded, we need to fold the whole plane
                else:
                    self.planes[pid].face_ids.extend(new_plane_face_ids)
                    planes_to_fold_vs_origin_plane[pid] = pid

        self._remap_relations_to_extensions(
            planes_to_fold_vs_origin_plane,
            layers_snapshot,
        )

        return planes_to_fold_vs_origin_plane

    def _flip_relations_between_folded_planes(self, planes_to_fold: set[int]) -> None:
        layers_snapshot = self.layers.copy()
        visited = set()
        for pid1 in planes_to_fold:
            for pid2 in planes_to_fold:
                if (pid1, pid2) in visited or (pid2, pid1) in visited or pid1 == pid2:
                    continue
                visited.add((pid1, pid2))
                if nx.has_path(layers_snapshot, pid1, pid2):
                    if self.layers.has_edge(pid1, pid2):
                        self.layers.remove_edge(pid1, pid2)
                    self.layers.add_edge(pid2, pid1)
                elif nx.has_path(layers_snapshot, pid2, pid1):
                    if self.layers.has_edge(pid2, pid1):
                        self.layers.remove_edge(pid2, pid1)
                    self.layers.add_edge(pid1, pid2)

    def _drop_conflicting_split_origin_relations(
        self,
        planes_to_fold_vs_origin_plane: dict[int, int],
    ) -> list[tuple[int, int]]:
        """Drop stale split-origin edges that now contradict the flipped folded stack."""
        split_origin_pids = {
            origin_pid
            for folded_pid, origin_pid in planes_to_fold_vs_origin_plane.items()
            if folded_pid != origin_pid
        }
        folded_pids = set(planes_to_fold_vs_origin_plane.keys())
        removed_edges = []

        changed = True
        while changed:
            changed = False
            for origin_pid in split_origin_pids:
                for folded_pid in folded_pids:
                    if planes_to_fold_vs_origin_plane.get(folded_pid) == origin_pid:
                        continue

                    for source_pid, target_pid in (
                        (origin_pid, folded_pid),
                        (folded_pid, origin_pid),
                    ):
                        if not self.layers.has_edge(source_pid, target_pid):
                            continue

                        self.layers.remove_edge(source_pid, target_pid)
                        if nx.has_path(self.layers, target_pid, source_pid):
                            self.root_logger.debug(
                                f"Dropping stale split-origin relation "
                                f"{self._pid(source_pid)} above {self._pid(target_pid)}"
                            )
                            removed_edges.append((source_pid, target_pid))
                            changed = True
                        else:
                            self.layers.add_edge(source_pid, target_pid)

        return removed_edges

    def _drop_conflicting_fold_boundary_relations(self, planes_to_fold: set[int]) -> list[tuple[int, int]]:
        """Drop stale relations that cross the folded/static boundary after restacking."""
        removed_edges = []
        changed = True

        while changed:
            changed = False
            for source_pid, target_pid in list(self.layers.edges()):
                source_is_folded = source_pid in planes_to_fold
                target_is_folded = target_pid in planes_to_fold
                if source_is_folded == target_is_folded:
                    continue

                self.layers.remove_edge(source_pid, target_pid)
                if nx.has_path(self.layers, target_pid, source_pid):
                    self.root_logger.debug(
                        f"Dropping stale folded/static relation "
                        f"{self._pid(source_pid)} above {self._pid(target_pid)}"
                    )
                    removed_edges.append((source_pid, target_pid))
                    changed = True
                else:
                    self.layers.add_edge(source_pid, target_pid)

        return removed_edges

    def _place_folded_plane_above_original(self, planes_to_fold_vs_origin_plane: dict[int, int]) -> None:
        for folded_pid, origin_pid in planes_to_fold_vs_origin_plane.items():
            if folded_pid == origin_pid:
                continue
            self.layers.add_edge(folded_pid, origin_pid)

    def _sort_planes_by_layering(self) -> list[int]:
        # Topological sort of the layering graph to get the planes in order from top to bottom
        sorted_pids = list(nx.topological_sort(self.layers))
        return sorted_pids

    def _find_highest_face_that_intersects_line(
        self,
        edge: tuple[int, int],
    ) -> tuple[int | None, tuple[int, int] | None]:
        pids = self._sort_planes_by_layering()
        fid_with_edge = None, None
        for pid in pids:
            for fid in self.planes[pid].face_ids:
                face = self.faces[fid]
                if self._face_has_edge(face, edge):
                    if edge in self.half_edges and self.half_edges[edge].type == 'F':
                        fid_with_edge = fid, edge
                    continue
                else:
                    cut_edge = self._cut_face_along_line(fid, edge)
                    if cut_edge is not None:
                        if self._face_has_edge(face, cut_edge):
                            cut_half_edge = self.half_edges.get(cut_edge)
                            if cut_half_edge is not None and cut_half_edge.type != 'B':
                                fid_with_edge = fid, cut_edge
                            continue
                        self.root_logger.debug(f"Found face {fid} in plane {pid} that is intersected by the fold line {edge} at edge {cut_edge}")
                        return fid, cut_edge
        return fid_with_edge

    def _get_topmost_planes(self) -> list[int]:
        # return list of planes that have no other planes above them
        topmost_pids = []
        for pid in self.layers.nodes:
            if self.layers.in_degree(pid) == 0:
                topmost_pids.append(pid)
        return topmost_pids

    def _faces_overlap(self, fid1: int, fid2: int) -> bool:
        # Separating-axis test: the faces overlap unless some edge-normal axis
        # of either polygon separates their projections.
        face1 = [self.vertices[edge.origin.id].pos for edge in self._walk_face_half_edges(self.faces[fid1])]
        face2 = [self.vertices[edge.origin.id].pos for edge in self._walk_face_half_edges(self.faces[fid2])]
        return do_faces_overlap(face1, face2)

    def _are_planes_overlapping(self, pid1: int, pid2: int) -> bool:
        # Two planes are overlapping if they have a face that overlaps when projected on the XY plane
        for fid1 in self.planes[pid1].face_ids:
            for fid2 in self.planes[pid2].face_ids:
                if self._faces_overlap(fid1, fid2):
                    return True
        return False

    def _find_overlapping_plane(self, pid: int, planes_to_check: Iterable[int]) -> int | None:
        for other_pid in planes_to_check:
            if pid != other_pid and self._are_planes_overlapping(pid, other_pid):
                return other_pid
        return None

    def _find_plane_to_fold_over(self, folded_pid: int) -> int | None:
        # Find the topmost plane that overlaps with the folded plane, if any, otherwise None
        topmost_pids = self._get_topmost_planes()
        self.root_logger.debug(f"Topmost planes: {[self._pid(pid) for pid in topmost_pids]}")
        overlapping_pid = self._find_overlapping_plane(folded_pid, topmost_pids)

        if overlapping_pid is not None:
            self.root_logger.debug(f"Found overlapping plane {overlapping_pid} for folded plane {folded_pid}")
            return overlapping_pid
        else:
            return None

    def _get_highest_plane_from_list(self, list_of_pids: Iterable[int]) -> int | None:
        # Get the highest plane from the list according to the layering graph
        sorted_pids = self._sort_planes_by_layering()
        for pid in sorted_pids:
            if pid in list_of_pids:
                return pid
        return None

    def _set_first_plane_at_the_top_of_the_current_stack(self, planes_to_fold: set[int]) -> None:
        folded_pid = self._get_highest_plane_from_list(planes_to_fold)
        self.root_logger.debug(f"Folded plane {self._pid(folded_pid)} is the highest among folded planes")
        plane_to_fold_over = self._find_plane_to_fold_over(folded_pid)
        if plane_to_fold_over is not None:
            self.root_logger.debug(f"Placing folded plane {self._pid(folded_pid)} above overlapping plane {self._pid(plane_to_fold_over)}")
            self.layers.add_edge(folded_pid, plane_to_fold_over)
            return

    def _place_folded_planes_above_newly_overlapped_planes(self, planes_to_fold: set[int]) -> None:
        """Restack each reflected folded plane over any static planes it newly lands on."""
        if not planes_to_fold:
            return

        # Lower reflected planes should claim the static planes directly below them first.
        folded_pids = [
            pid
            for pid in reversed(self._sort_planes_by_layering())
            if pid in planes_to_fold
        ]

        for folded_pid in folded_pids:
            if folded_pid not in self.planes:
                continue

            for other_pid in list(self.planes.keys()):
                if other_pid == folded_pid or other_pid in planes_to_fold:
                    continue
                if not self._are_planes_overlapping(folded_pid, other_pid):
                    continue
                if nx.has_path(self.layers, folded_pid, other_pid):
                    continue
                if nx.has_path(self.layers, other_pid, folded_pid):
                    continue

                self.root_logger.debug(
                    f"Post-reflection restack: placing folded plane {self._pid(folded_pid)} "
                    f"above newly overlapped plane {self._pid(other_pid)}"
                )
                self.layers.add_edge(folded_pid, other_pid)
