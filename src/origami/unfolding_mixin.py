from __future__ import annotations

from typing import Iterable

import networkx as nx


class OrigamiUnfoldingMixin:
    """Merging planes back together when a fold is undone."""

    def _flatten_planes(
        self,
        planes_to_unfold: Iterable[int],
        fids_to_unfold_vs_split_edge: dict[int, tuple[int, int] | None],
    ) -> None:
        visited = set()
        for unfolded_fid, edge_tuple in fids_to_unfold_vs_split_edge.items():
            self.root_logger.debug(f"Processing edge {edge_tuple} for flattening")
            if edge_tuple is None:
                continue
            if edge_tuple in visited or (edge_tuple[1], edge_tuple[0]) in visited:
                continue
            visited.add(edge_tuple)
            edge = self.half_edges[edge_tuple]
            if edge.face.id == unfolded_fid:
                source_pid = edge.face.plane_id
                target_pid = edge.twin.face.plane_id
            else:
                source_pid = edge.twin.face.plane_id
                target_pid = edge.face.plane_id

            # A previous merge in this unfold pass may already place both sides on one plane.
            if source_pid == target_pid:
                continue

            self.root_logger.debug(f"Flattening plane {self._pid(source_pid)} into plane {self._pid(target_pid)} by merging along edge {edge_tuple}")

            if self.layers.has_edge(source_pid, target_pid):
                self.root_logger.debug(f"   Removing layer edge from {self._pid(source_pid)} to {self._pid(target_pid)}")
                self.layers.remove_edge(source_pid, target_pid)

            # Transform all source-plane relations onto the target plane using reachability,
            # so merges behave the same whether the graph is fully connected or already pruned.
            if self.layers.has_node(source_pid):
                layers_snapshot = self.layers.copy()
                for neighbor in nx.ancestors(layers_snapshot, source_pid):
                    if neighbor == target_pid:
                        continue
                    # If target is already above this neighbor, re-adding the opposite
                    # relation would introduce a cycle after pruning-aware rewrites.
                    if nx.has_path(layers_snapshot, target_pid, neighbor):
                        continue
                    self.layers.add_edge(neighbor, target_pid)

                for neighbor in nx.descendants(layers_snapshot, source_pid):
                    if neighbor == target_pid:
                        continue
                    # Likewise, keep the existing ordering if this neighbor is already
                    # above target in the snapshot.
                    if nx.has_path(layers_snapshot, neighbor, target_pid):
                        continue
                    self.layers.add_edge(target_pid, neighbor)

            # Remove the source plane from the layering graph
            if self.layers.has_node(source_pid):
                self.layers.remove_node(source_pid)

            # Transform the source plane to the target plane
            if source_pid not in self.planes or target_pid not in self.planes:
                self.root_logger.debug(
                    f"Skipping flatten merge for source={self._pid(source_pid)}, target={self._pid(target_pid)}: plane already merged"
                )
                continue

            source_fids = self.planes[source_pid].face_ids
            self.planes[target_pid].face_ids.extend(source_fids)

            # Keep face-level plane IDs consistent with the plane merge.
            for fid in source_fids:
                self.faces[fid].plane_id = target_pid

            self.planes.pop(source_pid)

    def _get_unfold_side(self, fid: int, edge_tuple: tuple[int, int]) -> int | None:
        face = self.faces[fid]
        line = self._get_line_equation(edge_tuple)
        for half_edge in self._walk_face_half_edges(face):
            vid = half_edge.origin.id
            if vid in edge_tuple:
                continue
            side = self._get_vid_side(vid, line)
            if side != 0:
                return side
