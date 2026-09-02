from __future__ import annotations

from .DCEL import Face, HalfEdge


class OrigamiSplittingMixin:
    """Cutting faces along a line when a crease doesn't yet exist."""

    def _split_face(self, fid: int, split_edge: tuple[int, int], side: int) -> tuple[int, int | None]:
        self.root_logger.debug(f"Splitting face {fid} along edge {split_edge} on side {side}")

        vid1, vid2 = split_edge

        self._check_fid_exists(fid)
        self._check_vid_exists(vid1)
        self._check_vid_exists(vid2)

        original_face = self.faces[fid]
        if self._face_has_edge(original_face, split_edge):
            if self.half_edges[split_edge].type == 'F':
                self.root_logger.debug(f"Edge {split_edge} exists and is flat, returning existing faces on each side without splitting")
                return self._get_edge_faces_by_side(split_edge, side)
            elif self._is_fold_type(self.half_edges[split_edge].type):
                self.root_logger.debug(f"Edge {split_edge} exists and is a fold, returning existing faces by layer order without splitting")
                return self._get_higher_face_of_edge(split_edge), None
            else:
                raise ValueError(f"Edge {split_edge} exists but is of type {self.half_edges[split_edge].type}, cannot split along it.")
            
        prev1, prev2 = None, None
        curr = original_face.edge
        while True:
            if curr.next.origin.id == vid1:
                prev1 = curr
            if curr.next.origin.id == vid2:
                prev2 = curr
            curr = curr.next
            if curr == original_face.edge: break
            
        if not prev1 or not prev2:
            raise ValueError("Vertices not found in the specified face.")

        # Create new half-edges for the split edge 
        he_new = HalfEdge('F')
        he_twin = HalfEdge('F')
        self.half_edges[(vid1, vid2)] = he_new
        self.half_edges[(vid2, vid1)] = he_twin

        he_new.origin = self.vertices[vid1]
        he_twin.origin = self.vertices[vid2]
        he_new.twin = he_twin
        he_twin.twin = he_new

        # Rewire the existing half-edges 
        next1 = prev1.next
        next2 = prev2.next
        prev1.next = he_new
        he_new.next = next2
        prev2.next = he_twin
        he_twin.next = next1

        # Create the new face
        new_fid = max(self.faces.keys()) + 1
        new_face = Face(id=new_fid, orientation=original_face.orientation)
        self.root_logger.debug(f"Created new face {new_fid} from splitting face {fid}")

        self.faces[new_fid] = new_face
        original_face.edge = he_new
        new_face.edge = he_twin

        # Assign the new face to the original face's plane
        new_face.plane_id = original_face.plane_id
        self.planes[original_face.plane_id].face_ids.append(new_fid)

        # Update the face references for the half-edges in both faces
        curr = he_twin
        while True:
            curr.face = new_face
            curr = curr.next
            if curr == he_twin: break
        
        curr = he_new
        while True:
            curr.face = original_face
            curr = curr.next
            if curr == he_new: break

        return self._get_edge_faces_by_side(split_edge, side)

    def _cut_face_along_line(self, fid: int, edge: tuple[int, int]) -> tuple[int, int] | None:

        self.root_logger.debug(f"Checking where edge {edge} intersects with face {fid}")
        self._check_fid_exists(fid)
        
        line = self._get_line_equation(edge)
        side_by_vid = self._side_by_vid(line)
        face = self.faces[fid]
        curr_edge = face.edge
        vertices = set()
        
        while True:
            vid = curr_edge.origin.id
            curr_tuple = (curr_edge.origin.id, curr_edge.next.origin.id)
            self.root_logger.debug(f"   Checking where edge {edge} intersects with edge {curr_tuple} of face {fid}")
            
            
            # cutting line and current edge are colinear
            next_vid = curr_edge.next.origin.id
            if (vid in edge and next_vid in edge) or ((vid in side_by_vid) and (next_vid in side_by_vid) and (side_by_vid[vid] == 0) and (side_by_vid[next_vid] == 0)):
                self.root_logger.debug(f"       Edge {curr_tuple} is colinear with the cutting line, adding {vid}, {next_vid} to the set")
                vertices.add(vid)
                vertices.add(next_vid)
            # vertex is part of the cutting edge
            elif vid in edge:
                self.root_logger.debug(f"       Vertex {vid} is part of the cutting edge")
                vertices.add(vid)
            # cutting line and current edge are not colinear, check for intersection
            else:
                intersection_ratio = self._get_edge_line_intersection(line, curr_tuple)
                if intersection_ratio == 0:
                    self.root_logger.debug(f"       Edge {curr_tuple} intersects with the cutting line at the start vertex {vid}")
                    vertices.add(vid)
                elif intersection_ratio == 1:
                    self.root_logger.debug(f"       Edge {curr_tuple} intersects with the cutting line at the end vertex {next_vid}")
                    vertices.add(next_vid)
                elif intersection_ratio is not None and 0 < intersection_ratio < 1:
                    self.root_logger.debug(f"       Edge {curr_tuple} intersects with the cutting line at ratio {intersection_ratio}, adding new vertex")
                    new_vid = self._insert_vertex_on_edge(curr_tuple, intersection_ratio)
                    vertices.add(new_vid)
            
            # move to the next edge
            curr_edge = curr_edge.next

            # if we have come full circle, stop
            if curr_edge == face.edge:
                break
          
        if len(vertices) == 2:
            result = tuple(sorted(vertices))
            self.root_logger.debug(f"Cut of face {fid} produced split edge: {result}")
            return result
        
        else:
            self.root_logger.debug(f"Cut face {fid} produced {len(vertices)} vertices, no split")
            return None

    def _get_face_edge_on_line(self, fid: int, edge: tuple[int, int]) -> tuple[int, int] | None:
        """Return an existing boundary edge of the face that lies on the given line."""
        line = self._get_line_equation(edge)
        candidate = None

        for half_edge in self._walk_face_half_edges(self.faces[fid]):
            if half_edge.origin is None or half_edge.next is None or half_edge.next.origin is None:
                continue

            edge_tuple = (half_edge.origin.id, half_edge.next.origin.id)
            if (
                self._get_vid_side(edge_tuple[0], line) != 0
                or self._get_vid_side(edge_tuple[1], line) != 0
            ):
                continue

            if self._is_fold_type(self.half_edges[edge_tuple].type):
                return edge_tuple
            if candidate is None:
                candidate = edge_tuple

        return candidate
