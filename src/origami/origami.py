from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np

from .config import DEFAULT_PAPER_BACK_COLOR, DEFAULT_PAPER_FRONT_COLOR, EPS
from .colors import resolve_paper_color
from .crease_pattern import CreasePattern, CreasePatternValidationResult
from .fold import cp_to_fold, fold_to_json, fold_to_origami, is_fold_format, origami_to_fold
from .core import OrigamiCore
from .representation import OrigamiRepresentation
from .visualizer import OrigamiVisualizer


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class Origami(OrigamiCore):
    def __init__(
        self,
        representation: str | dict | None = None,
        *,
        front_color: str | None = None,
        back_color: str | None = None,
    ) -> None:
        resolved_front_color = DEFAULT_PAPER_FRONT_COLOR if front_color is None else front_color
        resolved_back_color = DEFAULT_PAPER_BACK_COLOR if back_color is None else back_color
        super().__init__(front_color=resolved_front_color, back_color=resolved_back_color)
        if representation is not None:
            if is_fold_format(representation):
                origami = fold_to_origami(representation, origami_factory=self.__class__)
            else:
                origami = OrigamiRepresentation.to_origami(
                    representation,
                    origami_factory=self.__class__,
                )
            self.__dict__.update(origami.__dict__)
            if front_color is not None:
                self.paper_front_color = resolve_paper_color(front_color)
            if back_color is not None:
                self.paper_back_color = resolve_paper_color(back_color)
            self.visualizer = OrigamiVisualizer(self)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"vertices={len(self.vertices)}, faces={len(self.faces)}, "
            f"front_color={self.paper_front_color!r}, back_color={self.paper_back_color!r})"
        )

    def _ensure_valid_crease_pattern(self, operation: str) -> None:
        result = self.validate_crease_pattern()
        if result.is_valid:
            return

        issues = result.errors or ("Unknown crease pattern validation failure.",)
        raise ValueError(
            f"Invalid crease pattern after {operation}: {'; '.join(issues)}"
        )

    def add_vertex(self, edge: Tuple[int, int], position: float) -> int:
        new_vertex = OrigamiCore.add_vertex(self, edge, position)
        self._ensure_valid_crease_pattern("add_vertex")
        return new_vertex

    def fold(self, edge: Tuple[int, int], side: int) -> set[int]:
        fids_to_fold = OrigamiCore.fold(self, edge, side)
        self._ensure_valid_crease_pattern("fold")
        return fids_to_fold

    def unfold(self, edge: Tuple[int, int]) -> set[int]:
        fids_to_unfold = OrigamiCore.unfold(self, edge)
        self._ensure_valid_crease_pattern("unfold")
        return fids_to_unfold

    def flip(self, axis: str) -> None:
        OrigamiCore.flip(self, axis)
        self._ensure_valid_crease_pattern("flip")

    def rotate(
        self,
        degrees: float,
        center: Tuple[float, float] | np.ndarray | None = None,
    ) -> None:
        OrigamiCore.rotate(self, degrees, center)
        self._ensure_valid_crease_pattern("rotate")

    def export(
        self,
        *,
        FOLD: bool = False,
        save_path: str | None = None,
        indent: int | None = 2,
    ) -> dict:
        """Export the current state as a plain dict, or a FOLD-format dict when FOLD=True."""
        if not FOLD:
            return OrigamiRepresentation.export(
                self,
                save_path=save_path,
            )

        fold_dict = origami_to_fold(self)
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(save_path).write_text(json.dumps(fold_dict, indent=indent), encoding="utf-8")
        return fold_dict

    def export_cp(
        self,
        *,
        save_path: str | None = None,
    ) -> dict:
        """Export the unfolded crease pattern as a FOLD-format dict."""
        cp_dict = cp_to_fold(self)

        if save_path is not None:
            _write(Path(save_path), fold_to_json(cp_dict))

        return cp_dict

    @classmethod
    def from_representation(cls, representation: str | dict) -> Origami:
        """Build an Origami from the dict/JSON produced by export()."""
        return OrigamiRepresentation.to_origami(representation, origami_factory=cls)

    def to_crease_pattern(self, unfolded: bool = True) -> CreasePattern:
        """Extract the crease pattern as a CreasePattern."""
        return CreasePattern.from_origami(self, unfolded=unfolded)

    def validate_crease_pattern(
        self,
        unfolded: bool = True,
        angle_tolerance: float = 2e-2,
        point_tolerance: float = EPS,
        require_mv_assignment: bool = False,
    ) -> CreasePatternValidationResult:
        return super().validate_crease_pattern(
            unfolded=unfolded,
            angle_tolerance=angle_tolerance,
            point_tolerance=point_tolerance,
            require_mv_assignment=require_mv_assignment,
        )

    def plot(
        self,
        show: bool = True,
        save_path: str | None = None,
        debug: bool = True,
    ) -> None:
        """Render the folded origami. If given, save_path must end in .png."""
        if save_path is not None and Path(save_path).suffix.lower() != ".png":
            raise ValueError("save_path must end in .png")
        return super().plot(
            show_vertices_indices=debug,
            show_faces_indices=debug,
            show=show,
            save_path=save_path,
        )

    def plot_cp(
        self,
        show: bool = True,
        save_path: str | None = None,
    ) -> None:
        """Render the crease pattern. If given, save_path must end in .png."""
        if save_path is not None and Path(save_path).suffix.lower() != ".png":
            raise ValueError("save_path must end in .png")
        return super().plot_cp(show=show, save_path=save_path)
