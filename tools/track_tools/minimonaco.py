"""Mini Monaco track extraction and containment checks.

This module reads the PathCreator Bezier path embedded in Unity's
``mini_monaco.unity`` scene and builds a 2D region for point-in-track checks.
It is intentionally dependency-free so it can be used beside the bicycle model
without requiring Unity or geometry packages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple


Point = Tuple[float, float]
Vector3 = Tuple[float, float, float]

DEFAULT_SCENE_PATH = (
    Path(__file__).resolve().parent
    / "mini_monaco.unity"
)
DEFAULT_REGION_JSON_PATH = Path(__file__).resolve().parent / "mini_monaco_track_region.json"

PATH_CREATOR_POSITION = (-0.24987087, 0.009999999, -0.1518895)
PATH_CREATOR_SCALE = (0.75, 1.0, 0.75)
MINI_MONACO_PARENT_SCALE = (6.5, 6.5, 6.5)
MINI_MONACO_BARRIER_OFFSET = 6.0
WidthFn = Callable[[float], float]


@dataclass(frozen=True)
class TrackRegion:
    """A closed 2D track region in Unity world X/Z coordinates."""

    centerline: List[Point]
    left_boundary: List[Point]
    right_boundary: List[Point]
    polygon: List[Point]
    boundary_offset: float

    @property
    def area(self) -> float:
        return polygon_area(self.polygon)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in self.polygon]
        zs = [p[1] for p in self.polygon]
        return min(xs), min(zs), max(xs), max(zs)

    def contains(self, x: float, z: float) -> bool:
        """Return True if the point is strictly inside the track strip."""

        return self.distance_to_centerline(x, z) < self.boundary_offset

    def covers(self, x: float, z: float, tolerance: float = 1e-9) -> bool:
        """Return True if the point is inside or on the track strip boundary."""

        return self.distance_to_centerline(x, z) <= self.boundary_offset + tolerance

    def distance_to_centerline(self, x: float, z: float) -> float:
        """Return unsigned distance from the point to the sampled centerline."""

        return point_to_polyline_distance((x, z), self.centerline, closed=True)

    def distance_to_boundary(self, x: float, z: float) -> float:
        """Return unsigned distance from the point to the nearest track edge."""

        return abs(self.boundary_offset - self.distance_to_centerline(x, z))

    def signed_margin(self, x: float, z: float, boundary_is_inside: bool = True) -> float:
        """Return positive distance inside the track and negative outside it."""

        margin = self.boundary_offset - self.distance_to_centerline(x, z)
        if boundary_is_inside and abs(margin) <= 1e-9:
            return 0.0
        return margin

    def check_point(self, x: float, z: float) -> dict:
        """Return a compact diagnostic record for one Unity X/Z point."""

        margin = self.signed_margin(x, z)
        return {
            "x": x,
            "z": z,
            "inside": margin >= 0,
            "signed_margin": margin,
            "distance_to_centerline": self.distance_to_centerline(x, z),
            "distance_to_boundary": abs(margin),
        }

    def check_bicycle_pose(
        self,
        x: float,
        y: float,
        origin_x: float = 0.0,
        origin_z: float = 0.0,
        heading_offset_rad: float = 0.0,
        scale: float = 1.0,
    ) -> dict:
        """Check a Donkeycar bicycle pose after mapping planar x/y to Unity x/z.

        By default this assumes the bicycle model's planar ``x`` is Unity ``x``
        and planar ``y`` is Unity ``z``. Use ``origin_*``, ``heading_offset_rad``,
        and ``scale`` if your bicycle pose is in a local frame.
        """

        ux, uz = bicycle_xy_to_unity_xz(
            x,
            y,
            origin_x=origin_x,
            origin_z=origin_z,
            heading_offset_rad=heading_offset_rad,
            scale=scale,
        )
        result = self.check_point(ux, uz)
        result.update({"bicycle_x": x, "bicycle_y": y, "unity_x": ux, "unity_z": uz})
        return result

    def to_json_dict(self) -> dict:
        return {
            "coordinate_frame": "Unity world X/Z",
            "boundary_offset": self.boundary_offset,
            "area": self.area,
            "bounds": {
                "min_x": self.bounds[0],
                "min_z": self.bounds[1],
                "max_x": self.bounds[2],
                "max_z": self.bounds[3],
            },
            "centerline": self.centerline,
            "left_boundary": self.left_boundary,
            "right_boundary": self.right_boundary,
            "polygon": self.polygon,
        }


@dataclass(frozen=True)
class VariableWidthTrackRegion:
    """A closed 2D track region with independent left/right widths."""

    centerline: List[Point]
    left_boundary: List[Point]
    right_boundary: List[Point]
    polygon: List[Point]
    left_widths: List[float]
    right_widths: List[float]

    @property
    def area(self) -> float:
        return polygon_area(self.polygon)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in self.polygon]
        zs = [p[1] for p in self.polygon]
        return min(xs), min(zs), max(xs), max(zs)

    def contains(self, x: float, z: float) -> bool:
        return self.signed_margin(x, z) > 0

    def covers(self, x: float, z: float, tolerance: float = 1e-9) -> bool:
        return self.signed_margin(x, z) >= -tolerance

    def signed_margin(self, x: float, z: float) -> float:
        projection = project_point_to_centerline((x, z), self.centerline)
        left_width, right_width = interpolate_segment_widths(
            projection.segment_index,
            projection.segment_t,
            self.left_widths,
            self.right_widths,
        )
        return min(left_width - projection.signed_lateral_offset,
                   projection.signed_lateral_offset + right_width)

    def check_point(self, x: float, z: float) -> dict:
        projection = project_point_to_centerline((x, z), self.centerline)
        left_width, right_width = interpolate_segment_widths(
            projection.segment_index,
            projection.segment_t,
            self.left_widths,
            self.right_widths,
        )
        margin = min(left_width - projection.signed_lateral_offset,
                     projection.signed_lateral_offset + right_width)
        return {
            "x": x,
            "z": z,
            "inside": margin >= 0,
            "signed_margin": margin,
            "distance_to_centerline": projection.distance_to_centerline,
            "distance_to_boundary": abs(margin),
            "signed_lateral_offset": projection.signed_lateral_offset,
            "left_width": left_width,
            "right_width": right_width,
            "nearest_centerline_x": projection.nearest_point[0],
            "nearest_centerline_z": projection.nearest_point[1],
            "segment_index": projection.segment_index,
            "segment_t": projection.segment_t,
        }

    def check_bicycle_pose(
        self,
        x: float,
        y: float,
        origin_x: float = 0.0,
        origin_z: float = 0.0,
        heading_offset_rad: float = 0.0,
        scale: float = 1.0,
    ) -> dict:
        ux, uz = bicycle_xy_to_unity_xz(
            x,
            y,
            origin_x=origin_x,
            origin_z=origin_z,
            heading_offset_rad=heading_offset_rad,
            scale=scale,
        )
        result = self.check_point(ux, uz)
        result.update({"bicycle_x": x, "bicycle_y": y, "unity_x": ux, "unity_z": uz})
        return result

    def to_json_dict(self) -> dict:
        return {
            "coordinate_frame": "Unity world X/Z",
            "area": self.area,
            "bounds": {
                "min_x": self.bounds[0],
                "min_z": self.bounds[1],
                "max_x": self.bounds[2],
                "max_z": self.bounds[3],
            },
            "centerline": self.centerline,
            "left_boundary": self.left_boundary,
            "right_boundary": self.right_boundary,
            "left_widths": self.left_widths,
            "right_widths": self.right_widths,
            "polygon": self.polygon,
        }


@dataclass(frozen=True)
class CenterlineProjection:
    nearest_point: Point
    segment_index: int
    segment_t: float
    distance_to_centerline: float
    signed_lateral_offset: float


@dataclass(frozen=True)
class DirectBezierTrackRegion:
    """Track checker that evaluates the Bezier curve on demand."""

    control_points: List[Point]
    left_width: float
    right_width: float
    samples_per_segment: int = 80
    left_width_fn: Optional[WidthFn] = None
    right_width_fn: Optional[WidthFn] = None

    @property
    def segment_count(self) -> int:
        return len(self.control_points) // 3

    def contains(self, x: float, z: float) -> bool:
        return self.signed_margin(x, z) > 0

    def covers(self, x: float, z: float, tolerance: float = 1e-9) -> bool:
        return self.signed_margin(x, z) >= -tolerance

    def signed_margin(self, x: float, z: float) -> float:
        return self.check_point(x, z)["signed_margin"]

    def check_point(self, x: float, z: float) -> dict:
        projection = project_point_to_bezier_path(
            (x, z),
            self.control_points,
            self.samples_per_segment,
        )
        progress = (projection.segment_index + projection.segment_t) / self.segment_count
        left_width = self.left_width_fn(progress) if self.left_width_fn else self.left_width
        right_width = self.right_width_fn(progress) if self.right_width_fn else self.right_width
        margin = min(left_width - projection.signed_lateral_offset,
                     projection.signed_lateral_offset + right_width)
        return {
            "x": x,
            "z": z,
            "inside": margin >= 0,
            "signed_margin": margin,
            "distance_to_centerline": projection.distance_to_centerline,
            "distance_to_boundary": abs(margin),
            "signed_lateral_offset": projection.signed_lateral_offset,
            "left_width": left_width,
            "right_width": right_width,
            "nearest_centerline_x": projection.nearest_point[0],
            "nearest_centerline_z": projection.nearest_point[1],
            "segment_index": projection.segment_index,
            "segment_t": projection.segment_t,
            "path_progress": progress,
        }

    def check_bicycle_pose(
        self,
        x: float,
        y: float,
        origin_x: float = 0.0,
        origin_z: float = 0.0,
        heading_offset_rad: float = 0.0,
        scale: float = 1.0,
    ) -> dict:
        ux, uz = bicycle_xy_to_unity_xz(
            x,
            y,
            origin_x=origin_x,
            origin_z=origin_z,
            heading_offset_rad=heading_offset_rad,
            scale=scale,
        )
        result = self.check_point(ux, uz)
        result.update({"bicycle_x": x, "bicycle_y": y, "unity_x": ux, "unity_z": uz})
        return result

    def to_json_dict(self) -> dict:
        return {
            "coordinate_frame": "Unity world X/Z",
            "method": "direct_bezier",
            "control_point_count": len(self.control_points),
            "segment_count": self.segment_count,
            "samples_per_segment_for_projection": self.samples_per_segment,
            "left_width": self.left_width,
            "right_width": self.right_width,
            "has_left_width_fn": self.left_width_fn is not None,
            "has_right_width_fn": self.right_width_fn is not None,
            "control_points": self.control_points,
        }


class MiniMonacoTrack:
    """Factory for Mini Monaco track regions extracted from the Unity scene."""

    @classmethod
    def from_unity_scene(
        cls,
        scene_path: Path | str = DEFAULT_SCENE_PATH,
        samples_per_segment: int = 80,
        boundary_offset: float = MINI_MONACO_BARRIER_OFFSET,
    ) -> TrackRegion:
        points = extract_path_creator_points(Path(scene_path))
        centerline = sample_closed_bezier_path(points, samples_per_segment)
        world_centerline = [
            unity_transform_xz(point, PATH_CREATOR_POSITION, PATH_CREATOR_SCALE, MINI_MONACO_PARENT_SCALE)
            for point in centerline
        ]
        left, right = offset_boundaries(world_centerline, boundary_offset)
        polygon = left + list(reversed(right))
        return TrackRegion(
            centerline=world_centerline,
            left_boundary=left,
            right_boundary=right,
            polygon=polygon,
            boundary_offset=boundary_offset,
        )

    @classmethod
    def from_unity_scene_variable_width(
        cls,
        scene_path: Path | str = DEFAULT_SCENE_PATH,
        samples_per_segment: int = 80,
        left_width: float = MINI_MONACO_BARRIER_OFFSET,
        right_width: float = MINI_MONACO_BARRIER_OFFSET,
        left_width_fn: Optional[WidthFn] = None,
        right_width_fn: Optional[WidthFn] = None,
        left_widths: Optional[Sequence[float]] = None,
        right_widths: Optional[Sequence[float]] = None,
    ) -> VariableWidthTrackRegion:
        """Build a region by following the Bezier path with variable side widths.

        Width functions receive normalized path progress ``t`` in ``[0, 1)``.
        Explicit width arrays must match the sampled centerline length.
        """

        points = extract_path_creator_points(Path(scene_path))
        centerline = sample_closed_bezier_path(points, samples_per_segment)
        world_centerline = [
            unity_transform_xz(point, PATH_CREATOR_POSITION, PATH_CREATOR_SCALE, MINI_MONACO_PARENT_SCALE)
            for point in centerline
        ]
        count = len(world_centerline)
        left_profile = build_width_profile(count, left_width, left_width_fn, left_widths)
        right_profile = build_width_profile(count, right_width, right_width_fn, right_widths)
        left, right = offset_boundaries_variable_width(world_centerline, left_profile, right_profile)
        polygon = left + list(reversed(right))
        return VariableWidthTrackRegion(
            centerline=world_centerline,
            left_boundary=left,
            right_boundary=right,
            polygon=polygon,
            left_widths=left_profile,
            right_widths=right_profile,
        )

    @classmethod
    def from_unity_scene_direct_bezier(
        cls,
        scene_path: Path | str = DEFAULT_SCENE_PATH,
        left_width: float = MINI_MONACO_BARRIER_OFFSET,
        right_width: float = MINI_MONACO_BARRIER_OFFSET,
        samples_per_segment: int = 80,
        left_width_fn: Optional[WidthFn] = None,
        right_width_fn: Optional[WidthFn] = None,
    ) -> DirectBezierTrackRegion:
        """Build an on-demand checker from the Bezier control points only."""

        local_points = extract_path_creator_points(Path(scene_path))
        world_points = [
            unity_transform_xz(point, PATH_CREATOR_POSITION, PATH_CREATOR_SCALE, MINI_MONACO_PARENT_SCALE)
            for point in local_points
        ]
        return DirectBezierTrackRegion(
            control_points=world_points,
            left_width=left_width,
            right_width=right_width,
            samples_per_segment=samples_per_segment,
            left_width_fn=left_width_fn,
            right_width_fn=right_width_fn,
        )


def load_region_json(path: Path | str) -> TrackRegion | VariableWidthTrackRegion | DirectBezierTrackRegion:
    """Load a saved Mini Monaco track region JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("method") == "direct_bezier" or "control_points" in data:
        return DirectBezierTrackRegion(
            control_points=points_from_json(data, "control_points"),
            left_width=float(data.get("left_width", MINI_MONACO_BARRIER_OFFSET)),
            right_width=float(data.get("right_width", MINI_MONACO_BARRIER_OFFSET)),
            samples_per_segment=int(data.get("samples_per_segment_for_projection", 80)),
        )

    centerline = points_from_json(data, "centerline")
    left_boundary = points_from_json(data, "left_boundary")
    right_boundary = points_from_json(data, "right_boundary")
    polygon = points_from_json(data, "polygon")

    if "left_widths" in data or "right_widths" in data:
        if "left_widths" not in data or "right_widths" not in data:
            raise ValueError(f"{path} must contain both left_widths and right_widths")
        return VariableWidthTrackRegion(
            centerline=centerline,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            polygon=polygon,
            left_widths=[float(width) for width in data["left_widths"]],
            right_widths=[float(width) for width in data["right_widths"]],
        )

    if "boundary_offset" not in data:
        raise ValueError(f"{path} must contain boundary_offset for fixed-width regions")
    return TrackRegion(
        centerline=centerline,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        polygon=polygon,
        boundary_offset=float(data["boundary_offset"]),
    )


def points_from_json(data: dict, key: str) -> List[Point]:
    if key not in data:
        raise ValueError(f"region JSON is missing {key}")
    return [(float(point[0]), float(point[1])) for point in data[key]]


def extract_path_creator_points(scene_path: Path) -> List[Vector3]:
    """Extract PathCreator Bezier points from ``mini_monaco.unity``."""

    text = scene_path.read_text(encoding="utf-8")
    object_match = re.search(
        r"m_Name: PathCreator.*?editorData:\s*\n\s*_bezierPath:\s*\n\s*points:\s*\n(?P<points>.*?)\n\s*isClosed:\s*1",
        text,
        flags=re.DOTALL,
    )
    if not object_match:
        raise ValueError(f"Could not find a closed PathCreator Bezier path in {scene_path}")

    point_pattern = re.compile(
        r"- \{x:\s*(?P<x>[-+0-9.eE]+), y:\s*(?P<y>[-+0-9.eE]+), z:\s*(?P<z>[-+0-9.eE]+)\}"
    )
    points = [
        (float(m.group("x")), float(m.group("y")), float(m.group("z")))
        for m in point_pattern.finditer(object_match.group("points"))
    ]
    if len(points) < 4 or len(points) % 3 != 0:
        raise ValueError(
            f"Unexpected PathCreator point count {len(points)}. Closed paths should have 3*N points."
        )
    return points


def sample_closed_bezier_path(points: Sequence[Vector3], samples_per_segment: int) -> List[Vector3]:
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least 2")

    sampled: List[Vector3] = []
    segment_count = len(points) // 3
    for segment_index in range(segment_count):
        i = segment_index * 3
        p0 = points[i]
        p1 = points[(i + 1) % len(points)]
        p2 = points[(i + 2) % len(points)]
        p3 = points[(i + 3) % len(points)]
        for sample_index in range(samples_per_segment):
            t = sample_index / samples_per_segment
            sampled.append(cubic_bezier(p0, p1, p2, p3, t))
    return sampled


def cubic_bezier(p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3, t: float) -> Vector3:
    mt = 1.0 - t
    a = mt * mt * mt
    b = 3.0 * mt * mt * t
    c = 3.0 * mt * t * t
    d = t * t * t
    return (
        a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
        a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
        a * p0[2] + b * p1[2] + c * p2[2] + d * p3[2],
    )


def unity_transform_xz(
    point: Vector3,
    local_position: Vector3,
    local_scale: Vector3,
    parent_scale: Vector3,
) -> Point:
    x = parent_scale[0] * (local_position[0] + local_scale[0] * point[0])
    z = parent_scale[2] * (local_position[2] + local_scale[2] * point[2])
    return x, z


def offset_boundaries(centerline: Sequence[Point], offset: float) -> Tuple[List[Point], List[Point]]:
    left: List[Point] = []
    right: List[Point] = []
    count = len(centerline)
    for index, point in enumerate(centerline):
        prev_point = centerline[(index - 1) % count]
        next_point = centerline[(index + 1) % count]
        tx = next_point[0] - prev_point[0]
        tz = next_point[1] - prev_point[1]
        length = math.hypot(tx, tz)
        if length == 0:
            continue
        tx /= length
        tz /= length
        left_normal = (-tz, tx)
        right_normal = (tz, -tx)
        left.append((point[0] + offset * left_normal[0], point[1] + offset * left_normal[1]))
        right.append((point[0] + offset * right_normal[0], point[1] + offset * right_normal[1]))
    return left, right


def offset_boundaries_variable_width(
    centerline: Sequence[Point],
    left_widths: Sequence[float],
    right_widths: Sequence[float],
) -> Tuple[List[Point], List[Point]]:
    if len(centerline) != len(left_widths) or len(centerline) != len(right_widths):
        raise ValueError("centerline, left_widths, and right_widths must have the same length")

    left: List[Point] = []
    right: List[Point] = []
    count = len(centerline)
    for index, point in enumerate(centerline):
        prev_point = centerline[(index - 1) % count]
        next_point = centerline[(index + 1) % count]
        tx = next_point[0] - prev_point[0]
        tz = next_point[1] - prev_point[1]
        length = math.hypot(tx, tz)
        if length == 0:
            continue
        tx /= length
        tz /= length
        left_normal = (-tz, tx)
        right_normal = (tz, -tx)
        left.append((
            point[0] + left_widths[index] * left_normal[0],
            point[1] + left_widths[index] * left_normal[1],
        ))
        right.append((
            point[0] + right_widths[index] * right_normal[0],
            point[1] + right_widths[index] * right_normal[1],
        ))
    return left, right


def build_width_profile(
    count: int,
    base_width: float,
    width_fn: Optional[WidthFn] = None,
    widths: Optional[Sequence[float]] = None,
) -> List[float]:
    if widths is not None:
        if len(widths) != count:
            raise ValueError(f"width profile has {len(widths)} values, expected {count}")
        return [float(width) for width in widths]
    if width_fn is not None:
        return [float(width_fn(index / count)) for index in range(count)]
    return [float(base_width)] * count


def sinusoidal_width_fn(base: float, amplitude: float, cycles: float = 1.0, phase: float = 0.0) -> WidthFn:
    return lambda t: base + amplitude * math.sin(2.0 * math.pi * cycles * t + phase)


def load_width_profile_csv(path: Path) -> Tuple[List[float], List[float]]:
    """Load width profiles from CSV columns named left_width and right_width."""

    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        missing = {"left_width", "right_width"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        left: List[float] = []
        right: List[float] = []
        for row in reader:
            left.append(float(row["left_width"]))
            right.append(float(row["right_width"]))
    return left, right


def interpolate_segment_widths(
    segment_index: int,
    segment_t: float,
    left_widths: Sequence[float],
    right_widths: Sequence[float],
) -> Tuple[float, float]:
    next_index = (segment_index + 1) % len(left_widths)
    left = left_widths[segment_index] + (
        left_widths[next_index] - left_widths[segment_index]
    ) * segment_t
    right = right_widths[segment_index] + (
        right_widths[next_index] - right_widths[segment_index]
    ) * segment_t
    return left, right


def project_point_to_centerline(point: Point, centerline: Sequence[Point]) -> CenterlineProjection:
    best_projection: Optional[CenterlineProjection] = None
    for segment_index, (start, end) in enumerate(closed_pairs(centerline)):
        projection = project_point_to_segment(point, start, end, segment_index)
        if (
            best_projection is None
            or projection.distance_to_centerline < best_projection.distance_to_centerline
        ):
            best_projection = projection
    if best_projection is None:
        raise ValueError("centerline must contain at least one point")
    return best_projection


def project_point_to_bezier_path(
    point: Point,
    control_points: Sequence[Point],
    samples_per_segment: int,
) -> CenterlineProjection:
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least 2")
    if len(control_points) < 4 or len(control_points) % 3 != 0:
        raise ValueError("closed Bezier control point count must be 3*N")

    best_projection: Optional[CenterlineProjection] = None
    segment_count = len(control_points) // 3
    for bezier_segment_index in range(segment_count):
        i = bezier_segment_index * 3
        p0 = point2_to_vector3(control_points[i])
        p1 = point2_to_vector3(control_points[(i + 1) % len(control_points)])
        p2 = point2_to_vector3(control_points[(i + 2) % len(control_points)])
        p3 = point2_to_vector3(control_points[(i + 3) % len(control_points)])
        prev_point = vector3_to_point2(cubic_bezier(p0, p1, p2, p3, 0.0))

        for sample_index in range(samples_per_segment):
            t0 = sample_index / samples_per_segment
            t1 = (sample_index + 1) / samples_per_segment
            next_point = vector3_to_point2(cubic_bezier(p0, p1, p2, p3, t1))
            projection = project_point_to_segment(
                point,
                prev_point,
                next_point,
                bezier_segment_index,
            )
            segment_t = t0 + (t1 - t0) * projection.segment_t
            projection = CenterlineProjection(
                projection.nearest_point,
                bezier_segment_index,
                segment_t,
                projection.distance_to_centerline,
                projection.signed_lateral_offset,
            )
            if (
                best_projection is None
                or projection.distance_to_centerline < best_projection.distance_to_centerline
            ):
                best_projection = projection
            prev_point = next_point

    if best_projection is None:
        raise ValueError("could not project point onto Bezier path")
    return best_projection


def point2_to_vector3(point: Point) -> Vector3:
    return point[0], 0.0, point[1]


def vector3_to_point2(point: Vector3) -> Point:
    return point[0], point[2]


def project_point_to_segment(
    point: Point,
    start: Point,
    end: Point,
    segment_index: int,
) -> CenterlineProjection:
    px, pz = point
    sx, sz = start
    ex, ez = end
    vx = ex - sx
    vz = ez - sz
    segment_len_sq = vx * vx + vz * vz
    if segment_len_sq == 0:
        nearest = start
        distance = math.hypot(px - sx, pz - sz)
        return CenterlineProjection(nearest, segment_index, 0.0, distance, 0.0)

    t = ((px - sx) * vx + (pz - sz) * vz) / segment_len_sq
    t = max(0.0, min(1.0, t))
    nearest = (sx + t * vx, sz + t * vz)
    tangent_len = math.sqrt(segment_len_sq)
    tx = vx / tangent_len
    tz = vz / tangent_len
    left_normal = (-tz, tx)
    lateral = (px - nearest[0]) * left_normal[0] + (pz - nearest[1]) * left_normal[1]
    distance = math.hypot(px - nearest[0], pz - nearest[1])
    return CenterlineProjection(nearest, segment_index, t, distance, lateral)


def bicycle_xy_to_unity_xz(
    x: float,
    y: float,
    origin_x: float = 0.0,
    origin_z: float = 0.0,
    heading_offset_rad: float = 0.0,
    scale: float = 1.0,
) -> Point:
    cos_h = math.cos(heading_offset_rad)
    sin_h = math.sin(heading_offset_rad)
    sx = x * scale
    sy = y * scale
    return (
        origin_x + sx * cos_h - sy * sin_h,
        origin_z + sx * sin_h + sy * cos_h,
    )


def polygon_area(points: Sequence[Point]) -> float:
    area = 0.0
    for p1, p2 in closed_pairs(points):
        area += p1[0] * p2[1] - p2[0] * p1[1]
    return abs(area) / 2.0


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, z = point
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        previous = polygon[j]
        xi, zi = current
        xj, zj = previous
        if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (zj - zi) + xi):
            inside = not inside
        j = i
    return inside


def point_on_polygon_boundary(point: Point, polygon: Sequence[Point], tolerance: float = 1e-9) -> bool:
    return point_to_polyline_distance(point, polygon, closed=True) <= tolerance


def point_to_polyline_distance(point: Point, polyline: Sequence[Point], closed: bool) -> float:
    distances = [
        point_to_segment_distance(point, start, end)
        for start, end in closed_pairs(polyline) if closed
    ]
    if not closed:
        distances = [
            point_to_segment_distance(point, polyline[index], polyline[index + 1])
            for index in range(len(polyline) - 1)
        ]
    return min(distances)


def point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    px, pz = point
    sx, sz = start
    ex, ez = end
    vx = ex - sx
    vz = ez - sz
    segment_len_sq = vx * vx + vz * vz
    if segment_len_sq == 0:
        return math.hypot(px - sx, pz - sz)
    t = ((px - sx) * vx + (pz - sz) * vz) / segment_len_sq
    t = max(0.0, min(1.0, t))
    closest = (sx + t * vx, sz + t * vz)
    return math.hypot(px - closest[0], pz - closest[1])


def closed_pairs(points: Sequence[Point]) -> Iterator[Tuple[Point, Point]]:
    for index, point in enumerate(points):
        yield point, points[(index + 1) % len(points)]


def annotate_csv(
    input_path: Path,
    output_path: Path,
    region: TrackRegion,
    x_column: str,
    z_column: str,
    bicycle_frame: bool = False,
    origin_x: float = 0.0,
    origin_z: float = 0.0,
    heading_offset_rad: float = 0.0,
    scale: float = 1.0,
) -> None:
    with input_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} has no CSV header")
        for column in (x_column, z_column):
            if column not in reader.fieldnames:
                raise ValueError(f"Column {column!r} not found in {input_path}")

        fieldnames = list(reader.fieldnames)
        for field in ("track_inside", "track_signed_margin", "track_boundary_distance"):
            if field not in fieldnames:
                fieldnames.append(field)

        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                x = float(row[x_column])
                z = float(row[z_column])
                if bicycle_frame:
                    result = region.check_bicycle_pose(
                        x,
                        z,
                        origin_x=origin_x,
                        origin_z=origin_z,
                        heading_offset_rad=heading_offset_rad,
                        scale=scale,
                    )
                else:
                    result = region.check_point(x, z)
                row["track_inside"] = str(result["inside"])
                row["track_signed_margin"] = f"{result['signed_margin']:.9f}"
                row["track_boundary_distance"] = f"{result['distance_to_boundary']:.9f}"
                writer.writerow(row)


def region_summary_widths(region) -> dict:
    if isinstance(region, DirectBezierTrackRegion):
        return {
            "method": "direct_bezier",
            "control_point_count": len(region.control_points),
            "segment_count": region.segment_count,
            "samples_per_segment_for_projection": region.samples_per_segment,
            "left_width": region.left_width,
            "right_width": region.right_width,
            "has_left_width_fn": region.left_width_fn is not None,
            "has_right_width_fn": region.right_width_fn is not None,
        }
    if isinstance(region, VariableWidthTrackRegion):
        return {
            "left_width_min": min(region.left_widths),
            "left_width_max": max(region.left_widths),
            "right_width_min": min(region.right_widths),
            "right_width_max": max(region.right_widths),
        }
    return {"boundary_offset": region.boundary_offset}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini Monaco track-region tools")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument(
        "--region-json",
        type=Path,
        help=(
            "Load a saved sampled-region or direct-Bezier JSON instead of extracting geometry from --scene. "
            f"Default saved region: {DEFAULT_REGION_JSON_PATH}"
        ),
    )
    parser.add_argument("--samples-per-segment", type=int, default=80)
    parser.add_argument("--offset", type=float, default=MINI_MONACO_BARRIER_OFFSET)
    parser.add_argument("--direct-bezier", action="store_true")
    parser.add_argument("--variable-width", action="store_true")
    parser.add_argument("--left-width", type=float, default=MINI_MONACO_BARRIER_OFFSET)
    parser.add_argument("--right-width", type=float, default=MINI_MONACO_BARRIER_OFFSET)
    parser.add_argument("--left-width-variance", type=float, default=0.0)
    parser.add_argument("--right-width-variance", type=float, default=0.0)
    parser.add_argument("--width-variance-cycles", type=float, default=1.0)
    parser.add_argument("--width-profile-csv", type=Path)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--export-json", type=Path)
    parser.add_argument("--check", nargs=2, type=float, metavar=("X", "Z"))
    parser.add_argument("--check-bicycle", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--origin-x", type=float, default=0.0)
    parser.add_argument("--origin-z", type=float, default=0.0)
    parser.add_argument("--heading-offset-rad", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--annotate-csv", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--x-column", default="pos/pos_x")
    parser.add_argument("--z-column", default="pos/pos_z")
    parser.add_argument("--csv-bicycle-frame", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.region_json:
        region = load_region_json(args.region_json)
    elif args.direct_bezier:
        if args.width_profile_csv:
            parser.error("--width-profile-csv is not supported with --direct-bezier")
        left_fn = None
        right_fn = None
        if args.left_width_variance:
            left_fn = sinusoidal_width_fn(
                args.left_width,
                args.left_width_variance,
                args.width_variance_cycles,
            )
        if args.right_width_variance:
            right_fn = sinusoidal_width_fn(
                args.right_width,
                args.right_width_variance,
                args.width_variance_cycles,
            )
        region = MiniMonacoTrack.from_unity_scene_direct_bezier(
            scene_path=args.scene,
            left_width=args.left_width,
            right_width=args.right_width,
            samples_per_segment=args.samples_per_segment,
            left_width_fn=left_fn,
            right_width_fn=right_fn,
        )
    elif args.variable_width or args.width_profile_csv:
        left_widths = None
        right_widths = None
        if args.width_profile_csv:
            left_widths, right_widths = load_width_profile_csv(args.width_profile_csv)
        left_fn = None
        right_fn = None
        if args.left_width_variance:
            left_fn = sinusoidal_width_fn(
                args.left_width,
                args.left_width_variance,
                args.width_variance_cycles,
            )
        if args.right_width_variance:
            right_fn = sinusoidal_width_fn(
                args.right_width,
                args.right_width_variance,
                args.width_variance_cycles,
            )
        region = MiniMonacoTrack.from_unity_scene_variable_width(
            scene_path=args.scene,
            samples_per_segment=args.samples_per_segment,
            left_width=args.left_width,
            right_width=args.right_width,
            left_width_fn=left_fn,
            right_width_fn=right_fn,
            left_widths=left_widths,
            right_widths=right_widths,
        )
    else:
        region = MiniMonacoTrack.from_unity_scene(
            scene_path=args.scene,
            samples_per_segment=args.samples_per_segment,
            boundary_offset=args.offset,
        )

    if args.summary:
        summary = region_summary_widths(region)
        if not isinstance(region, DirectBezierTrackRegion):
            summary = {
                "centerline_points": len(region.centerline),
                "polygon_points": len(region.polygon),
                "area": region.area,
                "bounds": region.bounds,
                **summary,
            }
        print(json.dumps(summary, indent=2))

    if args.export_json:
        args.export_json.write_text(json.dumps(region.to_json_dict(), indent=2), encoding="utf-8")

    if args.check:
        print(json.dumps(region.check_point(args.check[0], args.check[1]), indent=2))

    if args.check_bicycle:
        print(json.dumps(region.check_bicycle_pose(
            args.check_bicycle[0],
            args.check_bicycle[1],
            origin_x=args.origin_x,
            origin_z=args.origin_z,
            heading_offset_rad=args.heading_offset_rad,
            scale=args.scale,
        ), indent=2))

    if args.annotate_csv:
        if args.output_csv is None:
            parser.error("--output-csv is required with --annotate-csv")
        annotate_csv(
            input_path=args.annotate_csv,
            output_path=args.output_csv,
            region=region,
            x_column=args.x_column,
            z_column=args.z_column,
            bicycle_frame=args.csv_bicycle_frame,
            origin_x=args.origin_x,
            origin_z=args.origin_z,
            heading_offset_rad=args.heading_offset_rad,
            scale=args.scale,
        )

    if not any([args.summary, args.export_json, args.check, args.check_bicycle, args.annotate_csv]):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
