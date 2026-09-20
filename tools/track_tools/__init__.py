"""Utilities for using simulator track geometry outside Unity."""

__all__ = [
    "DirectBezierTrackRegion",
    "MiniMonacoTrack",
    "TrackRegion",
    "VariableWidthTrackRegion",
    "load_region_json",
]


def __getattr__(name):
    if name in __all__:
        from .minimonaco import (
            DirectBezierTrackRegion,
            MiniMonacoTrack,
            TrackRegion,
            VariableWidthTrackRegion,
            load_region_json,
        )

        return {
            "DirectBezierTrackRegion": DirectBezierTrackRegion,
            "MiniMonacoTrack": MiniMonacoTrack,
            "TrackRegion": TrackRegion,
            "VariableWidthTrackRegion": VariableWidthTrackRegion,
            "load_region_json": load_region_json,
        }[name]
    raise AttributeError(name)
