__all__ = ["run_point_track", "run_rtmpose_csv", "run_segmentation"]


def __getattr__(name: str):
    if name == "run_point_track":
        from .point_track import run_point_track

        return run_point_track
    if name == "run_rtmpose_csv":
        from .rtmpose_csv import run_rtmpose_csv

        return run_rtmpose_csv
    if name == "run_segmentation":
        from .segmentation import run_segmentation

        return run_segmentation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
