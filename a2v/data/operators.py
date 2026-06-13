"""A2V dataset operators used by local wrappers and future training code."""

from __future__ import annotations

from diffsynth.core.data.operators import (
    ImageCropAndResize,
    LoadGIF,
    LoadImage,
    LoadVideo,
    RouteByExtensionName,
    RouteByType,
    SequencialProcess,
    ToAbsolutePath,
    ToList,
)


def frame_list_video_operator(
    base_path: str = "",
    max_pixels: int | None = 1920 * 1080,
    height: int | None = None,
    width: int | None = None,
    height_division_factor: int = 16,
    width_division_factor: int = 16,
    num_frames: int = 81,
    time_division_factor: int = 4,
    time_division_remainder: int = 1,
    frame_rate: int = 24,
    fix_frame_rate: bool = False,
):
    """Return a DiffSynth-compatible video operator that accepts PNG lists.

    DiffSynth samples ``video`` and ``vace_video`` independently when they are
    loaded from compressed videos. A2V prepares both streams as aligned PNG
    frame lists, so this operator keeps list inputs fixed while retaining the
    normal image/gif/video routes for debug data.
    """
    frame_processor = ImageCropAndResize(
        height,
        width,
        max_pixels,
        height_division_factor,
        width_division_factor,
    )
    return RouteByType(operator_map=[
        (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
            (("jpg", "jpeg", "png", "webp", "bmp"), LoadImage() >> frame_processor >> ToList()),
            (("gif",), LoadGIF(
                num_frames,
                time_division_factor,
                time_division_remainder,
                frame_processor=frame_processor,
            )),
            (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                num_frames,
                time_division_factor,
                time_division_remainder,
                frame_processor=frame_processor,
                frame_rate=frame_rate,
                fix_frame_rate=fix_frame_rate,
            )),
        ])),
        (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> frame_processor)),
    ])


__all__ = ["frame_list_video_operator"]
