from pathlib import Path
import math
import shutil
import numpy as np
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import get_relative_transform
from isaacsim.sensors.camera import Camera


_topdown_camera_counter = 0
def capture_topdown_screenshot(
    output_dir: str,
    room_half_extent=(2.777, 2.177),
    room_height=2.8,
    floor_z=0.0,
    width=640,
    height=480,
    *,
    eye=None,
    target=None,
    focal_length_mm=16.0,
    filename="topdown.png",
):
    """Capture an independent scene-overview camera as a PNG.

    Uses the same pose and synchronous replicator pipeline as visual_physics'
    ``diagonal_overview``: a fresh camera + ``force_new`` render product +
    BasicWriter, driven by ``rep.orchestrator.step()``/``wait_until_complete()``
    until the PNG lands on disk.
    """
    import omni.replicator.core as rep

    output_path = (Path(output_dir).resolve()) / str(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    half_x, half_y = room_half_extent
    scene_radius = 0.5 * math.hypot(2.0 * half_x, 2.0 * half_y)
    eye_pos = tuple(
        float(value)
        for value in (
            eye
            if eye is not None
            else (
                -2.0 * half_x * 0.44,
                -2.0 * half_y * 0.44,
                floor_z + room_height + scene_radius * 0.78,
            )
        )
    )
    target_pos = tuple(
        float(value)
        for value in (
            target
            if target is not None
            else (
                0.0,
                0.0,
                floor_z
                + min(max(room_height * 0.22, 0.62), room_height - 0.75),
            )
        )
    )
    if len(eye_pos) != 3 or len(target_pos) != 3:
        raise ValueError("overview camera eye and target must each contain three numbers")
    camera_values = np.asarray((*eye_pos, *target_pos, focal_length_mm), dtype=float)
    if not np.isfinite(camera_values).all() or float(focal_length_mm) <= 0.0:
        raise ValueError("overview camera values must be finite and focal length positive")
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("overview camera resolution must be positive")
    if np.linalg.norm(np.asarray(target_pos) - np.asarray(eye_pos)) <= 1e-9:
        raise ValueError("overview camera eye and target must be different")
    up_axis = (0.0, 0.0, 1.0)

    global _topdown_camera_counter
    _topdown_camera_counter += 1

    camera = rep.create.camera(
        name=f"topdown_debug_{_topdown_camera_counter}",
        position=eye_pos,
        look_at=target_pos,
        look_at_up_axis=up_axis,
        focal_length=float(focal_length_mm),
        clipping_range=(0.01, 1000.0),
    )
    render_product = rep.create.render_product(camera, (int(width), int(height)), force_new=True)
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(output_path.parent),
        rgb=True,
        image_output_format="png",
        frame_padding=4,
    )
    rgb_source = output_path.parent / "rgb_0000.png"
    if rgb_source.exists():
        rgb_source.unlink()
    attached = False
    try:
        writer.attach([render_product])
        attached = True
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=32, pause_timeline=False)
            rep.orchestrator.wait_until_complete()
            if rgb_source.exists():
                break
    finally:
        try:
            if attached:
                writer.detach()
        finally:
            render_product.destroy()
    if rgb_source.exists():
        shutil.move(str(rgb_source), str(output_path))
    if not output_path.exists():
        raise RuntimeError(
            f"[topdown] render did not produce {output_path} (writer output dir={output_path.parent})"
        )
    print(f"[topdown] Screenshot saved to {output_path} (eye={eye_pos}, target={target_pos})")

def _get_annotator(camera: Camera, annotator_name: str):
    custom_annotators = getattr(camera, "_custom_annotators", None)
    if not isinstance(custom_annotators, dict):
        return None
    return custom_annotators.get(annotator_name)


def _get_current_frame(camera: Camera):
    get_current_frame = getattr(camera, "get_current_frame", None)
    if not callable(get_current_frame):
        return None
    try:
        frame = get_current_frame()
    except Exception:
        return None
    return frame if isinstance(frame, dict) else None


def _get_annotator_data(camera: Camera, annotator_name: str):
    frame = _get_current_frame(camera)
    if frame is not None and annotator_name in frame:
        data = frame[annotator_name]
        if data is not None:
            return data
    annotator = _get_annotator(camera, annotator_name)
    if annotator is None:
        return None
    return annotator.get_data()


def _get_frame(frame):
    if isinstance(frame, np.ndarray) and frame.size > 0:
        return frame[:, :, :3]
    return None


def _get_depth(depth):
    if isinstance(depth, np.ndarray) and depth.size > 0:
        if depth.ndim == 3 and depth.shape[-1] == 1:
            return depth[..., 0]
        if depth.ndim == 2:
            return depth
    return None


def _get_rgb_image(camera: Camera):
    output_mode = getattr(camera, "output_mode", "rgb")
    if output_mode == "rgb":
        frame = _get_current_frame(camera)
        if frame is not None and "rgb" in frame:
            frame_data = _get_frame(frame["rgb"])
            if frame_data is not None:
                return frame_data
        return _get_frame(camera.get_rgba())
    if output_mode == "diffuse_albedo":
        return _get_frame(_get_annotator_data(camera, "DiffuseAlbedo"))
    raise NotImplementedError(f"Unsupported output mode: {output_mode}")


def _get_depth_image(camera: Camera):
    frame = _get_current_frame(camera)
    if frame is not None and "distance_to_image_plane" in frame:
        frame_data = _get_depth(frame["distance_to_image_plane"])
        if frame_data is not None:
            return frame_data
    get_depth = getattr(camera, "get_depth", None)
    if callable(get_depth):
        return _get_depth(get_depth())
    return _get_depth(_get_annotator_data(camera, "distance_to_image_plane"))


def _get_object_mask(camera: Camera):
    annotation_data = _get_annotator_data(camera, "semantic_segmentation")
    if (
        not isinstance(annotation_data, dict)
        or "data" not in annotation_data
        or "info" not in annotation_data
    ):
        return None
    info = annotation_data["info"]
    if not isinstance(info, dict) or "idToLabels" not in info:
        return None
    mask = annotation_data["data"]
    if isinstance(mask, np.ndarray) and mask.size > 0:
        return {"mask": mask, "id2labels": info["idToLabels"]}
    return None


def _get_bbox(camera: Camera, bbox_type: str):
    annotation_data = _get_annotator_data(camera, bbox_type)
    if (
        not isinstance(annotation_data, dict)
        or "data" not in annotation_data
        or "info" not in annotation_data
    ):
        return None
    info = annotation_data["info"]
    if not isinstance(info, dict) or "idToLabels" not in info:
        return None
    return annotation_data["data"], info["idToLabels"]


def _get_motion_vectors(camera: Camera):
    annotation_data = _get_annotator_data(camera, "motion_vectors")
    if isinstance(annotation_data, np.ndarray) and annotation_data.size > 0:
        return annotation_data
    return None


def _get_camera2env_pose(camera: Camera):
    prim_path = getattr(camera, "prim_path", None)
    root_prim_path = getattr(camera, "root_prim_path", None)
    if not prim_path or not root_prim_path:
        return None
    return get_relative_transform(get_prim_at_path(prim_path), get_prim_at_path(root_prim_path))


def _get_camera_params(camera: Camera):
    camera_matrix = getattr(camera, "is_camera_matrix", None)
    if camera_matrix is None:
        return None
    if isinstance(camera_matrix, np.ndarray):
        return camera_matrix.tolist()
    return camera_matrix


def get_src(camera: Camera, data_type: str):
    if data_type == "rgb":
        return _get_rgb_image(camera)
    if data_type == "depth":
        return _get_depth_image(camera)
    if data_type == "seg":
        return _get_object_mask(camera)
    if data_type == "bbox2d_tight":
        return _get_bbox(camera, "bounding_box_2d_tight")
    if data_type == "bbox2d_loose":
        return _get_bbox(camera, "bounding_box_2d_loose")
    if data_type == "bbox3d":
        return _get_bbox(camera, "bounding_box_3d")
    if data_type == "motion_vectors":
        return _get_motion_vectors(camera)
    if data_type == "camera2env_pose":
        return _get_camera2env_pose(camera)
    if data_type == "camera_params":
        return _get_camera_params(camera)
    raise NotImplementedError(f"Unsupported source type: {data_type}")
