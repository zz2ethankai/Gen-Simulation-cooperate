from pathlib import Path
import numpy as np
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import get_relative_transform
from isaacsim.sensors.camera import Camera

def capture_topdown_screenshot(output_dir: str, world, task_cameras: dict | None = None, eye=None, target=None):
    """Capture a top-down screenshot using an existing task camera.

    Temporarily repositions the first available camera to look straight
    down, renders one frame, saves ``topdown_check.png``, then restores
    the original pose.  This works in headless mode because it reuses
    an already-initialised camera with a valid RenderProduct.
    """
    output_path = Path(output_dir) / "topdown_check.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _eye = eye or [0.5, 0.5, 3.0]

    if not task_cameras:
        print("[topdown_check] ERROR: No task cameras available")
        return

    cam_name, cam = next(iter(task_cameras.items()))
    try:
        orig_trans, orig_orient = cam.get_local_pose()
        cam.set_local_pose(translation=np.array(_eye))
        world.render()
        obs = cam.get_observations()
        rgb = obs.get("color_image")
        if rgb is not None and rgb.size > 0:
            from PIL import Image

            img = rgb.astype(np.uint8) if rgb.dtype != np.uint8 else rgb
            Image.fromarray(img).save(str(output_path))
            print(f"[topdown_check] Screenshot saved to {output_path} (via {cam_name})")
        else:
            print("[topdown_check] ERROR: Camera get_observations returned no color_image")
    except Exception as exc:
        print(f"[topdown_check] Failed: {exc}")
    finally:
        try:
            cam.set_local_pose(translation=orig_trans, orientation=orig_orient)
        except Exception:
            pass

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
