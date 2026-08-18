import importlib
import os
import time
from fractions import Fraction
from pathlib import Path

from nimbus.components.data.iterator import Iterator
from nimbus.components.data.package import Package
from nimbus.components.data.scene import Scene
from nimbus.components.load import SceneLoader
from nimbus.daemon import ComponentStatus, StatusReporter
from nimbus.daemon.decorators import status_monitor
from nimbus.utils.flags import get_random_seed
from workflows.base import create_workflow


def _resolve_headless_experience() -> str:
    candidate_paths = []
    for env_key in ("ISAAC_SIM_ROOT", "ISAAC_SIM_PATH"):
        value = os.environ.get(env_key, "").strip()
        if value:
            root = Path(value)
            candidate_paths.extend(
                [
                    # Keep the same priority as Isaac Sim 6's
                    # SimulationApp.  The old gym headless experience is
                    # retained only as a legacy fallback below.
                    root / "apps/omni.isaac.sim.python.kit",
                    root / "apps/isaacsim.exp.base.python.kit",
                    root / "apps/isaacsim.exp.base.kit",
                    root / "apps/omni.isaac.sim.headless.native.kit",
                    root / "apps/omni.isaac.sim.python.gym.headless.kit",
                    root / "apps/isaacsim.exp.full.kit",
                ]
            )
    for root in (Path("/isaac-sim"), Path("/workspace/isaac-sim")):
        candidate_paths.extend(
            [
                root / "apps/omni.isaac.sim.python.kit",
                root / "apps/isaacsim.exp.base.python.kit",
                root / "apps/isaacsim.exp.base.kit",
                root / "apps/omni.isaac.sim.headless.native.kit",
                root / "apps/omni.isaac.sim.python.gym.headless.kit",
                root / "apps/isaacsim.exp.full.kit",
            ]
        )

    for experience in candidate_paths:
        if experience.is_file():
            return str(experience)

    # An empty experience lets Isaac Sim 6's SimulationApp resolve the
    # installation-specific default through EXP_PATH.  Passing the removed
    # Isaac Sim 4/5 gym path here makes startup fail before any workflow code
    # is imported when the new image omits that legacy file.
    return ""


def _resolve_experience(configured_path: str, *, headless: bool) -> str:
    configured_path = str(configured_path or "").strip()
    if configured_path:
        candidate = Path(configured_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.is_file():
            return str(candidate.resolve())
        # Isaac Sim 6 renamed the bundled experiences.  Keep old task
        # templates usable while preferring the new Python base experience.
        return _resolve_headless_experience() if headless else ""
    if headless:
        return _resolve_headless_experience()
    return ""


def _apply_optional_setting(simulation_app, simulator: dict, key: str, setting_path: str, cast):
    if key not in simulator:
        return None
    value = _cast_setting_value(simulator[key], cast)
    simulation_app.set_setting(setting_path, value)
    return value


def _cast_setting_value(value, cast):
    if cast is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(value)
    return cast(value)


def _resolve_torch_cuda_device(configured_device):
    if configured_device is None:
        return None
    import torch

    visible_count = torch.cuda.device_count()
    if visible_count <= 0:
        return None

    configured_device = int(configured_device)
    if configured_device < visible_count:
        return configured_device

    return None


def _patch_curobo_quad_triangulation_launch_device(logger) -> None:
    try:
        import warp as wp
    except Exception as exc:
        logger.warning(f"Failed to import Warp for CuRobo launch-device patch: {exc}")
        return

    if getattr(wp.launch, "_simbox_curobo_quad_device_patch", False):
        return

    original_launch = wp.launch

    def launch_with_matching_curobo_quad_device(*args, **kwargs):
        kernel = args[0] if args else kwargs.get("kernel")
        kernel_label = str(
            getattr(kernel, "key", "")
            or getattr(kernel, "name", "")
            or getattr(getattr(kernel, "func", None), "__name__", "")
            or kernel
        )
        has_positional_device = len(args) > 6
        if "_triangulate_quads_kernel" in kernel_label and not has_positional_device and kwargs.get("device") is None:
            inputs = kwargs.get("inputs")
            if inputs is None and len(args) > 2:
                inputs = args[2]
            outputs = kwargs.get("outputs")
            if outputs is None and len(args) > 3:
                outputs = args[3]
            for value in list(inputs or []) + list(outputs or []):
                device = getattr(value, "device", None)
                if device is not None:
                    kwargs["device"] = device
                    break
        return original_launch(*args, **kwargs)

    launch_with_matching_curobo_quad_device._simbox_curobo_quad_device_patch = True
    wp.launch = launch_with_matching_curobo_quad_device
    logger.info("Patched Warp launch device for CuRobo quad mesh triangulation")


def _ensure_simbox_sensor_extension_ready(simulation_app, *, max_wait_sec: float = 30.0) -> None:
    """Enable SimBox's camera dependency before importing its workflow module."""

    from isaacsim.core.utils.extensions import enable_extension

    extension_name = "isaacsim.sensors.camera"
    enable_extension(extension_name)
    deadline = time.monotonic() + max(float(max_wait_sec), 1.0)
    last_error = None
    while time.monotonic() < deadline:
        simulation_app.update()
        try:
            importlib.import_module(extension_name)
            return
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc

    raise RuntimeError(
        "SimBox camera dependency 'isaacsim.sensors.camera' was not ready after enabling its Isaac extension"
    ) from last_error


class EnvLoader(SceneLoader):
    """
    Environment loader that initializes Isaac Sim and loads scenes based on workflow configurations.

    This loader integrates with the workflow system to manage scene loading and task execution.
    It supports two operating modes:
    - Standalone mode (pack_iter=None): Loads tasks directly from workflow configuration
    - Pipeline mode (pack_iter provided): Loads tasks from a package iterator

    It also supports task repetition for data augmentation across different random seeds.

    Args:
        pack_iter (Iterator[Package]): An iterator from the previous component. None for standalone.
        cfg_path (str): Path to the workflow configuration file.
        workflow_type (str): Type of workflow to create (e.g., 'SimBoxDualWorkFlow').
        simulator (dict): Simulator configuration including physics_dt, rendering_dt, headless, etc.
        task_repeat (int): How many times to repeat each task before advancing (-1 means single execution).
        need_preload (bool): Whether to preload assets on scene initialization.
        scene_info (str): Configuration key for scene information in the workflow config.
    """

    def __init__(
        self,
        pack_iter: Iterator[Package],
        cfg_path: str,
        workflow_type: str,
        simulator: dict,
        task_repeat: int = -1,
        need_preload: bool = False,
        scene_info: str = "dining_room_scene_info",
    ):
        init_start_time = time.time()
        super().__init__(pack_iter)

        self.status_reporter = StatusReporter(self.__class__.__name__)
        self.status_reporter.update_status(ComponentStatus.IDLE)
        self.need_preload = need_preload
        self.task_repeat_cnt = task_repeat
        self.task_repeat_idx = 0
        self.workflow_type = workflow_type

        # Parse simulator config
        physics_dt = simulator.get("physics_dt", "1/30")
        rendering_dt = simulator.get("rendering_dt", "1/30")
        if isinstance(physics_dt, str):
            physics_dt = float(Fraction(physics_dt))
        if isinstance(rendering_dt, str):
            rendering_dt = float(Fraction(rendering_dt))

        cuda_device = simulator.get("active_gpu", None)
        torch_cuda_device = None
        if cuda_device is not None:
            import torch

            torch_cuda_device = _resolve_torch_cuda_device(cuda_device)
            if torch_cuda_device is not None:
                torch.cuda.set_device(torch_cuda_device)
                self.logger.info(
                    f"PyTorch default CUDA device set to cuda:{torch_cuda_device}"
                    f" (configured active_gpu={cuda_device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})"
                )

        from isaacsim import SimulationApp

        launch_config = {
            "headless": simulator.get("headless", True),
            "anti_aliasing": simulator.get("anti_aliasing", 3),
            "multi_gpu": simulator.get("multi_gpu", True),
            "renderer": simulator.get("renderer", "RayTracedLighting"),
            # Isaac Sim 6 defaults to synchronous asset loading; preserve the
            # configured choice explicitly so stage readiness is deterministic.
            "sync_loads": _cast_setting_value(simulator.get("sync_loads", True), bool),
        }
        for key in ("active_gpu", "physics_gpu", "width", "height"):
            if key in simulator:
                launch_config[key] = int(simulator[key])
        for key in ("max_gpu_count", "denoiser", "samples_per_pixel_per_frame", "max_bounces",
                    "max_specular_transmission_bounces", "max_volume_bounces", "subdiv_refinement_level"):
            if key in simulator:
                launch_config[key] = simulator[key]
        self.logger.info(
            "SimulationApp launch GPU config: active_gpu=%s physics_gpu=%s"
            " (configured active_gpu=%s physics_gpu=%s, CUDA_VISIBLE_DEVICES=%s)",
            launch_config.get("active_gpu"),
            launch_config.get("physics_gpu"),
            simulator.get("active_gpu"),
            simulator.get("physics_gpu"),
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        )
        experience = _resolve_experience(simulator.get("experience", ""), headless=bool(launch_config["headless"]))
        if launch_config["headless"]:
            if "disable_viewport_updates" in simulator:
                launch_config["disable_viewport_updates"] = bool(simulator.get("disable_viewport_updates"))
            elif not simulator.get("experience", ""):
                launch_config["disable_viewport_updates"] = True
            self.simulation_app = SimulationApp(
                launch_config,
                experience=experience,
            )
        else:
            if experience:
                self.simulation_app = SimulationApp(launch_config, experience=experience)
            else:
                self.simulation_app = SimulationApp(launch_config)

        # Align Warp's default device with Isaac's configured active_gpu. In
        # pipe mode Ray may expose a single CUDA device to torch, but Isaac
        # still launches with the absolute active_gpu from the config.
        if cuda_device is not None:
            try:
                import warp as wp

                warp_cuda_device = int(cuda_device)
                wp.set_device(f"cuda:{warp_cuda_device}")
                self.logger.info(
                    f"Warp default CUDA device set to cuda:{warp_cuda_device}"
                    f" (configured active_gpu={cuda_device}, torch_default_cuda={torch_cuda_device})"
                )
            except Exception as exc:
                self.logger.warning(
                    f"Failed to set Warp default CUDA device to configured active_gpu cuda:{cuda_device}: {exc}"
                )
        _patch_curobo_quad_triangulation_launch_device(self.logger)

        applied_renderer_settings = {}
        optional_renderer_settings = (
            # RayTracedLighting runtime settings. These are separate from the
            # PathTracing-only SimulationApp launch options below.
            ("rt_new_denoiser", "/rtx/newDenoiser/enabled", bool),
            ("rt_shadows_enabled", "/rtx/shadows/enabled", bool),
            ("rt_shadow_sample_count", "/rtx/shadows/sampleCount", int),
            ("rt_direct_lighting_enabled", "/rtx/directLighting/enabled", bool),
            ("rt_sampled_direct_lighting", "/rtx/directLighting/sampledLighting/enabled", bool),
            ("rt_sampled_direct_lighting_spp", "/rtx/directLighting/sampledLighting/samplesPerPixel", int),
            ("rt_sampled_direct_lighting_max_ray_intensity", "/rtx/directLighting/sampledLighting/maxRayIntensity", float),
            ("rt_sampled_reflections_spp", "/rtx/reflections/sampledLighting/samplesPerPixel", int),
            ("rt_sampled_reflections_max_ray_intensity", "/rtx/reflections/sampledLighting/maxRayIntensity", float),
            ("rt_dome_lighting_enabled", "/rtx/directLighting/domeLight/enabled", bool),
            ("rt_dome_lighting_in_reflections", "/rtx/directLighting/domeLight/enabledInReflections", bool),
            ("rt_dome_lighting_sample_count", "/rtx/directLighting/domeLight/sampleCount", int),
            ("rt_reflections_enabled", "/rtx/reflections/enabled", bool),
            ("rt_reflections_max_roughness", "/rtx/reflections/maxRoughness", float),
            ("rt_reflections_max_bounces", "/rtx/reflections/maxReflectionBounces", int),
            ("rt_translucency_enabled", "/rtx/translucency/enabled", bool),
            ("rt_translucency_max_refraction_bounces", "/rtx/translucency/maxRefractionBounces", int),
            ("rt_translucency_reflect_at_all_bounces", "/rtx/translucency/reflectAtAllBounce", bool),
            ("rt_translucency_reflection_throughput_threshold", "/rtx/translucency/reflectionThroughputThreshold", float),
            ("rt_translucency_virtual_depth", "/rtx/translucency/virtualDepth", bool),
            ("rt_translucency_virtual_motion", "/rtx/translucency/virtualMotion", bool),
            ("rt_translucency_world_eps", "/rtx/translucency/worldEps", float),
            ("rt_translucency_sample_roughness", "/rtx/translucency/sampleRoughness", bool),
            ("rt_fractional_cutout_opacity", "/rtx/raytracing/fractionalCutoutOpacity", bool),
            ("rt_indirect_diffuse_enabled", "/rtx/indirectDiffuse/enabled", bool),
            ("rt_indirect_diffuse_spp", "/rtx/indirectDiffuse/fetchSampleCount", int),
            ("rt_indirect_diffuse_max_bounces", "/rtx/indirectDiffuse/maxBounces", int),
            ("rt_indirect_diffuse_intensity", "/rtx/indirectDiffuse/scalingFactor", float),
            ("rt_indirect_diffuse_max_ray_intensity", "/rtx/indirectDiffuse/maxRayIntensity", float),
            ("rt_ambient_occlusion_enabled", "/rtx/ambientOcclusion/enabled", bool),
            ("rt_ambient_occlusion_ray_length", "/rtx/ambientOcclusion/rayLength", float),
            ("rt_ambient_occlusion_min_samples", "/rtx/ambientOcclusion/minSamples", int),
            ("rt_ambient_occlusion_max_samples", "/rtx/ambientOcclusion/maxSamples", int),
            ("rt_ambient_light_intensity", "/rtx/sceneDb/ambientLightIntensity", float),
            ("rt_caustics_enabled", "/rtx/caustics/enabled", bool),
            ("rt_caustics_photon_count_multiplier", "/rtx/raytracing/caustics/photonCountMultiplier", int),
            ("rt_caustics_photon_max_bounces", "/rtx/raytracing/caustics/photonMaxBounces", int),
            ("rt_subsurface_enabled", "/rtx/raytracing/subsurface/enabled", bool),
            ("rt_subsurface_max_samples_per_frame", "/rtx/raytracing/subsurface/maxSamplePerFrame", int),
            ("rt_subsurface_firefly_filtering", "/rtx/raytracing/subsurface/fireflyFiltering/enabled", bool),
            ("rt_subsurface_denoiser", "/rtx/raytracing/subsurface/denoiser/enabled", bool),
            ("rt_eco_mode", "/rtx/ecoMode/enabled", bool),
            ("rt_eco_mode_max_frames_without_change", "/rtx/ecoMode/maxFramesWithoutChange", int),
            ("total_spp", "/rtx/pathtracing/totalSpp", int),
            ("adaptive_sampling", "/rtx/pathtracing/adaptiveSampling/enabled", bool),
            ("adaptive_sampling_target_error", "/rtx/pathtracing/adaptiveSampling/targetError", float),
            ("optix_denoiser", "/rtx/pathtracing/optixDenoiser/enabled", bool),
            ("optix_temporal_denoiser", "/rtx/pathtracing/optixDenoiser/temporalMode/enabled", bool),
            ("denoise_aovs", "/rtx/pathtracing/optixDenoiser/AOV", bool),
            ("firefly_filter", "/rtx/pathtracing/fireflyFilter/enabled", bool),
            ("firefly_max_intensity_glossy", "/rtx/pathtracing/fireflyFilter/maxIntensityPerSample", float),
            ("firefly_max_intensity_diffuse", "/rtx/pathtracing/fireflyFilter/maxIntensityPerSampleDiffuse", float),
            ("reset_pt_accum_on_time_change", "/rtx/resetPtAccumOnAnimTimeChange", bool),
            ("fractional_cutout_opacity", "/rtx/pathtracing/fractionalCutoutOpacity", bool),
        )
        for key, setting_path, cast in optional_renderer_settings:
            value = _apply_optional_setting(self.simulation_app, simulator, key, setting_path, cast)
            if value is not None:
                applied_renderer_settings[key] = value
        if applied_renderer_settings:
            self.logger.info(f"Applied renderer settings: {applied_renderer_settings}")

        self.logger.info(f"simulator params: physics dt={physics_dt}, rendering dt={rendering_dt}")
        from isaacsim.core.api import World

        world = World(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=simulator.get("stage_units_in_meters", 1.0),
        )

        # Import workflow extensions and create workflow
        from workflows import import_extensions

        if workflow_type == "SimBoxDualWorkFlow":
            _ensure_simbox_sensor_extension_ready(self.simulation_app)
        import_extensions(workflow_type)
        workflow_kwargs = {
            "scene_info": scene_info,
            "random_seed": get_random_seed(),
        }
        if workflow_type == "SimBoxDualWorkFlow":
            workflow_kwargs["planning_step_render"] = bool(simulator.get("planning_step_render", False))

        self.workflow = create_workflow(
            workflow_type,
            world,
            cfg_path,
            **workflow_kwargs,
        )
        self.workflow.simulation_app = self.simulation_app

        self.scene = None
        self.task_finish = False
        self.cur_index = 0
        self.record_init_time(time.time() - init_start_time)

        self.status_reporter.update_status(ComponentStatus.READY)

    @status_monitor()
    def _init_next_task(self):
        """
        Internal helper method to initialize and return the next task as a Scene object.

        Handles task repetition logic and advances the task index when all repetitions are complete.

        Returns:
            Scene: Initialized scene object for the next task.

        Raises:
            StopIteration: When all tasks have been exhausted.
        """
        if self.scene is not None and self.task_repeat_cnt > 0 and self.task_repeat_idx < self.task_repeat_cnt:
            self.logger.info(f"Task execute times {self.task_repeat_idx + 1}/{self.task_repeat_cnt}")
            self.workflow.init_task(self.cur_index - 1, self.need_preload)
            self.task_repeat_idx += 1
            scene = Scene(
                name=self.workflow.get_task_name(),
                wf=self.workflow,
                task_id=self.cur_index - 1,
                task_exec_num=self.task_repeat_idx,
                simulation_app=self.simulation_app,
            )
            return scene
        if self.cur_index >= len(self.workflow.task_cfgs):
            self.logger.info("No more tasks to load, stopping iteration.")
            raise StopIteration
        self.logger.info(f"Loading task {self.cur_index + 1}/{len(self.workflow.task_cfgs)}")
        self.workflow.init_task(self.cur_index, self.need_preload)
        self.task_repeat_idx = 1
        scene = Scene(
            name=self.workflow.get_task_name(),
            wf=self.workflow,
            task_id=self.cur_index,
            task_exec_num=self.task_repeat_idx,
            simulation_app=self.simulation_app,
        )
        self.cur_index += 1
        return scene

    def load_asset(self) -> Scene:
        """
        Load and initialize the next scene from workflow.

        Supports two modes:
        - Standalone: Iterates through workflow tasks directly
        - Pipeline: Synchronizes with incoming packages and applies plan info to scene

        Returns:
            Scene: The loaded and initialized Scene object.

        Raises:
            StopIteration: When no more scenes are available.
        """
        try:
            # Standalone mode: load tasks directly from workflow
            if self.pack_iter is None:
                self.scene = self._init_next_task()
            # Pipeline mode: load tasks from package iterator
            else:
                package = next(self.pack_iter)
                self.cur_index = package.task_id

                # Initialize scene if this is the first package or a new task
                if self.scene is None:
                    self.scene = self._init_next_task()
                elif self.cur_index > self.scene.task_id:
                    self.scene = self._init_next_task()

                # Apply plan information from package to scene
                package.data = self.scene.wf.dedump_plan_info(package.data)
                self.scene.add_plan_info(package.data)

            return self.scene
        except StopIteration:
            raise StopIteration
        except Exception as e:
            raise e
