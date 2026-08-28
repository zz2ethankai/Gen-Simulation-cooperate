"""Exact one-planner attachment state for the Physics Schema boundary."""

from __future__ import annotations

from typing import Any

from .domain_types import AttachmentResult, AttachmentSpec, AttachmentState


class AttachmentRuntimeError(RuntimeError):
    """An attachment operation violated the native CuRobo contract."""


class AttachmentRuntime:
    """Drive one native attachment manager and one typed scene owner."""

    def __init__(self, manager: Any, *, scene: Any) -> None:
        if manager is None:
            raise ValueError("AttachmentRuntime requires one native attachment manager")
        self.manager = manager
        self.scene = scene
        self._spec: AttachmentSpec | None = None
        self._state = AttachmentState.DETACHED

    @property
    def state(self) -> AttachmentState:
        return self._state

    @property
    def attached(self) -> bool:
        return self._spec is not None

    @property
    def current(self) -> AttachmentSpec | None:
        return self._spec

    @property
    def attached_names(self) -> tuple[str, ...]:
        return () if self._spec is None else (self._spec.name,)

    @property
    def attached_obstacle_names(self) -> tuple[str, ...]:
        return () if self._spec is None else self._spec.disable_obstacle_names

    def attach(self, spec: AttachmentSpec) -> AttachmentResult:
        if not isinstance(spec, AttachmentSpec):
            raise TypeError("attach requires AttachmentSpec")
        self.manager.attach(
            spec.state,
            spec.meshes,
            link_name=spec.link_name,
            num_spheres=spec.num_spheres,
            surface_radius=spec.surface_radius,
            sphere_fit_type=spec.sphere_fit_type,
            world_objects_pose_offset=spec.pose_offset,
            disable_obstacle_names=list(spec.disable_obstacle_names),
        )
        for name in spec.disable_obstacle_names:
            self.scene.set_obstacle_enabled(name, False)
        self._spec = spec
        self._state = AttachmentState.ATTACHED
        return AttachmentResult(spec=spec, state=AttachmentState.ATTACHED)

    def detach(self, name: str | None = None) -> AttachmentResult:
        if self._spec is None:
            return AttachmentResult(state=AttachmentState.DETACHED)
        if name is not None and name != self._spec.name:
            raise AttachmentRuntimeError(f"cannot detach unknown attachment: {name}")
        spec = self._spec
        self.manager.detach()
        for obstacle in spec.disable_obstacle_names:
            self.scene.set_obstacle_enabled(obstacle, True)
        self._spec = None
        self._state = AttachmentState.DETACHED
        return AttachmentResult(spec=spec, state=AttachmentState.DETACHED)


__all__ = [
    "AttachmentRuntime",
    "AttachmentRuntimeError",
]
