"""Semantic execution-variant signature contract."""

from __future__ import annotations

import copy

from agent.tools.signatures import variant_signature


def _compiled_document():
    return {
        "tasks": [
            {
                "arena_file": "/tmp/run_a/scene/simbox_arena.yaml",
                "metadata": {
                    "agent_plan": {
                        "selected_task_id": "task_a",
                        "robot_profile_id": "profile_a",
                        "robot_profile_hash": "a" * 64,
                        "placement_family": "floor_standing",
                        "scene_revision": "scene_a",
                        "workspace_candidate_id": "candidate_000",
                        "subtasks": [
                            {
                                "subtask_id": "transfer",
                                "arm": "left",
                                "relation": "inside",
                            }
                        ],
                    },
                    "robot_position_plan": {
                        "initial": {
                            "candidate_id": "candidate_000",
                            "world_xy": [1.0, 2.0],
                            "yaw_deg": 90.0,
                        }
                    },
                },
                "regions": [
                    {
                        "object": "cup",
                        "center": [1.0, 2.0],
                        "random_config": {"yaw_rotation": [0.0, 0.0]},
                    }
                ],
                "skills": [{"robot": [{"left": [{"name": "Pick"}]}]}],
            }
        ]
    }


def test_signature_ignores_run_paths_and_candidate_labels():
    first = _compiled_document()
    second = copy.deepcopy(first)
    second["tasks"][0]["arena_file"] = "/tmp/run_b/scene/simbox_arena.yaml"
    metadata = second["tasks"][0]["metadata"]
    metadata["agent_plan"]["workspace_candidate_id"] = "candidate_999"
    metadata["robot_position_plan"]["initial"]["candidate_id"] = "candidate_999"
    second["tasks"][0]["regions"][0]["candidate_id"] = "candidate_999"
    first["tasks"][0]["regions"][0]["candidate_id"] = "candidate_000"

    assert variant_signature(first) == variant_signature(second)


def test_signature_changes_with_pose_layout_arm_or_skill_semantics():
    original = _compiled_document()
    original_signature = variant_signature(original)
    mutations = []
    changed_pose = copy.deepcopy(original)
    changed_pose["tasks"][0]["metadata"]["robot_position_plan"]["initial"][
        "world_xy"
    ][0] += 0.1
    mutations.append(changed_pose)
    changed_layout = copy.deepcopy(original)
    changed_layout["tasks"][0]["regions"][0]["center"][0] += 0.1
    mutations.append(changed_layout)
    changed_arm = copy.deepcopy(original)
    changed_arm["tasks"][0]["metadata"]["agent_plan"]["subtasks"][0][
        "arm"
    ] = "right"
    mutations.append(changed_arm)
    changed_skill = copy.deepcopy(original)
    changed_skill["tasks"][0]["skills"][0]["robot"][0]["left"][0][
        "pre_grasp_offset"
    ] = 0.12
    mutations.append(changed_skill)

    assert all(variant_signature(value) != original_signature for value in mutations)
