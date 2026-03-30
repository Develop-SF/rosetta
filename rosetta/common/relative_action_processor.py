# Copyright 2025 Isaac Blankenau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Relative-action processor steps for Rosetta.

These steps convert between absolute and relative action representations.
"Relative" means every action is expressed as an offset from the **first**
robot command (the initial state at the start of an episode).

Two steps are provided:

1. ``AbsoluteToRelativeActionStep`` — used during **data porting**
   (``port_bags``) to convert the recorded absolute joint commands into
   relative offsets before they are written to the LeRobot dataset.

2. ``RelativeToAbsoluteActionStep`` — used during **inference** as a
   post-processor step to convert the policy's relative action output
   back to absolute joint commands that can be sent to the robot.

Usage (data porting)::

    from rosetta.common.relative_action_processor import AbsoluteToRelativeActionStep
    from lerobot.processor.pipeline import RobotProcessorPipeline
    from lerobot.processor.converters import robot_action_to_transition, transition_to_robot_action

    pipeline = RobotProcessorPipeline(
        steps=[AbsoluteToRelativeActionStep(action_keys=["arm_joint_positions"])],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )
    pipeline.save_pretrained("/path/to/save", config_filename="robot_action_processor.json")

    # Then pass to port_bags:
    #   port_bags ... --action-processor /path/to/save

Usage (inference post-processor)::

    The ``RelativeToAbsoluteActionStep`` should be inserted into the
    policy's **post-processor** pipeline (after unnormalization) so that
    the relative action tensor coming out of the policy is converted to
    absolute commands before being sent to the robot.

    It reads the current observation state from the transition to obtain
    the "initial" reference. On the first call of an episode it captures
    the initial state; subsequent calls accumulate relative deltas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import (
    ProcessorStepRegistry,
    RobotActionProcessorStep,
)
from lerobot.types import RobotAction, TransitionKey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_numeric_keys(action: RobotAction) -> list[str]:
    """Return action keys whose values are numeric (float / ndarray)."""
    return [
        k
        for k, v in action.items()
        if isinstance(v, (int, float, np.integer, np.floating, np.ndarray))
    ]


# ---------------------------------------------------------------------------
# Data-porting step: absolute → relative  (operates on RobotAction dicts)
# ---------------------------------------------------------------------------


@ProcessorStepRegistry.register("absolute_to_relative_action")
@dataclass
class AbsoluteToRelativeActionStep(RobotActionProcessorStep):
    """Convert absolute action values to offsets relative to the first action.

    During dataset creation (``port_bags``), each episode's first action
    is captured as the *reference*.  Every subsequent action in that
    episode is stored as ``action - reference``, so the first action
    becomes all-zeros.

    Call :meth:`reset` between episodes to clear the captured reference
    so the next episode starts fresh.

    Attributes:
        action_keys: List of action-dict keys to convert.  Keys not in
            this list are passed through unchanged. If empty, **all**
            keys whose values are numeric (float / np.ndarray) are
            converted.
    """

    action_keys: list[str] = field(default_factory=list)
    _reference: dict[str, np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )

    # -- RobotActionProcessorStep interface --

    def action(self, action: RobotAction) -> RobotAction:
        keys = self.action_keys or _infer_numeric_keys(action)

        # Capture reference on first call (or after reset)
        if not self._reference:
            for k in keys:
                self._reference[k] = np.asarray(action[k], dtype=np.float64)

        new_action: RobotAction = dict(action)
        for k in keys:
            if k in new_action:
                abs_val = np.asarray(new_action[k], dtype=np.float64)
                new_action[k] = abs_val - self._reference[k]
        return new_action

    # -- Lifecycle --

    def reset(self) -> None:
        """Clear the captured reference (start of a new episode)."""
        self._reference.clear()

    # -- Serialisation --

    def get_config(self) -> dict[str, Any]:
        return {"action_keys": self.action_keys}

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features  # shape unchanged


# ---------------------------------------------------------------------------
# Inference step: relative → absolute  (operates on RobotAction dicts)
# ---------------------------------------------------------------------------


@ProcessorStepRegistry.register("relative_to_absolute_action")
@dataclass
class RelativeToAbsoluteActionStep(RobotActionProcessorStep):
    """Convert relative (delta-from-first) actions back to absolute values.

    At **inference** time, the policy outputs relative actions.  This step
    adds them back to a captured *initial state* to produce absolute
    commands for the robot.

    The initial state is captured from the **observation** on the very
    first call after construction or :meth:`reset`.  It expects the
    observation dict to contain keys that mirror the action keys (e.g. if
    the action key is ``"arm_joint_positions"`` the observation must
    contain ``"arm_joint_positions"`` with the current joint state).

    Attributes:
        action_keys: Keys to convert. Same semantics as
            :class:`AbsoluteToRelativeActionStep`.
        obs_key_map: Optional mapping from action key → observation key
            when names differ.  E.g. ``{"arm_joint_positions":
            "arm_joint_positions"}`` (identity map is the default).
    """

    action_keys: list[str] = field(default_factory=list)
    obs_key_map: dict[str, str] = field(default_factory=dict)
    _initial_state: dict[str, np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )

    # -- RobotActionProcessorStep interface --

    def action(self, action: RobotAction) -> RobotAction:
        keys = self.action_keys or _infer_numeric_keys(action)

        # Capture initial state from observation on first call
        if not self._initial_state:
            obs = self.transition.get(TransitionKey.OBSERVATION)
            if obs is None or not isinstance(obs, dict):
                raise ValueError(
                    "RelativeToAbsoluteActionStep requires an "
                    "observation in the transition to capture "
                    "the initial robot state. Make sure the "
                    "pipeline's to_transition converter "
                    "includes the observation."
                )
            missing = []
            for k in keys:
                obs_k = self.obs_key_map.get(k, k)
                if obs_k in obs:
                    self._initial_state[k] = np.asarray(obs[obs_k], dtype=np.float64)
                else:
                    missing.append(obs_k)
            if missing:
                raise ValueError(
                    f"Observation is missing keys needed "
                    f"by RelativeToAbsoluteActionStep: "
                    f"{missing}. Available keys: "
                    f"{list(obs.keys())}"
                )

        new_action: RobotAction = dict(action)
        for k in keys:
            if k in new_action and k in self._initial_state:
                rel_val = np.asarray(new_action[k], dtype=np.float64)
                new_action[k] = rel_val + self._initial_state[k]
        return new_action

    # -- Lifecycle --

    def reset(self) -> None:
        """Clear the initial state (start of a new episode)."""
        self._initial_state.clear()

    # -- Serialisation --

    def get_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"action_keys": self.action_keys}
        if self.obs_key_map:
            cfg["obs_key_map"] = self.obs_key_map
        return cfg

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features  # shape unchanged
