"""Patched LeRobot policy server for Rosetta.

Monkey-patches ``DiffusionPolicy.predict_action_chunk`` so that it handles
observation queue population and image stacking internally. The upstream
version assumes these are done externally by ``select_action``, but the
``PolicyServer._get_action_chunk`` path calls ``predict_action_chunk`` directly.
"""

import torch
from torch import Tensor

from lerobot.async_inference.policy_server import serve  # noqa: F401
from lerobot.utils.constants import OBS_IMAGES, ACTION
from lerobot.policies.utils import populate_queues
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

@torch.no_grad()
def _patched_predict_action_chunk(
    self: DiffusionPolicy, batch: dict[str, Tensor], noise: Tensor | None = None
) -> Tensor:
    """Predict a chunk of actions given environment observations."""
    if ACTION in batch:
            batch.pop(ACTION)
    # Stack n latest observations from the queue
    if self.config.image_features:
        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
    # NOTE: It's important that this happens after stacking the images into a single key.
    self._queues = populate_queues(self._queues, batch)
    batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
    actions = self.diffusion.generate_actions(batch, noise=noise)
    return actions


# Apply the patch.
DiffusionPolicy.predict_action_chunk = _patched_predict_action_chunk


if __name__ == "__main__":
    serve()
