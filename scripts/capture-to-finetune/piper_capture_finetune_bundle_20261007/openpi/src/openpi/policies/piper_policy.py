"""OpenPI transforms for the 6-DoF AgileX Piper X dataset.

The public policy interface deliberately uses the same keys as the training
repack transform in ``training.config.LeRobotPiperDataConfig``. Keep these
keys in sync with the real-robot inference harness.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


PIPER_ACTION_DIM = 7


def _parse_image(image: object) -> np.ndarray:
    """Convert either LeRobot CHW float images or runtime HWC uint8 images to HWC uint8."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected one RGB image with three dimensions, got shape {image.shape}.")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {image.shape}.")
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Map Piper observations to pi0.5's fixed image/state interface.

    Required runtime inputs:
      - ``observation/image``: D435i base RGB image
      - ``observation/wrist_image``: DaBai DC1 wrist RGB image
      - ``observation/state``: float32[7], [q1..q6 radians, gripper closedness]
      - ``prompt``: task text
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape[-1] != PIPER_ACTION_DIM:
            raise ValueError(
                f"Expected Piper state with {PIPER_ACTION_DIM} values "
                f"[q1..q6, gripper_closedness], got {state.shape}."
            )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """Return 6 joint targets and one absolute gripper command to the Piper harness."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :PIPER_ACTION_DIM]}
