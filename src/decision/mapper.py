"""Map model labels and routing aliases to hardware-neutral bin commands."""
from __future__ import annotations
from enum import IntEnum

class WasteCommand(IntEnum):
    YELLOW = 0
    RED = 1
    BLUE = 2
    WHITE = 3

LABEL_TO_COMMAND = {"yellow": WasteCommand.YELLOW, "red": WasteCommand.RED,
                    "blue": WasteCommand.BLUE, "white": WasteCommand.WHITE}
# Routing aliases only: this model predicts blue, not metal or glass.
ROUTING_ALIASES = {"metal": WasteCommand.BLUE, "glass": WasteCommand.BLUE}

def map_prediction(prediction: str | int | WasteCommand) -> int:
    if isinstance(prediction, WasteCommand):
        return int(prediction)
    if isinstance(prediction, bool):
        raise ValueError("A boolean is not a valid biomedical-waste prediction.")
    if isinstance(prediction, int):
        if prediction in {int(value) for value in WasteCommand}:
            return prediction
        raise ValueError("Class index must be one of 0 (yellow), 1 (red), 2 (blue), or 3 (white).")
    if isinstance(prediction, str):
        label = prediction.strip().lower().removesuffix(" waste")
        command = LABEL_TO_COMMAND.get(label, ROUTING_ALIASES.get(label))
        if command is not None:
            return int(command)
        raise ValueError("Prediction must be yellow, red, blue, white, metal, or glass.")
    raise TypeError("Prediction must be a label string or class index.")
