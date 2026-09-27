# Waste to bin mapping

The final project's MobileNetV2 model has four outputs. Model label order comes from `data/labels.json`; the hardware bin number is the same class index.

| Waste predicted by model | Model label | Hardware bin |
| --- | --- | ---: |
| Yellow category | `yellow` | 0 |
| Red category | `red` | 1 |
| Blue category | `blue` | 2 |
| White category | `white` | 3 |

Glass equipment packaging and metal equipment packaging both route to the blue bin (2). `glass` and `metal` are routing aliases only; neither is a model output or separate bin. The trained classifier returns the actual model label (for these items, `blue`) and its confidence.

## JSONL serial protocol

Each request is one UTF-8 JSON object terminated by a newline:

```json
{"object_id":"unique-object-id","action":"classify"}
```

The camera preview does not run inference. A complete valid request triggers one inference from the latest available frame and one JSON response line. The optional action defaults to `classify`. Invalid JSON, unsupported actions, oversized requests, and unavailable camera frames receive a bounded error response. Successful responses include the same `object_id`, the model label, class index, confidence, inference time, and bin number. The response is newline terminated; logs remain outside the serial stream.

Do not reuse an `object_id`. The dashboard remembers recent IDs and returns an error for duplicates without classifying again. Legacy numeric command methods remain available for callers, but an ACK timeout after a write is ambiguous and the controller will not resend that command.

No firmware is included in this repository. Hardware framing, parsing, and actuator behavior require validation on the intended board before deployment.
