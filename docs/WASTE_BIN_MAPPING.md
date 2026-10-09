# Waste to bin mapping

The trained MobileNetV2 model has four outputs, defined in data/labels.json:

| Model label | Category | Existing routing |
| --- | ---: | --- |
| yellow | 0 | Yellow |
| red | 1 | Red |
| blue | 2 | Blue |
| white | 3 | White |

The model does not predict metal or glass. Those names are routing aliases for category 2 in the existing decision mapper. The serial response preserves the model's genuine label, which is blue for the current four-class model.

## Arduino JSONL protocol

The Arduino sends one UTF-8 JSON object followed by a newline:

    {"command":"capture","object_id":1}

object_id must be an integer or non-empty string. Python preserves its JSON value in the corresponding response. The camera preview does not classify by itself; each valid capture request triggers exactly one classification using the latest available camera frame.

Success:

    {"status":"success","object_id":1,"category":2,"label":"blue","confidence":0.95}

Failure after a valid request ID was received:

    {"status":"failed","object_id":1,"category":0,"label":"unknown","confidence":0.0}

Every response is one compact UTF-8 JSON line terminated by newline. A valid request receives one response even if no camera frame is available or inference fails. Invalid JSON, missing or invalid IDs, and unsupported commands are logged in the dashboard and receive no response because there is no valid capture request to correlate.

The listener reads complete newline-framed messages in the background while connected. Classification and serial writing also run outside Tk's event thread. Disconnecting or closing the dashboard stops the workers and releases the serial port. No firmware is included, so physical communication remains to be verified on the intended Arduino.

## Hardware capabilities not present in this repository

No fill-level sensor driver, sensor measurements, or Arduino telemetry fields are present, so the dashboard cannot report bin fill percentages. No bin placement angles are documented or configured. Implement these only after receiving actual sensor inputs and approved angle values from the hardware design; this project does not invent either measurement.
