import base64
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, validator


# enum
class MessageType(str, Enum):
    ACTIVATIONS = "activations"
    ACTIVATIONS_AND_LABELS = "activations_and_labels"
    GRADS = "grads"
    LABELS = "labels"
    LOGITS = "logits"
    REQUEST_BATCH = "request_batch"
    BATCH = "batch"
    # Frontend sends its trained TF.js client weights to the server for
    # inspection. Payload: `data` carries the model topology JSON +
    # weight specs + training metadata; `raw.weights` carries the
    # concatenated Float32 weight bytes.
    SAVE_CLIENT_MODEL = "save_client_model"
    # Tell the server to re-initialise its half (random weights + fresh
    # optimizer) so joint training starts symmetric with a freshly-built
    # client. Without this, a previously-trained server clobbers its
    # learned features as soon as the new random client feeds it noise.
    RESET_SERVER = "reset_server"


class WSMessage(BaseModel):
    type: MessageType
    data: Dict[str, Any] = {}
    raw: Dict[str, bytes] = {}

    class Config:
        json_encoders = {bytes: lambda x: base64.b64encode(x).decode("utf-8")}

    @validator("raw", pre=True)
    def decode_base64(cls, value: Dict[str, str]) -> Dict[str, bytes]:
        return {
            k: (base64.b64decode(v) if isinstance(v, str) else v)
            for k, v in value.items()
        }
