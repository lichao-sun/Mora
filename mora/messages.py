from typing import Any

from pydantic import BaseModel
from pydantic.fields import Field
from asyncio import Queue, QueueEmpty, wait_for
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union
import importlib
MESSAGE_ROUTE_TO_ALL = "<all>"
def any_to_str(val: Any) -> str:
    """Return the class name or the class name of the object, or 'val' if it's a string type."""
    if isinstance(val, str):
        return val
    elif not callable(val):
        return get_class_name(type(val))
    else:
        return get_class_name(val)
def import_class(class_name: str, module_name: str) -> type:
    module = importlib.import_module(module_name)
    a_class = getattr(module, class_name)
    return a_class
def get_class_name(cls) -> str:
    """Return class name"""
    return f"{cls.__module__}.{cls.__name__}"
def any_to_str_set(val) -> set:
    """Convert any type to string set."""
    res = set()

    # Check if the value is iterable, but not a string (since strings are technically iterable)
    if isinstance(val, (dict, list, set, tuple)):
        # Special handling for dictionaries to iterate over values
        if isinstance(val, dict):
            val = val.values()

        for i in val:
            res.add(any_to_str(i))
    else:
        res.add(any_to_str(val))

    return res
class Message(BaseModel):
    """

    """
    id: str = Field("", validate_default=True)
    role: str = "user"  # system / user / assistant
    sender: str = Field("", validate_default=True)
    cause_by: str = Field("", validate_default=True)
    content: str= Field("", validate_default=True)
    image_content: Any = Field(None, validate_default=True)
    send_to: set[str] = Field(default={MESSAGE_ROUTE_TO_ALL}, validate_default=True)
    sent_from: str = Field(default="", validate_default=True)
    @field_validator("cause_by", mode="before")
    @classmethod
    def check_cause_by(cls, cause_by: Any) -> str:
        return any_to_str(cause_by if cause_by else import_class("UserRequirement", "mora.actions.add_requirement"))

    @field_validator("sent_from", mode="before")
    @classmethod
    def check_sent_from(cls, sent_from: Any) -> str:
        return any_to_str(sent_from if sent_from else "")

    @field_validator("send_to", mode="before")
    @classmethod
    def check_send_to(cls, send_to: Any) -> set:
        return any_to_str_set(send_to if send_to else {MESSAGE_ROUTE_TO_ALL})

class MessageQueue(BaseModel):
    """Message queue which supports asynchronous updates."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _queue: Queue = PrivateAttr(default_factory=Queue)

    def pop(self) -> Message | None:
        """Pop one message from the queue."""
        try:
            item = self._queue.get_nowait()
            if item:
                self._queue.task_done()
            return item
        except QueueEmpty:
            return None

    def pop_all(self) -> List[Message]:
        """Pop all messages from the queue."""
        ret = []
        while True:
            msg = self.pop()
            if not msg:
                break
            ret.append(msg)
        return ret

    def push(self, msg: Message):
        """Push a message into the queue."""
        self._queue.put_nowait(msg)

    def empty(self):
        """Return true if the queue is empty."""
        return self._queue.empty()

