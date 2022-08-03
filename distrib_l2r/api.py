from dataclasses import dataclass
from typing import Any
from typing import Optional


@dataclass
class BaseMsg:
    """A base message"""

    data: Optional[Any] = None


class InitMsg(BaseMsg):
    """Message a worker sends on startup"""

    pass


class BufferMsg(BaseMsg):
    """A replay buffer message sent from a worker"""

    pass


class EvalResultsMsg(BaseMsg):
    """An evaluation results message sent from a worker"""

    pass


class PolicyMsg(BaseMsg):
    """An RL policy message sent from a learner"""

    pass
