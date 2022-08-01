class InitMsg:
    """Message a worker sends on startup"""

    pass


class BufferMsg:
    """A replay buffer message sent from a worker"""

    pass


class EvalResultsMsg:
    """An evaluation results message sent from a worker"""

    pass


class PolicyMsg:
    """An RL policy message sent from a learner"""

    pass
