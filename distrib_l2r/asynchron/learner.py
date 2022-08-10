import logging
import pickle
import queue
import socketserver
from typing import Callable
from typing import Optional
from typing import Tuple

from tianshou.data import ReplayBuffer
from tianshou.policy import BasePolicy

from distrib_l2r.api import BufferMsg
from distrib_l2r.api import InitMsg
from distrib_l2r.api import EvalResultsMsg
from distrib_l2r.utils import receive_data
from distrib_l2r.utils import send_data


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    """Request handler thread created for every request"""

    def handle(self) -> None:
        """ReplayBuffers are not thread safe - pass data via thread-safe queues"""
        msg = receive_data(self.request)

        # Received a replay buffer from a worker
        # Add this buff
        if isinstance(msg, BufferMsg):
            logging.info("Received replay buffer")
            self.server.buffer_queue.put(msg.data)

        # Received an init message from a worker
        # Immediately reply with the most up-to-date policy
        elif isinstance(msg, InitMsg):
            logging.info("Receieved init message")

        # Receieved evaluation results from a worker
        elif isinstance(msg, EvalResultsMsg):
            logging.info("Received evalution results message")

        # unexpected
        else:
            logging.warning(f"Received unexpected data: {type(msg)}")
            return

        # Reply to the request with an up-to-date policy
        send_data(data=self.server.get_policy(), sock=self.request)


class AsyncLearningNode(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """A multi-threaded, offline, off-policy reinforcement learning server

    Args:
        policy: an intial Tianshou policy
        update_steps: the number of gradient updates for each buffer received
        batch_size: the batch size for gradient updates
        epochs: the number of buffers to receive before concluding learning
        server_address: the address the server runs on
        save_func: a function for saving which is called while learning with
          parameters `epoch` and `policy`
        save_freq: the frequency, in epochs, to save
    """

    def __init__(
        self,
        policy: BasePolicy,
        update_steps: int = 64,
        batch_size: int = 128,
        epochs: int = 500,
        buffer_size: int = 1_000_000,
        server_address: Tuple[str, int] = ("0.0.0.0", 4444),
        save_func: Optional[Callable] = None,
        save_freq: Optional[int] = None,
    ) -> None:

        super().__init__(server_address, ThreadedTCPRequestHandler)

        self.update_steps = update_steps
        self.batch_size = batch_size
        self.epochs = epochs

        # Create a replay buffer
        self.replay_buffer = ReplayBuffer(size=buffer_size)

        # Inital policy to use
        self.policy = policy

        # The bytes of the policy to reply to requests with
        self.policy_bytes = pickle.dumps(self.policy.state_dict())

        # A thread-safe policy queue to avoid blocking while learning. This marginally
        # increases off-policy error in order to improve throughput.
        self.policy_queue = queue.Queue(maxsize=1)

        # A queue of buffers that have been received but not yet added to the learner's
        # main replay buffer
        self.buffer_queue = queue.Queue()

        # Save function, called optionally
        self.save_func = save_func
        self.save_freq = save_freq

    def get_policy(self) -> bytes:
        """Get the most up-to-date version of the policy without blocking"""
        if not self.policy_queue.empty():
            try:
                self.policy_bytes = self.policy_bytes_queue.get_nowait()
            except queue.Empty:
                # non-blocking
                pass

        return self.policy_bytes

    def update_policy_bytes(self) -> None:
        """Update policy bytes"""
        if not self.policy_queue.empty():
            try:
                # empty queue for safe put()
                _ = self.policy_queue.get_nowait()
            except queue.Empty:
                pass

        self.policy_queue.put(pickle.dumps(self.policy.state_dict()))

    def learn(self) -> None:
        """The thread where thread-safe gradient updates occur"""
        for epoch in range(self.epochs):

            # block until new data is received
            batch = self.buffer_queue.get()

            # Add new data to the primary replay buffer
            self.replay_buffer.update(batch)

            # Learning steps for the policy
            for _ in range(self.update_steps):
                _ = self.policy.update(
                    sample_size=self.batch_size, buffer=self.replay_buffer
                )

            # Update policy bytes
            self.update_policy_bytes()

            # Optionally save
            if self.save_func and epoch % self.save_every == 0:
                self.save_fn(epoch=epoch, policy=self.get_policy())
