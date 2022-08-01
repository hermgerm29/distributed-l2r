import pickle
import socketserver

from distrib_l2r.api import BufferMsg
from distrib_l2r.api import InitMsg
from distrib_l2r.api import EvalResultsMsg

# from distrib_l2r.api import PolicyMsg
from distrib_l2r.utils import recv_bytes_with_prefix_size

# from distrib_l2r.utils import send_data


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    """Request handler thread created for every request"""

    def handle(self) -> None:
        """ReplayBuffers are not thread safe - pass data via thread-safe queues"""
        data = pickle.loads(recv_bytes_with_prefix_size(self.request))

        # Received a replay buffer from a worker
        # Add this buff
        if isinstance(data, BufferMsg):
            pass

        # Received an init message from a worker
        # Immediately reply with the most up-to-date policy
        elif isinstance(data, InitMsg):
            pass

        # Receieved evaluation results from a worker
        elif isinstance(data, EvalResultsMsg):
            pass

        # unexpected
        else:
            print(f"Received unexpected data: {type(data)}", flush=True)
            return
