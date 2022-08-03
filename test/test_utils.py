import random
import socketserver
import threading
import unittest

from deepdiff import DeepDiff
from tianshou.data import Batch
from tianshou.data import ReplayBuffer

from distrib_l2r.utils import send_data
from distrib_l2r.utils import receive_data


class RequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        """Simply respond back with the data that was received"""
        data = receive_data(self.request)
        send_data(sock=self.request, data=data, reply=False)


class TestDataUtilities(unittest.TestCase):
    """Tests associated with sending and receiving data"""

    @classmethod
    def setUpClass(cls):
        """Create and start a server"""
        server = socketserver.TCPServer(
            server_address=("127.0.0.1", 9999), RequestHandlerClass=RequestHandler
        )
        cls._server_thread = threading.Thread(target=server.serve_forever)
        cls._server_thread.daemon = True
        cls._server_thread.start()

    def test_send_and_receive_buffer(self) -> None:
        """Send a buffer to the server and assert it comes back the same"""
        buf_size = 10
        replay_buf = ReplayBuffer(size=buf_size)

        for _ in range(buf_size):
            replay_buf.add(
                Batch(
                    obs={
                        "random_numbers": random.sample(range(1, 30), 5),
                        "more_numbers": random.sample(range(5, 20), 10),
                    },
                    act=0,
                    rew=0,
                    done=0,
                )
            )

        response = send_data(addr=("127.0.0.1", 9999), data=replay_buf, reply=True)
        self.assertFalse(DeepDiff(response, replay_buf, ignore_order=True))
