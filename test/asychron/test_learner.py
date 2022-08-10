import threading
import time
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from tianshou.data import Batch
from tianshou.data import ReplayBuffer

from distrib_l2r.asynchron.learner import AsyncLearningNode


class TestAsyncLearner(unittest.TestCase):
    """Tests associated with the asnychronous learner"""

    @classmethod
    def setUpClass(cls):
        """Create and start a server"""
        cls.mock_policy = MagicMock()

        with patch("pickle.dumps"):
            cls.learner = AsyncLearningNode(
                server_address=("127.0.0.1", 7777), policy=cls.mock_policy
            )
            cls._learner_thread = threading.Thread(target=cls.learner.learn)
            cls._learner_thread.daemon = True
            cls._learner_thread.start()

    @patch("pickle.dumps")
    def test_server_updates_policy_when_batch_received(
        self, mock_pickle: MagicMock
    ) -> None:
        """Validate learner updates policy when a batch is received"""

        replay_buf = ReplayBuffer(size=5)
        replay_buf.add(Batch(obs={"numbers": [1, 1, 1]}, act=0, rew=0, done=0))

        # Add a buffer to the queue
        self.learner.buffer_queue.put(replay_buf)

        # Validate policy is updated
        time.sleep(0.1)
        self.learner.policy.update.assert_called()


if __name__ == "__main__":
    unittest.main()
