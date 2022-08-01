import pickle
import socket
import struct
from select import epoll, EPOLLIN
from typing import Any


INT_SIZE = 4


def send_data(ip: str, port: int, data: Any, reply: bool = False) -> Any:
    """Creates a TCP socket and sends data to the specified address

    :param ip: ip address
    :param port: port
    :param data: any data that is either binary or able to be pickled
    :param reply: listen on the same socket for a reply. if True, this
      function returns unpickled data
    """
    if not isinstance(data, bytes):
        data = pickle.dumps(data)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(address=(ip, port))
        send_bytes_with_prefix_size(msg=data, sock=sock)

        if reply:
            response = None
            polly = epoll()
            polly.register(sock.fileno(), EPOLLIN)

            while not response:
                events = polly.poll(1)
                for fileno, event in events:
                    if fileno == sock.fileno():
                        return pickle.loads(recv_bytes_with_prefix_size(socket=sock))


def send_bytes_with_prefix_size(msg: bytes, sock: socket.Socket) -> None:
    """Utility to send bytes across a socket"""
    if not isinstance(msg, bytes):
        raise TypeError

    # Prefix message with length and send
    sock.sendall(bytes=struct.pack(">I", len(msg)) + msg)


def recv_bytes_with_prefix_size(sock: socket.Socket) -> bytes:
    """Utility to receive bytes that are prefixed with the size"""
    raw_size = sock.recv(INT_SIZE)
    msg_size = struct.unpack(">I", raw_size)[0]
    raw_data = b""

    # Continue receiving data until expected size is reached
    while len(raw_data) < msg_size:
        chunk = sock.recv(msg_size - len(raw_data))
        if not chunk:
            return None
        raw_data += chunk

    return raw_data
