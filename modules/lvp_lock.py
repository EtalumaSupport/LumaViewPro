

import socket

class LvpLock:

    def __init__(self, lock_port: int):
        self._lock_port = lock_port
        self._lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    

    def lock(self) -> bool:
        try:
            self._lock_socket.bind(("127.0.0.1", self._lock_port))
            return True
        except (socket.error, OSError) as e:
            return False
