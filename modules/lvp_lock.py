# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import socket

class LvpLock:

    def __init__(self, lock_port: int = 0):
        """Create an instance lock.

        Args:
            lock_port: TCP port to bind for the lock. Pass 0 to let the OS
                       assign an ephemeral port (more secure — avoids predictable
                       port that could be targeted for local DoS).
        """
        self._lock_port = lock_port
        self._lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._lock_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


    def lock(self) -> bool:
        try:
            self._lock_socket.bind(("127.0.0.1", self._lock_port))
            return True
        except (socket.error, OSError) as e:
            return False

    @property
    def port(self) -> int:
        """Return the actual bound port (useful when lock_port=0)."""
        try:
            return self._lock_socket.getsockname()[1]
        except Exception:
            return self._lock_port

    def close(self):
        try:
            self._lock_socket.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()
