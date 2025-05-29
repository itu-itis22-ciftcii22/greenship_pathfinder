import threading
import time
import numpy as np

class ObstacleSimulator(threading.Thread):
    def __init__(self, path_func, update_interval=0.05):
        super().__init__(daemon=True)
        self.path_func      = path_func
        self.dt             = update_interval
        self.start_t        = time.time()
        self.position       = np.zeros(2)  # N, E
        self._running       = True
        self._lock          = threading.Lock()

    def run(self):
        while self._running:
            t = time.time() - self.start_t
            with self._lock:
                center = self.position.copy()
            new_pos = np.array(self.path_func(t, center))
            with self._lock:
                self.position = new_pos
            time.sleep(self.dt)

    def update_position(self, new_pos):
        with self._lock:
            self.position = np.array(new_pos)

    def stop(self):
        self._running = False


def circular_path(t, center, radius=10.0, omega=0.1):
    x0, y0 = center
    return (
        x0 + radius * np.cos(omega * t),
        y0 + radius * np.sin(omega * t)
    )
