import threading, time
import numpy as np

class ObstacleSimulator(threading.Thread):
    def __init__(self, path_func, update_interval=0.05):
        super().__init__(daemon=True)
        self.path_func = path_func
        self.dt       = update_interval
        self.start_t  = time.time()
        self.position = np.zeros(2)  # N, E
        self._running = True

    def run(self):
        while self._running:
            t = time.time() - self.start_t
            self.position = np.array(self.path_func(t))
            time.sleep(self.dt)

    def stop(self):
        self._running = False

def circular_path(t, radius=10.0, omega=0.1):
    return ( radius * np.cos(omega*t),
             radius * np.sin(omega*t) )