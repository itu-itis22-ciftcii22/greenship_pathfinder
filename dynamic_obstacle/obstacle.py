import threading
import time
import numpy as np

class ObstacleSimulator(threading.Thread):
    def __init__(self, path_func, base, target, update_interval=0.05):
        super().__init__(daemon=True)
        self.path_func      = path_func        # signature: f(t, base: array, target: array) -> array
        self.base           = base
        self.target         = target
        self.dt             = update_interval
        self.start_t        = time.time()
        self.position       = np.zeros(2)
        self._running       = True
        self._lock          = threading.Lock()

    def run(self):
        while self._running:
            t      = time.time() - self.start_t
            new_pos = np.array(self.path_func(t, self.base, self.target))
            with self._lock:
                self.position = new_pos
            time.sleep(self.dt)

    def get_position(self):
        with self._lock:
            return self.position.copy()

    def stop(self):
        self._running = False


# Example path functions:

def linear_oscillation(t, base, target, period=5.0):
    """
    Moves back and forth along the segment [base → target].
    fraction = 0.5 * (1 + sin(2π t / period))
    """
    frac = 0.5 * (1 + np.sin(2 * np.pi * t / period))
    return base + frac * (target - base)

def circular_between(t, base, target, omega=0.2):
    """
    Circles around the midpoint of base and target,
    with radius = half the distance.
    """
    midpoint = (base + target) / 2
    radius   = 1
    return midpoint + radius * np.array([
        np.cos(omega * t),
        np.sin(omega * t)
    ])
