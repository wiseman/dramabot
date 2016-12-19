
import time

class Timer(object):
    """Keeps track of wall-clock time."""
    def __init__(self):
        self.start_time = None
        self.reset()

    def reset(self):
        """Resets the timer."""
        self.start_time = time.time()

    def elapsed(self):
        """Returns the elapsed time in seconds.

        Elapsed time is the time since the timer was created or last
        reset.
        """
        return time.time() - self.start_time
