import os
import time
import threading


class RateLimiter:
    """Simple process-wide token-bucket rate limiter with cooldown.

    - rps: steady tokens per second
    - burst: bucket capacity
    - cooldown: when set, pauses token issuance until cooldown ends
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, rps: float = 2.0, burst: float = 2.0):
        self.rps = max(0.1, float(rps))
        self.burst = max(1.0, float(burst))
        self._tokens = self.burst
        self._last = time.monotonic()
        self._cooldown_until = 0.0
        self._mutex = threading.Lock()

    @classmethod
    def from_env(cls):
        with cls._lock:
            if cls._instance is None:
                rps = float(os.getenv('BERGHAIN_RPS', '2.0'))
                burst = float(os.getenv('BERGHAIN_BURST', '2.0'))
                cls._instance = cls(rps=rps, burst=burst)
            return cls._instance

    def _refill(self):
        now = time.monotonic()
        dt = now - self._last
        if dt > 0:
            self._tokens = min(self.burst, self._tokens + dt * self.rps)
            self._last = now

    def wait_for_token(self):
        while True:
            with self._mutex:
                now = time.monotonic()
                if now < self._cooldown_until:
                    sleep_for = self._cooldown_until - now
                else:
                    self._refill()
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return
                    # time until next token
                    needed = 1.0 - self._tokens
                    sleep_for = max(0.01, needed / self.rps)
            time.sleep(min(1.0, sleep_for))

    def cooldown(self, seconds: float):
        with self._mutex:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + max(0.0, float(seconds)))

