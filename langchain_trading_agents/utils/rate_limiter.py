"""
rate_limiter.py
Token bucket rate limiter for LLM API calls
Supports configurable request limits within time windows
"""
import asyncio
import time
from typing import Optional
from collections import deque


class RateLimiter:
    """
    Token bucket rate limiter that tracks API calls within a time window.
    
    Args:
        max_requests: Maximum number of requests allowed in the time window
        time_window_seconds: Time window in seconds (default: 5 hours = 18000 seconds)
    """
    
    def __init__(self, max_requests: int = 600, time_window_seconds: int = 18000):
        self.max_requests = max_requests
        self.time_window_seconds = time_window_seconds
        self.request_timestamps: deque = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """
        Try to acquire a request slot. Returns True if successful, False if rate limit exceeded.
        """
        async with self._lock:
            current_time = time.time()
            
            # Remove timestamps outside the current time window
            while (
                self.request_timestamps 
                and current_time - self.request_timestamps[0] > self.time_window_seconds
            ):
                self.request_timestamps.popleft()
            
            # Check if we can make another request
            if len(self.request_timestamps) < self.max_requests:
                self.request_timestamps.append(current_time)
                return True
            
            return False
    
    async def wait_for_slot(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until a request slot is available or timeout is reached.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait indefinitely)
        
        Returns:
            True if slot acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            if await self.acquire():
                return True
            
            # Calculate how long to wait
            async with self._lock:
                if not self.request_timestamps:
                    continue
                
                oldest_timestamp = self.request_timestamps[0]
                wait_time = (
                    oldest_timestamp + self.time_window_seconds - time.time()
                ) + 0.1  # Add small buffer
            
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
                wait_time = min(wait_time, timeout - elapsed)
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
    
    def get_current_usage(self) -> dict:
        """
        Get current usage statistics.
        
        Returns:
            Dictionary with current usage info
        """
        current_time = time.time()
        
        # Clean old timestamps
        while (
            self.request_timestamps 
            and current_time - self.request_timestamps[0] > self.time_window_seconds
        ):
            self.request_timestamps.popleft()
        
        return {
            "current_requests": len(self.request_timestamps),
            "max_requests": self.max_requests,
            "time_window_hours": self.time_window_seconds / 3600,
            "remaining": max(0, self.max_requests - len(self.request_timestamps)),
            "oldest_request_age_seconds": (
                current_time - self.request_timestamps[0] 
                if self.request_timestamps 
                else 0
            )
        }
    
    def reset(self):
        """Reset all request history."""
        self.request_timestamps.clear()


# Global rate limiter instance for MiniMax
minimax_rate_limiter = RateLimiter(max_requests=600, time_window_seconds=18000)  # 5 hours = 18000 seconds
