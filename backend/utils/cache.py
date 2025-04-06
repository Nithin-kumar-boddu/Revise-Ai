import time
from threading import Lock

class Cache:
    """Simple in-memory cache with expiration"""
    
    def __init__(self, expiration=3600):
        """
        Initialize cache
        
        Args:
            expiration: Cache expiration time in seconds (default: 1 hour)
        """
        self.cache = {}
        self.expiration = expiration
        self.lock = Lock()
    
    def get(self, key):
        """Get a value from the cache if it exists and is not expired"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry['expiry'] > time.time():
                    return entry['value']
                else:
                    # Remove expired entry
                    del self.cache[key]
        return None
    
    def set(self, key, value):
        """Set a value in the cache with expiration"""
        with self.lock:
            self.cache[key] = {
                'value': value,
                'expiry': time.time() + self.expiration
            }
    
    def clear(self):
        """Clear the entire cache"""
        with self.lock:
            self.cache = {}
    
    def remove(self, key):
        """Remove a specific key from the cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]