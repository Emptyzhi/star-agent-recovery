"""Task threads and isolated environment processes; no candidate generation pool."""
from concurrent.futures import Future,ProcessPoolExecutor,ThreadPoolExecutor,as_completed,wait
