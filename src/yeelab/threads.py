"""One thread pool for the compiled kernels (beta_tables.py, methods.py) and for
scipy's special functions (shares.py), which all release the GIL.

The kernels are not parallel themselves: a parallel numba kernel called from two
threads at once, as two requests of the web UI can, aborts with numba's default
threading layer. They run on chunks of their work in these threads instead.
"""

import os
from concurrent.futures import ThreadPoolExecutor

pool = ThreadPoolExecutor(max_workers=os.cpu_count())


def in_chunks(run, rows, chunk):
    """run(start, stop) for consecutive chunks of `rows` rows, in parallel."""
    list(pool.map(lambda start: run(start, min(start + chunk, rows)), range(0, rows, chunk)))
