# -*- coding: utf-8 -*-
from collections import defaultdict
from timeit import default_timer
from threading import Event, Thread, Lock
import os
import time
import sys

try:
    from dask.callbacks import Callback
    from dask.utils import ignoring
except ImportError as e:
    opt_import_err = e
else:
    opt_import_err = None

from africanus.util.requirements import requires_optional


def format_time(t):
    """Format seconds into a human readable form."""
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)

    if d:
        return ("{0:2.0f}d{0:2.0f}h".format(d, h))
    elif h:
        return "{0:2.0f}h{1:2.0f}m".format(h, m)
    elif m:
        return "{0:2.0f}m{1:2.0f}s".format(m, s)
    else:
        return "{0:4.0f}s".format(s)


def key_bin(key):
    if type(key) == tuple:
        key = key[0]

    if type(key) == bytes:
        key = key.decode()

    try:
        return str(key)
    except Exception:
        return "other"


class TaskData(object):
    __slots__ = ("total", "completed", "time_sum")

    def __init__(self, completed=0, total=0, time_sum=0.0):
        self.completed = completed
        self.total = total
        self.time_sum = time_sum

    def __iadd__(self, other):
        self.completed += other.completed
        self.total += other.total
        self.time_sum += other.time_sum
        return self

    def __add__(self, other):
        return TaskData(self.completed + other.completed,
                        self.total + other.total,
                        self.time_sum + other.time_sum)

    def __repr__(self):
        return "TaskData(%s, %s, %s)" % (self.completed,
                                         self.total,
                                         self.time_sum)

    __str__ = __repr__


def update_bar(elapsed, prev_completed, prev_estimated, pb):
    total = 0
    completed = 0
    estimated = 0.0
    time_guess = 0.0

    # update
    with pb._lock:
        for k, v in pb.task_data.items():
            total += v.total
            completed += v.completed

            if v.completed > 0:
                avg_time = v.time_sum / v.completed
                estimated += avg_time * v.total
                time_guess += v.time_sum

    # If we've completed some new tasks, update our estimate
    # otherwise use previous estimate. This prevents jumps
    # relative to the elapsed time
    if completed != prev_completed:
        estimated = estimated * elapsed / time_guess
    else:
        estimated = prev_estimated

    # Print out the progress bar
    # fraction = completed / total
    fraction = elapsed / estimated if estimated > 0.0 else 0.0
    bar = "#" * int(pb._width * fraction)

    percent = int(100 * fraction)
    msg = "\r[{0:{1}.{1}}] | {2}% Complete (Estimate) | {3} / ~{4}".format(
                bar, pb._width, percent,
                format_time(elapsed),
                "???" if estimated == 0.0 else format_time(estimated))
    with ignoring(ValueError):
        pb._file.write(msg)
        pb._file.flush()

    return completed, estimated


def timer_func(pb):
    start = default_timer()

    while pb.running.is_set():
        elapsed = default_timer() - start
        prev_completed = 0
        prev_estimated = 0.0

        if elapsed > pb._minimum:
            prev_completed, prev_estimated = update_bar(elapsed,
                                                        prev_completed,
                                                        prev_estimated,
                                                        pb)

        time.sleep(pb._dt)


default_out = object()


@requires_optional("dask", opt_import_err)
class EstimatingProgressBar(Callback):
    def __init__(self, minimum=0, width=42, dt=0.1, out=default_out):
        if out is None:
            out = open(os.devnull, "w")
        elif out is default_out:
            out = sys.stdout

        self._minimum = minimum
        self._width = width
        self._dt = dt
        self._file = out
        self._lock = Lock()

    def _start(self, dsk):
        self.task_start = {}
        self.task_data = defaultdict(TaskData)

        for k, v in dsk.items():
            self.task_data[key_bin(k)].total += 1

        self.running = Event()
        self.running.set()
        self.thread = Thread(target=timer_func, args=(self,))
        self.daemon = True
        self.thread.start()

    def _finish(self, dsk, state, errored):
        self.running.clear()
        self.task_data.clear()
        self.task_start.clear()

    def _pretask(self, key, dsk, state):
        with self._lock:
            self.task_start[key] = default_timer()

    def _posttask(self, key, result, dsk, state, worker_id):
        with self._lock:
            td = self.task_data[key_bin(key)]
            td.time_sum += default_timer() - self.task_start.pop(key)
            td.completed += 1
