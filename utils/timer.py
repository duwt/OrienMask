import time
from collections import OrderedDict

import numpy as np
import torch
from prettytable import PrettyTable

_is_cuda = True
_timer_history = OrderedDict()


def cuda():
    global _is_cuda
    _is_cuda = True


def cpu():
    global _is_cuda
    _is_cuda = False


def reset():
    _timer_history.clear()


def get_elapsed_time(timer_list):
    elapsed_times = [t.elapsed_time() for t in timer_list]
    return np.array(elapsed_times).mean()


def get_all_elapsed_time():
    global _is_cuda
    if _is_cuda:
        torch.cuda.synchronize()
    elapsed_time_dict = dict()
    for key, value in _timer_history.items():
        elapsed_time_dict[key] = get_elapsed_time(value)
    return elapsed_time_dict


def log_elapsed_time(logger=None):
    elapsed_time_table = PrettyTable()
    elapsed_time_table.field_names = ["Item", "Time (ms)", "FPS"]
    elapsed_time_table.align["Train"] = 'l'
    elapsed_time_table.float_format = '.2'
    all_elapsed_time = get_all_elapsed_time()
    for item in _timer_history.keys():
        elapsed_time_table.add_row([item, all_elapsed_time[item], 1000 / all_elapsed_time[item]])
    if logger:
        logger.info('\n' + str(elapsed_time_table))
    else:
        print('\n' + str(elapsed_time_table))


class CPUTimer:
    def __init__(self):
        self.start_timer = -1
        self.end_timer = -1

    def start(self):
        self.start_timer = time.time() * 1000

    def end(self):
        self.end_timer = time.time() * 1000

    def elapsed_time(self):
        return self.end_timer - self.start_timer


class CUDATimer:
    def __init__(self):
        self.start_timer = torch.cuda.Event(enable_timing=True)
        self.end_timer = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_timer.record()

    def end(self):
        self.end_timer.record()

    def elapsed_time(self):
        return self.start_timer.elapsed_time(self.end_timer)


class timer:
    """A timing wrapper with cpu or cuda backend

    Common usage:
        timer.cuda()
        timer.reset()
        for _ in range(N):
            with timer.timer('stage a'):
                run_stage_a()
            with timer.timer('stage b'):
                run_stage_b()
        timer.log_elapsed_time(logger=None)
    """

    def __init__(self, name):
        global _is_cuda
        self.name = name
        if name not in _timer_history:
            _timer_history[name] = list()
        self.idx = len(_timer_history[name])
        _timer_history[name].append(CUDATimer() if _is_cuda else CPUTimer())

    def __enter__(self):
        _timer_history[self.name][self.idx].start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        _timer_history[self.name][self.idx].end()
