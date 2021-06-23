import torch


class EvalCounter:
    """A counter for loss or metric

    Attributes:
        items (dict(str: int)) - the number of items during one stage
        counter (dict(str: float)) - sum of valid data during one stage
        items_epoch (dict(str: int)) - the number of items during one epoch
        counter_epoch (dict(str: float)) - sum of valid data during one epoch
    """

    def __init__(self):
        self.keys = []
        self.items = {}
        self.counter = {}
        self.items_epoch = {}
        self.counter_epoch = {}

    def reset(self):
        for key in self.keys:
            self.items_epoch[key] += self.items[key]
            self.counter_epoch[key] += self.counter[key]
        self.items = {key: 0 for key in self.keys}
        self.counter = {key: 0. for key in self.keys}

    def reset_epoch(self):
        self.items = {key: 0 for key in self.keys}
        self.counter = {key: 0. for key in self.keys}
        self.items_epoch = {key: 0 for key in self.keys}
        self.counter_epoch = {key: 0. for key in self.keys}

    def update(self, key, value):
        value, item = value if isinstance(value, (dict, tuple)) else (value, 1)
        try:
            self.items[key] += item
            self.counter[key] += value
        except KeyError:
            self.keys.append(key)
            self.items[key] = item
            self.counter[key] = value
            self.items_epoch[key] = 0
            self.counter_epoch[key] = 0.

    def average(self, key):
        try:
            return self.counter[key] / self.items[key]
        except ZeroDivisionError:
            return -1

    def average_epoch(self, key):
        try:
            self.items_epoch[key] += self.items[key]
            self.counter_epoch[key] += self.counter[key]
            self.items[key] = 0
            self.counter[key] = 0.
            return self.counter_epoch[key] / self.items_epoch[key]
        except ZeroDivisionError:
            return -1

    def save(self, filename):
        torch.save({'items': self.items, 'counter': self.counter}, filename)

    def save_epoch(self, filename):
        torch.save({'items_epoch': self.items_epoch,
                    'counter_epoch': self.counter_epoch}, filename)

    def merge(self, counter_dict):
        for key in self.keys:
            self.items[key] += counter_dict['items'][key]
            self.counter[key] += counter_dict['counter'][key]

    def merge_epoch(self, counter_dict):
        for key in self.keys:
            self.items_epoch[key] += counter_dict['items_epoch'][key]
            self.counter_epoch[key] += counter_dict['counter_epoch'][key]
