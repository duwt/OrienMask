from torch.utils.data import DataLoader

from .collate import naive_collate


class AspectRatioGroupedDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actual_collate_fn = self.collate_fn
        self.collate_fn = naive_collate

    def __iter__(self):
        return AspectRatioGroupedIter(
            super().__iter__(), len(self.batch_sampler),
            self.batch_size, self.actual_collate_fn
        )


class AspectRatioGroupedIter:
    def __init__(self, dataloader_iter, max_batches, batch_size, collate_fn):
        self.dataloader_iter = dataloader_iter
        self.max_batches = max_batches
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.data_count = 0
        self.grouped_data = [[], []]

    def __iter__(self):
        return self

    def __next__(self):
        # return collate_plus(self.dataloader_iter._dataset_fetcher.fetch([974, 2493]))
        if len(self.grouped_data[0]) >= self.batch_size:
            batch = self.grouped_data[0][:self.batch_size]
            self.grouped_data[0] = self.grouped_data[0][self.batch_size:]
            return self.collate_fn(batch)
        elif len(self.grouped_data[1]) >= self.batch_size:
            batch = self.grouped_data[1][:self.batch_size]
            self.grouped_data[1] = self.grouped_data[1][self.batch_size:]
            return self.collate_fn(batch)
        else:
            if self.data_count == self.max_batches and \
                    (len(self.grouped_data[0]) or len(self.grouped_data[1])):
                num_needed = self.batch_size - len(self.grouped_data[0])
                last_batch = self.grouped_data[0] + self.grouped_data[1][:num_needed]
                self.grouped_data[0] = []
                self.grouped_data[1] = self.grouped_data[1][num_needed:]
                if len(self.grouped_data[1]) == 0:
                    self.data_count = 0
                return self.collate_fn(last_batch)
            data = next(self.dataloader_iter)
            self.data_count += 1
            for sample in data:
                h, w = sample['image'].shape[-2:]
                group = 0 if h > w else 1
                self.grouped_data[group].append(sample)
            return next(self)
