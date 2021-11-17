from src.tracktor.datasets.mot_wrapper import MOT17Wrapper


_sets = {}

# Fill all available datasets, change here to modify / add new datasets
for split in ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08', '09',
              '10', '11', '12', '13', '14']:
    for dets in ['DPM16', 'DPM', 'FRCNN', 'SDP', 'ALL']:
        name = f'mot17_{split}_{dets}'
        _sets[name] = (lambda *args, split=split,
                       dets=dets: MOT17Wrapper(split, dets, *args))


class Datasets:
    """A central class to manage the individual dataset loaders"""

    def __init__(self, datasets, *args):
        if isinstance(datasets, str):
            datasets = [datasets]

        if len(args) == 0:
            args = [{}]

        self.datasets = None
        for dataset in datasets:
            assert dataset in _sets, f"[!] Dataset not found: {dataset}"

            self.datasets = _sets[dataset](*args)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]