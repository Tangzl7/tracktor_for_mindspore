from src.tracktor.datasets.mot_sequence import MOTSequence


class MOT17Wrapper:
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split, dets, dataloader):
        mot_dir = './data/MOT17'
        train_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
        test_sequences = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']

        if split == 'train':
            sequences = train_sequences
        elif split == 'test':
            sequences = test_sequences
        elif split == 'all':
            sequences = train_sequences + test_sequences
        elif f"MOT17-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT17-{split}"]
        else:
            raise NotImplementedError('MOT split not available')

        self._data = []
        for s in sequences:
            if dets == 'ALL':
                self._data.append(MOTSequence(f"{s}-DPM", mot_dir, **dataloader))
                self._data.append(MOTSequence(f"{s}-FRCNN", mot_dir, **dataloader))
                self._data.append(MOTSequence(f"{s}-SDP", mot_dir, **dataloader))
            elif dets == 'DPM16':
                self._data.append(MOTSequence(s.replace('17', '16'), 'MOT16', **dataloader))
            else:
                self._data.append(MOTSequence(f"{s}-FRCNN", mot_dir, **dataloader))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]