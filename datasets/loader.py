import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
import pdb

# train_dataset, batch_size=batch_size,
#                                              num_workers=self.settings.num_cpu_workers,
#                                              shuffle=True, device=self.device)

class Loader:
    def __init__(self, dataset, batch_size, num_workers, shuffle, device):
        self.device = device
        split_indices = list(range(len(dataset)))
        self.dataset = dataset
        sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)

        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                            num_workers=num_workers,
                                            collate_fn=collate_events)
        
    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) if not isinstance(d, list) else d for d in data]
            yield data

    def __len__(self):
        return len(self.loader)
    
    def __dataset__(self):
        return self.dataset

def collate_events(data):
    labels = []
    events = []
    images = []
    stacks = []
    # if len(data) == 2:
    #     for i, d in enumerate(data):
    #         labels.append(d[1])
    #         ev = np.concatenate([d[0], i*np.ones((len(d[0]),1), dtype=np.float32)],1)
    #         events.append(ev)
    #     events = torch.from_numpy(np.concatenate(events,0))
    #     labels = default_collate(labels)
    #     return events, labels
    # elif len(data) == 3:
    for i, d in enumerate(data):
        stacks.append(d[3])
        labels.append(d[2])
        images.append(d[1])
        ev = np.concatenate([d[0], i*np.ones((len(d[0]),1), dtype=np.float32)],1)
        events.append(ev)
    events = torch.from_numpy(np.concatenate(events,0))
    labels = default_collate(labels)
    images = default_collate(images)
    stacks = default_collate(stacks)
    return events, images, labels, stacks