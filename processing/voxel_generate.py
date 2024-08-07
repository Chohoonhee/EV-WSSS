import os
import numpy as np
import pdb
import torch
from pathlib import Path
import hdf5plugin
import h5py
from eventslicer import EventSlicer
from representations import VoxelGrid
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from joblib import Parallel, delayed
import tqdm
import argparse

height = 480
width = 640



event_representation: str = 'voxel_grid'
                 


mode = 'train'


if __name__=='__main__':
    # dataset_dir = './stvsr_parse'
    parser = argparse.ArgumentParser(description='Dataset Processing.')
    parser.add_argument('--dataset_path', help='Path to dataset', required=True)
    args = parser.parse_args()
    
    dataset_dir = os.path.join(args.dataset_path, mode)
    
    # dataset_dir = './' + mode
    folder_list_all = os.listdir(dataset_dir)
    folder_list_all.sort()
    
    event_prefix = 'events'
    
    
    class Sequence():
        def __init__(self, seq_path: Path, event_representation: str = 'voxel_grid'):
            
            self.height = 480
            self.width = 640
            
            
            nr_events_data: int = 20
            delta_t_per_data: int = 50
            nr_events_per_data: int = 100000
            nr_bins_per_data: int = 5
            require_paired_data=False
            normalize_event=False
            separate_pol=False
            fixed_duration=False
            remove_time_window: int = 250
            short_mul=5.0
            require_paired_data=True
            semseg_num_classes: int = 11
            
            
            
            self.short_mul = short_mul
            assert nr_bins_per_data >= 1
            assert seq_path.is_dir()
            self.sequence_name = seq_path.name
            
            
            self.mode = mode
            
            # Set event representation
            self.nr_events_data = nr_events_data
            self.num_bins = nr_bins_per_data
            assert nr_events_per_data > 0
            self.nr_events_per_data = nr_events_per_data
            self.event_representation = event_representation
            self.separate_pol = separate_pol
            self.normalize_event = normalize_event
            self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=self.normalize_event)
            self.short_voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=self.normalize_event)

            self.locations = ['left']
            self.semseg_num_classes = semseg_num_classes
            
            # Save delta timestamp
            self.fixed_duration = fixed_duration
            if self.fixed_duration:
                delta_t_ms = nr_events_data * delta_t_per_data
                self.delta_t_us = delta_t_ms * 1000
            self.remove_time_window = remove_time_window

            self.require_paired_data = require_paired_data

            # load timestamps
            # self.timestamps = np.loadtxt(str(seq_path / 'semantic' / 'timestamps.txt'), dtype='int64')
            self.timestamps = np.loadtxt(str(seq_path / 'timestamps.txt'), dtype='int64')
            
            
            if mode == 'train':
                if self.semseg_num_classes == 11:
                    # label_dir = seq_path / '11classes_weak'
                    label_dir = seq_path / '11classes_weak_1point_per_class' 
                elif self.semseg_num_classes == 19:
                    label_dir = seq_path / '19classes_weak' 
                else:
                    raise ValueError
            else:
                if self.semseg_num_classes == 11:
                    label_dir = seq_path / '11classes' 
                elif self.semseg_num_classes == 19:
                    label_dir = seq_path / '19classes' 
                else:
                    raise ValueError
            assert label_dir.is_dir()
            label_pathstrings = list()
            for entry in label_dir.iterdir():
                assert str(entry.name).endswith('.png')
                label_pathstrings.append(str(entry))
            label_pathstrings.sort()
            self.label_pathstrings = label_pathstrings
            
            assert len(self.label_pathstrings) == self.timestamps.size
            
            # load images paths
            if self.require_paired_data:
                # img_dir = seq_path / 'images'
                img_dir = seq_path 
                img_left_dir = img_dir / 'left' / 'ev_inf'
                assert img_left_dir.is_dir()
                img_left_pathstrings = list()
                for entry in img_left_dir.iterdir():
                    assert str(entry.name).endswith('.png')
                    img_left_pathstrings.append(str(entry))
                img_left_pathstrings.sort()
                self.img_left_pathstrings = img_left_pathstrings

                assert len(self.img_left_pathstrings) == self.timestamps.size

            
            
            # Remove several label paths and corresponding timestamps in the remove_time_window.
            # This is necessary because we do not have enough events before the first label.
            self.timestamps = self.timestamps[(self.remove_time_window // 100 + 1) * 2:]
            del self.label_pathstrings[:(self.remove_time_window // 100 + 1) * 2]
            assert len(self.label_pathstrings) == self.timestamps.size
            if self.require_paired_data:
                del self.img_left_pathstrings[:(self.remove_time_window // 100 + 1) * 2]
                assert len(self.img_left_pathstrings) == self.timestamps.size

            self.h5f = dict()
            self.rectify_ev_maps = dict()
            self.event_slicers = dict()

            self.ev_voxel_dir = dict()
            self.ev_reverse_voxel_dir = dict()
            
            # ev_dir = seq_path / 'events'
            ev_dir = seq_path 
            for location in self.locations:
                ev_dir_location = ev_dir / location
                ev_data_file = ev_dir_location / 'events.h5'
                ev_rect_file = ev_dir_location / 'rectify_map.h5'

                h5f_location = h5py.File(str(ev_data_file), 'r')
                self.h5f[location] = h5f_location
                self.event_slicers[location] = EventSlicer(h5f_location)
                with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                    self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]
                    
                ev_voxel_dir = ev_dir_location / 'voxel'
                ev_reverse_voxel_dir = ev_dir_location / 'reverse_voxel_5'
                if not os.path.exists(ev_voxel_dir):
                    os.makedirs(ev_voxel_dir)
                if not os.path.exists(ev_reverse_voxel_dir):
                    os.makedirs(ev_reverse_voxel_dir)

                self.ev_voxel_dir[location] = ev_voxel_dir
                self.ev_reverse_voxel_dir[location] = ev_reverse_voxel_dir
                            
                    
        def events_to_voxel_grid(self, x, y, p, t):
            t = (t - t[0]).astype('float32')
            t = (t/t[-1])
            x = x.astype('float32')
            y = y.astype('float32')
            pol = p.astype('float32')
            return self.voxel_grid.convert(
                    torch.from_numpy(x),
                    torch.from_numpy(y),
                    torch.from_numpy(pol),
                    torch.from_numpy(t))

        def events_to_short_voxel_grid(self, x, y, p, t):
            t = (t - t[0]).astype('float32')
            t = (t/t[-1])
            x = x.astype('float32')
            y = y.astype('float32')
            pol = p.astype('float32')
            return self.short_voxel_grid.convert(
                    torch.from_numpy(x),
                    torch.from_numpy(y),
                    torch.from_numpy(pol),
                    torch.from_numpy(t))


        def getHeightAndWidth(self):
            return self.height, self.width
        
        
        @staticmethod
        def get_img(filepath: Path, shape_resize=None):
            assert filepath.is_file()
            img = Image.open(str(filepath))
            if shape_resize is not None:
                img = img.resize((shape_resize[1], shape_resize[0]))
            img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = img_transform(img)
            return img_tensor

        @staticmethod
        def get_label(filepath: Path):
            assert filepath.is_file()
            label = Image.open(str(filepath))
            label = np.array(label)
            return label

        @staticmethod
        def close_callback(h5f_dict):
            for k, h5f in h5f_dict.items():
                h5f.close()

        def __len__(self):
            return (self.timestamps.size + 1) // 2

        def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
            assert location in self.locations
            # From distorted to undistorted
            rectify_map = self.rectify_ev_maps[location]
            assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
            assert x.max() < self.width
            assert y.max() < self.height
            return rectify_map[y, x]

        def generate_event_tensor(self, job_id, events, event_tensor, nr_events_per_data):
            id_start = job_id * nr_events_per_data
            id_end = (job_id + 1) * nr_events_per_data
            events_temp = events[id_start:id_end]
            event_representation = self.events_to_voxel_grid(events_temp[:, 0], events_temp[:, 1], events_temp[:, 3],
                                                            events_temp[:, 2])
            event_tensor[(job_id * self.num_bins):((job_id+1) * self.num_bins), :, :] = event_representation
            
        
        def generate_short_event_tensor(self, job_id, short_events, short_event_tensor, nr_events_per_data):
            id_start = job_id * nr_events_per_data
            id_end = (job_id + 1) * nr_events_per_data
            events_temp = short_events[id_start:id_end]
            event_representation = self.events_to_voxel_grid(events_temp[:, 0], events_temp[:, 1], events_temp[:, 3],
                                                            events_temp[:, 2])
            short_event_tensor[(job_id * self.num_bins):((job_id+1) * self.num_bins), :, :] = event_representation
            
            
        def getitem(self, index):
            label_path = Path(self.label_pathstrings[index * 2])
            
            if self.mode == 'train':
                dense_label_path = Path(str(label_path).replace('_weak_1point_per_class', ''))
            
            ts_end = self.timestamps[index * 2]

            output = {}
            for location in self.locations:
                if self.fixed_duration:
                    ts_start = ts_end - self.delta_t_us
                    event_tensor = None
                    self.delta_t_per_data_us = self.delta_t_us / self.nr_events_data
                    for i in range(self.nr_events_data):
                        t_s = ts_start + i * self.delta_t_per_data_us
                        t_end = ts_start + (i+1) * self.delta_t_per_data_us
                        event_data = self.event_slicers[location].get_events(t_s, t_end)

                        p = event_data['p']
                        t = event_data['t']
                        x = event_data['x']
                        y = event_data['y']

                        xy_rect = self.rectify_events(x, y, location)
                        x_rect = xy_rect[:, 0]
                        y_rect = xy_rect[:, 1]

                     
                        event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
                   

                        if event_tensor is None:
                            event_tensor = event_representation
                        else:
                            event_tensor = torch.cat([event_tensor, event_representation], dim=0)

                else:
                    num_bins_total = self.nr_events_data * self.num_bins
                    long_event_tensor = torch.zeros((num_bins_total, self.height, self.width))
                    short_event_tensor = torch.zeros((num_bins_total, self.height, self.width))
                    
                    self.nr_events = self.nr_events_data * self.nr_events_per_data
                    event_data = self.event_slicers[location].get_events_fixed_num(ts_end, self.nr_events)

                    if self.nr_events >= event_data['t'].size:
                        start_index = 0
                    else:
                        start_index = -self.nr_events

                    p = event_data['p'][start_index:]
                    t = event_data['t'][start_index:]
                    x = event_data['x'][start_index:]
                    y = event_data['y'][start_index:]
                    nr_events_loaded = t.size

                    xy_rect = self.rectify_events(x, y, location)
                    x_rect = xy_rect[:, 0]
                    y_rect = xy_rect[:, 1]

                    ### long term events ###
                    nr_events_temp = nr_events_loaded // self.nr_events_data
                    events = np.stack([x_rect, y_rect, t, p], axis=-1)
                    Parallel(n_jobs=4, backend="threading")(
                        delayed(self.generate_event_tensor)(i, events, long_event_tensor, nr_events_temp) for i in range(self.nr_events_data))

                    if self.short_mul >= 1.0:
                        self.nr_events = int(self.nr_events_data * self.nr_events_per_data * self.short_mul)
                        
                        # event_data = self.event_slicers[location].get_events_fixed_num(ts_end, self.nr_events)
                        event_data = self.event_slicers[location].get_events_fixed_num_reverse(ts_end, self.nr_events)
                
                        if self.nr_events >= event_data['t'].size:
                            start_index = 0
                        else:
                            start_index = -self.nr_events

                        p = event_data['p'][start_index:]
                        t = event_data['t'][start_index:]
                        x = event_data['x'][start_index:]
                        y = event_data['y'][start_index:]
                        
                        p = 1 - p
                        p = p[::-1]
                        t = t[::-1]
                        x = x[::-1]
                        y = y[::-1]
                        t = t.max() - t
                        
                        
                        nr_events_loaded = t.size

                        xy_rect = self.rectify_events(x, y, location)
                        x_rect = xy_rect[:, 0]
                        y_rect = xy_rect[:, 1]

                        ### long term events ###
                        short_nr_events_temp = nr_events_loaded // self.nr_events_data
                        short_events = np.stack([x_rect, y_rect, t, p], axis=-1)
                        Parallel(n_jobs=4, backend="threading")(
                            delayed(self.generate_short_event_tensor)(i, short_events, short_event_tensor, short_nr_events_temp) for i in range(self.nr_events_data))

                    else:
                        ### short term events ###
                        # short_events_loaded = nr_events_loaded // 20
                        # short_nr_events_temp = nr_events_temp // 20
                        short_events_loaded = int(nr_events_loaded * self.short_mul)
                        short_nr_events_temp = short_events_loaded // self.nr_events_data
                        short_events = events[-short_events_loaded:]
                        Parallel(n_jobs=4, backend="threading")(
                            delayed(self.generate_short_event_tensor)(i, short_events, short_event_tensor, short_nr_events_temp) for i in range(self.nr_events_data))

                    

                # remove 40 bottom rows
                event_tensor = long_event_tensor[:, :-40, :]
                short_event_tensor = short_event_tensor[:, :-40, :]

                event_save_path = self.ev_voxel_dir[location] / str(dense_label_path).split('/')[-1].replace('png', 'npy')
                
                reverse_event_save_path = self.ev_reverse_voxel_dir[location] / str(dense_label_path).split('/')[-1].replace('png', 'npy')
                
                np.save(event_save_path, event_tensor)
                np.save(reverse_event_save_path, short_event_tensor)
                
                
                # print(dense_label_path, event_save_path, reverse_event_save_path)
                        

            # if 'representation' not in output:
            #     output['representation'] = dict()
            # output['representation'][location] = event_tensor

            # pdb.set_trace()
           
        
        
        
        
        
        
        
        
    
    for folder_name in folder_list_all:
        
        seq_path = Path(os.path.join(dataset_dir, folder_name))

        if not seq_path.is_dir():
            continue
        print(folder_name)

        folder_sequence = Sequence(seq_path)
        
        for i in tqdm.tqdm(range(len(folder_sequence))):
            folder_sequence.getitem(i)
     

      
