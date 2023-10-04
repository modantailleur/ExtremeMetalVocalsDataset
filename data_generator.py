import numpy as np
import h5py
import csv
import time
import logging
import torch
#from utilities import int16_to_float32
from sklearn.preprocessing import OneHotEncoder

class RegularSplitDataset(object):
    def __init__(self, hdf5_path, split_id, split_type='train'):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 
        """
        self.hdf5_path = hdf5_path
        self.split_type = split_type
        self.split_id = split_id
        #self.singer_data_lengths = np.zeros(len(split_id))

        # classes_weight_dict = {
        #     'ClearVoice': 1/39.868421,
        #     'DeathGrowl': 1/23.947368,
        #     'HardcoreScream': 1/21.447368,
        #     'BlackShriek': 1/14.736842
        # }
        
        classes = ['ClearVoice', 'DeathGrowl', 'HardcoreScream', 'BlackShriek']
        classes_dict = {
            'ClearVoice': 0,
            'DeathGrowl': 1,
            'HardcoreScream': 2,
            'BlackShriek': 3
        }
        # classes_weight_dict = {
        #     'ClearVoice': 39.868421,
        #     'DeathGrowl': 23.947368,
        #     'HardcoreScream': 21.447368,
        #     'BlackShriek': 14.736842
        # }

        with h5py.File(self.hdf5_path, 'r') as hf:
            gt = hf[f'split{split_id}'][split_type]['groundtruth'][:]
            gt = np.array([item.decode('utf-8') for item in gt])
            class_sample_count = np.array(
                        [len(np.where(gt == t)[0]) for t in classes_dict.keys()])
            weight = 1. / class_sample_count

            self.samples_weight = np.array([weight[classes_dict[t]] for t in gt])

            self.audio_names = [audio_name for audio_name in hf[f'split{split_id}'][split_type]['audio_name'][:]]
            # self.samples_weight = [classes_weight_dict[gt.decode('utf-8')] for gt in hf[f'regular_split'][split_type]['groundtruth'][:]]

        self.len_dataset = len(self.audio_names)

        self.gt_types = ['ClearVoice', 'DeathGrowl', 'HardcoreScream', 'BlackShriek']
        self.gt_str_to_onehot = {voice_type: [1 if i == idx else 0 for i in range(len(self.gt_types))] for idx, voice_type in enumerate(self.gt_types)}
        self.gt_list_to_str = {i: voice_type for i, voice_type in enumerate(self.gt_types)}


        #self.gt_list_to_str = {tuple(v): k for k, v in self.gt_str_to_list.items()}
        # self.enc = OneHotEncoder(handle_unknown='ignore')
        # self.enc.fit(np.array(self.gt_types).reshape(-1, 1))


    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        # hdf5_path = meta['hdf5_path']
        # index_in_hdf5 = meta['index_in_hdf5']
        # singer_id = self.singer_ids[0]

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_name =  hf[f'split{self.split_id}'][self.split_type]['audio_name'][idx].decode('utf-8')
            mel_spectrogram =  hf[f'split{self.split_id}'][self.split_type]['mel_spectrogram'][idx]
            groundtruth =  hf[f'split{self.split_id}'][self.split_type]['groundtruth'][idx]
            groundtruth = groundtruth.decode('utf-8')
            groundtruth = self.gt_str_to_onehot[groundtruth]
            # print('CCCCCC')
            # print(groundtruth)
            # groundtruth = self.enc.transform()
            # print('AAAAAAAA')
            # print(groundtruth)

        return idx, audio_name, torch.Tensor(mel_spectrogram), torch.Tensor(groundtruth)

    
    def __len__(self):
        return self.len_dataset


class SingerSplitDataset(object):
    def __init__(self, hdf5_path, split_id, split_type='train'):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 
        """
        self.hdf5_path = hdf5_path
        self.split_id = split_id
        self.split_type = split_type
        #self.singer_data_lengths = np.zeros(len(split_id))

        with h5py.File(self.hdf5_path, 'r') as hf:
            self.audio_names = [audio_name for audio_name in hf[f'Singer{split_id+1}'][split_type]['audio_name'][:]]
        self.len_dataset = len(self.audio_names)

        self.gt_types = ['ClearVoice', 'DeathGrowl', 'HardcoreScream', 'BlackShriek']
        self.gt_str_to_onehot = {voice_type: [1 if i == idx else 0 for i in range(len(self.gt_types))] for idx, voice_type in enumerate(self.gt_types)}
        self.gt_list_to_str = {i: voice_type for i, voice_type in enumerate(self.gt_types)}


        #self.gt_list_to_str = {tuple(v): k for k, v in self.gt_str_to_list.items()}
        # self.enc = OneHotEncoder(handle_unknown='ignore')
        # self.enc.fit(np.array(self.gt_types).reshape(-1, 1))


    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        # hdf5_path = meta['hdf5_path']
        # index_in_hdf5 = meta['index_in_hdf5']
        # singer_id = self.singer_ids[0]

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_name =  hf[f'Singer{self.split_id+1}'][self.split_type]['audio_name'][idx].decode('utf-8')
            mel_spectrogram =  hf[f'Singer{self.split_id+1}'][self.split_type]['mel_spectrogram'][idx]
            groundtruth =  hf[f'Singer{self.split_id+1}'][self.split_type]['groundtruth'][idx]
            groundtruth = groundtruth.decode('utf-8')
            groundtruth = self.gt_str_to_onehot[groundtruth]
            # print('CCCCCC')
            # print(groundtruth)
            # groundtruth = self.enc.transform()
            # print('AAAAAAAA')
            # print(groundtruth)

        return idx, audio_name, torch.Tensor(mel_spectrogram), torch.Tensor(groundtruth)

    
    def __len__(self):
        return self.len_dataset

class SingerClassificationDataset(object):
    def __init__(self, hdf5_path, split_id, split_type='train'):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 
        """
        self.hdf5_path = hdf5_path
        self.split_type = split_type
        self.split_id = split_id
        #self.singer_data_lengths = np.zeros(len(split_id))

        # classes_weight_dict = {
        #     'ClearVoice': 1/39.868421,
        #     'DeathGrowl': 1/23.947368,
        #     'HardcoreScream': 1/21.447368,
        #     'BlackShriek': 1/14.736842
        # }
        
        classes_dict = {i: i - 1 for i in range(1, 28)}
        # classes_weight_dict = {
        #     'ClearVoice': 39.868421,
        #     'DeathGrowl': 23.947368,
        #     'HardcoreScream': 21.447368,
        #     'BlackShriek': 14.736842
        # }

        with h5py.File(self.hdf5_path, 'r') as hf:
            gt = hf[f'split{split_id}'][split_type]['singer'][:]
            # gt = np.array([int(item.decode('utf-8')) for item in gt])
            gt = np.array(gt)
            class_sample_count = np.array(
                        [len(np.where(gt == t)[0]) for t in classes_dict.keys()])
            weight = 1. / class_sample_count

            self.samples_weight = np.array([weight[classes_dict[t]] for t in gt])

            self.audio_names = [audio_name for audio_name in hf[f'split{split_id}'][split_type]['audio_name'][:]]
            # self.samples_weight = [classes_weight_dict[gt.decode('utf-8')] for gt in hf[f'regular_split'][split_type]['groundtruth'][:]]

        self.len_dataset = len(self.audio_names)

        self.gt_types = np.arange(1, 28)
        self.gt_str_to_onehot = {voice_type: [1 if i == idx else 0 for i in range(len(self.gt_types))] for idx, voice_type in enumerate(self.gt_types)}
        self.gt_list_to_str = {i: voice_type for i, voice_type in enumerate(self.gt_types)}


        #self.gt_list_to_str = {tuple(v): k for k, v in self.gt_str_to_list.items()}
        # self.enc = OneHotEncoder(handle_unknown='ignore')
        # self.enc.fit(np.array(self.gt_types).reshape(-1, 1))


    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        # hdf5_path = meta['hdf5_path']
        # index_in_hdf5 = meta['index_in_hdf5']
        # singer_id = self.singer_ids[0]

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_name =  hf[f'split{self.split_id}'][self.split_type]['audio_name'][idx].decode('utf-8')
            mel_spectrogram =  hf[f'split{self.split_id}'][self.split_type]['mel_spectrogram'][idx]
            groundtruth =  hf[f'split{self.split_id}'][self.split_type]['singer'][idx]
            # groundtruth = groundtruth.decode('utf-8')
            groundtruth = self.gt_str_to_onehot[groundtruth]
            # print('CCCCCC')
            # print(groundtruth)
            # groundtruth = self.enc.transform()
            # print('AAAAAAAA')
            # print(groundtruth)

        return idx, audio_name, torch.Tensor(mel_spectrogram), torch.Tensor(groundtruth)

    
    def __len__(self):
        return self.len_dataset
    
class TechniqueClassificationDataset(object):
    def __init__(self, hdf5_path, split_id, split_type='train'):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 
        """
        self.hdf5_path = hdf5_path
        self.split_type = split_type
        self.split_id = split_id
        
        classes = ['ClearVoice', 'DeathGrowl', 'HardcoreScream', 'BlackShriek']
        classes_dict = {
            'ClearVoice': 0,
            'DeathGrowl': 1,
            'HardcoreScream': 2,
            'BlackShriek': 3
        }

        with h5py.File(self.hdf5_path, 'r') as hf:
            gt = hf[f'split{split_id}'][split_type]['technique'][:]
            gt = np.array([item.decode('utf-8') for item in gt])
            class_sample_count = np.array(
                        [len(np.where(gt == t)[0]) for t in classes_dict.keys()])
            weight = 1. / class_sample_count

            #MT: just for testing
            self.samples_weight = np.array([1 for t in gt])
            # self.samples_weight = np.array([weight[classes_dict[t]] for t in gt])

            self.audio_names = [audio_name for audio_name in hf[f'split{split_id}'][split_type]['audio_name'][:]]

        self.len_dataset = len(self.audio_names)

        self.gt_types = ['ClearVoice', 'DeathGrowl', 'HardcoreScream', 'BlackShriek']
        self.gt_str_to_onehot = {voice_type: [1 if i == idx else 0 for i in range(len(self.gt_types))] for idx, voice_type in enumerate(self.gt_types)}
        self.gt_list_to_str = {i: voice_type for i, voice_type in enumerate(self.gt_types)}

    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_name =  hf[f'split{self.split_id}'][self.split_type]['audio_name'][idx].decode('utf-8')
            mel_spectrogram =  hf[f'split{self.split_id}'][self.split_type]['mel_spectrogram'][idx]
            groundtruth =  hf[f'split{self.split_id}'][self.split_type]['technique'][idx]
            groundtruth = groundtruth.decode('utf-8')
            groundtruth = self.gt_str_to_onehot[groundtruth]

        return idx, audio_name, torch.Tensor(mel_spectrogram), torch.Tensor(groundtruth)

    def __len__(self):
        return self.len_dataset
    

class BinaryTechniqueClassificationDataset(object):
    def __init__(self, hdf5_path, split_id, split_type='train'):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 
        """
        self.hdf5_path = hdf5_path
        self.split_type = split_type
        self.split_id = split_id
        
        classes = ['ClearVoice', 'DeathGrowl', 'HardcoreScream', 'BlackShriek']
        classes_dict = {
            'ClearVoice': 0,
            'DeathGrowl': 1,
            'HardcoreScream': 1,
            'BlackShriek': 1
        }

        with h5py.File(self.hdf5_path, 'r') as hf:
            gt = hf[f'split{split_id}'][split_type]['technique'][:]
            gt = np.array([item.decode('utf-8') for item in gt])
            class_sample_count = np.array(
                        [len(np.where(gt == t)[0]) for t in classes_dict.keys()])
            weight = 1. / class_sample_count

            #MT: just for testing
            self.samples_weight = np.array([1 for t in gt])
            # self.samples_weight = np.array([weight[classes_dict[t]] for t in gt])

            self.audio_names = [audio_name for audio_name in hf[f'split{split_id}'][split_type]['audio_name'][:]]

        self.len_dataset = len(self.audio_names)

        self.gt_types = ['ClearVoice', 'DeathGrowl', 'HardcoreScream', 'BlackShriek']
        self.gt_str_to_onehot = {voice_type: [1 if i == idx else 0 for i in range(len(self.gt_types))] for idx, voice_type in enumerate(self.gt_types)}
        self.gt_list_to_str = {i: voice_type for i, voice_type in enumerate(['ClearVoice', 'DistortedVoice'])}

    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_name =  hf[f'split{self.split_id}'][self.split_type]['audio_name'][idx].decode('utf-8')
            mel_spectrogram =  hf[f'split{self.split_id}'][self.split_type]['mel_spectrogram'][idx]
            groundtruth =  hf[f'split{self.split_id}'][self.split_type]['technique'][idx]
            groundtruth = groundtruth.decode('utf-8')
            groundtruth = self.gt_str_to_onehot[groundtruth]
            groundtruth = [groundtruth[0], int(any(groundtruth[1:]))]
        return idx, audio_name, torch.Tensor(mel_spectrogram), torch.Tensor(groundtruth)

    def __len__(self):
        return self.len_dataset