"""Utility functions."""

import os
import h5py
import numpy as np

import torch
from torch.utils import data
from torch import nn

import matplotlib.pyplot as plt

EPS = 1e-17


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        #nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def save_dict_h5py(array_dict, fname):
    """Save dictionary containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for key in array_dict.keys():
            hf.create_dataset(key, data=array_dict[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            array_dict[key] = hf[key][:]
    return array_dict


def save_list_dict_h5py(array_dict, fname):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for i in range(len(array_dict)):
            grp = hf.create_group(str(i))
            for key in array_dict[i].keys():
                grp.create_dataset(key, data=array_dict[i][key])


def load_list_dict_h5py2(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                '''if 'obs' in key:
                    #print('hf[grp][key]', hf[grp][key].shape)
                    array_dict[i][key] = hf[grp][key][:,:3,:,:]
                    #print('hf[grp][key]--', array_dict[i][key].shape)
                else:'''
                array_dict[i][key] = hf[grp][key][:]
                if i == 0:
                    print('key', np.array(array_dict[i][key]).shape)
    print('len', array_dict[0]['obs'].shape)
    print('len(array_dict)', len(array_dict))
    return array_dict
ave_num = 2 #2
def load_list_dict_h5py1(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    #ave_num = 1
    
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            #array_dict.append(dict())
            #print(hf[grp]['obs'][:].shape)
            part_num = int(hf[grp]['obs'][:].shape[0]/ave_num)
            for k in range(part_num):
                array_dict.append(dict())
                for key in hf[grp].keys():
                    array_dict[i*part_num+k][key] = hf[grp][key][k*ave_num:(k+1)*ave_num]
                if i == 0 and k <5:
                    print('key', np.array(array_dict[i][key]).shape)
    print('len', array_dict[0]['obs'].shape)
    print('len(array_dict)', len(array_dict))
    return array_dict


def load_list_dict_h5py3(fname, seq_len):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    #ave_num = 1
    
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            #array_dict.append(dict())
            #print(hf[grp]['obs'][:].shape)
            part_num = int(hf[grp]['obs'][:].shape[0]/seq_len)
            for k in range(part_num):
                array_dict.append(dict())
                for key in hf[grp].keys():
                    array_dict[i*part_num+k][key] = hf[grp][key][k*seq_len:(k+1)*seq_len]
                if i == 0 and k <5:
                    print('key', np.array(array_dict[i][key]).shape)
    print('len', array_dict[0]['obs'].shape)
    print('len(array_dict)', len(array_dict))
    return array_dict

def load_list_dict_h5py4(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    #ave_num = 1
    
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            #array_dict.append(dict())
            #print(hf[grp]['obs'][:].shape)
            #part_num = int(hf[grp]['obs'][:].shape[0]/seq_len)
            for k in range(hf[grp]['obs'][:].shape[0]):
                array_dict.append(dict())
                for key in hf[grp].keys():
                    array_dict[i*hf[grp]['obs'][:].shape[0]+k][key] = hf[grp][key][k]
                
    print('len--', array_dict[0]['obs'].shape)
    print('len(array_dict)--', len(array_dict))
    return array_dict



def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                if 'obs' in key:
                    array_dict[i][key] = hf[grp][key][:, :3, :, :]
                else:
                    array_dict[i][key] = hf[grp][key][:]
    print('len(array_dict)', len(array_dict), array_dict[0]['obs'].shape)
    return array_dict


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


def pairwise_distance_matrix(x, y):
    num_samples = x.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1).expand(num_samples, num_samples, dim)
    y = y.unsqueeze(0).expand(num_samples, num_samples, dim)

    return torch.pow(x - y, 2).sum(2)


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    elif act_fn == 'softmax':
        return nn.Softmax()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def to_float(np_array):
    """Convert numpy array to float32."""
    return np.array(np_array, dtype=np.float32)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


class StateTransitionsDataseth5seq(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py1(hdf5_file)
        #self.experience_buffer = load_list_dict_h5py_txt1(hdf5_file)

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        '''step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step'''
        self.num_steps = len(self.experience_buffer)

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        #ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[idx]['obs'])
        action = self.experience_buffer[idx]['action']
        next_obs = to_float(self.experience_buffer[idx]['next_obs'])

        return obs, action, next_obs

import random
class StateTransitionsDataseth5seq1(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file, seq_len):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py3(hdf5_file, seq_len)
        #self.experience_buffer_neg = load_list_dict_h5py4(hdf5_file)
        self.idx2episode = list()
        for i in range(len(self.experience_buffer)):
            for k in range(seq_len):
                self.idx2episode.append((i, k))
        random.shuffle(self.idx2episode)
        #self.experience_buffer = load_list_dict_h5py_txt1(hdf5_file)

        # Build table for conversion between linear idx -> episode/step idx
        # self.idx2episode = list()
        '''step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step'''
        self.num_steps = len(self.experience_buffer)
        self.seq_len = seq_len

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        #ep, step = self.idx2episode[idx]
        #print(len(self.experience_buffer), idx, self.seq_len)
        obs_neg1 = []
        action_neg1 = []
        next_obs_neg1 = []
        for i in range(self.seq_len):
            ep, step = self.idx2episode[idx+i]
            obs_neg1.append(self.experience_buffer[ep]['obs'][step])
            action_neg1.append(self.experience_buffer[ep]['action'][step])
            next_obs_neg1.append(self.experience_buffer[ep]['next_obs'][step])
        #print(len(obs_neg1), obs_neg1[0].shape)
        obs_neg = to_float(obs_neg1)
        action_neg = to_float(action_neg1)
        next_obs_neg = to_float(next_obs_neg1)
        #print(obs_neg.shape())
        #for i in range(self.seq_len):
            
        obs = to_float(self.experience_buffer[idx]['obs'])
        action = self.experience_buffer[idx]['action']
        next_obs = to_float(self.experience_buffer[idx]['next_obs'])

        return obs, action, next_obs, obs_neg, action_neg, next_obs_neg




class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        action = self.experience_buffer[ep]['action'][step]
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])

        return obs, action, next_obs



class PathDataset(data.Dataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """

    def __init__(self, hdf5_file, path_length=5):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        self.path_length = path_length

    def __len__(self):
        return len(self.experience_buffer)

    def __getitem__(self, idx):
        observations = []
        actions = []
        for i in range(self.path_length):
            obs = to_float(self.experience_buffer[idx]['obs'][i])
            action = self.experience_buffer[idx]['action'][i]
            observations.append(obs)
            actions.append(action)
        obs = to_float(
            self.experience_buffer[idx]['next_obs'][self.path_length - 1])
        observations.append(obs)
        return observations, actions



class PathDataseth5seq(data.Dataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """

    def __init__(self, hdf5_file, path_length=5):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py1(hdf5_file)
        #self.experience_buffer = load_list_dict_h5py_txtseq(hdf5_file)
        #self.experience_buffer = self.experience_buffer[:1000]
        self.path_length = path_length

    def __len__(self):
        return len(self.experience_buffer)

    def __getitem__(self, idx):
        observations = []
        actions = []
        #print('self.path_length', self.path_length)
        #for i in range(self.path_length):
            #print('idx', idx, i, len(self.experience_buffer))
        idx = int(idx*10/ave_num)
        obs = to_float(self.experience_buffer[idx]['obs'])
        action = self.experience_buffer[idx]['action']
        observations.append(obs)
        actions.append(action)
        obs = to_float(
            self.experience_buffer[idx]['next_obs'])
        observations.append(obs)
        return observations, actions

class PathDataseth5seq1(data.Dataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """

    def __init__(self, hdf5_file, seq_len, path_length=5):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py3(hdf5_file, seq_len)
        #self.experience_buffer = load_list_dict_h5py_txtseq(hdf5_file)
        #self.experience_buffer = self.experience_buffer[:1000]
        print(len(self.experience_buffer), '--------')
        self.path_length = path_length
        self.seq_len = seq_len

    def __len__(self):
        return len(self.experience_buffer)

    def __getitem__(self, idx):
        observations = []
        actions = []
        #print('self.path_length', self.path_length)
        #for i in range(self.path_length):
            #print('idx', idx, i, len(self.experience_buffer))
        #print(idx)
        #idx = int(idx*10/ave_num)
        #print(len(self.experience_buffer), idx, ave_num)
        obs = to_float(self.experience_buffer[idx]['obs'])
        action = self.experience_buffer[idx]['action']
        observations.append(obs)
        actions.append(action)
        obs = to_float(
            self.experience_buffer[idx]['next_obs'])
        observations.append(obs)
        return observations, actions
