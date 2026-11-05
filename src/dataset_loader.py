import torch
import torch.utils.data as data
import os
import numpy as np
import utils

class UCF_crime(data.Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.samples = []

        # read list file
        split_path = os.path.join('list','UCF_{}.list'.format(self.mode))
        with open(split_path, 'r') as f:
            vid_list = [line.strip().split()[0] for line in f]

        # assign label
        for vid_info in vid_list:
            label = 0 if "Normal" in vid_info.split("/")[-1] else 1
            self.samples.append((vid_info, label))

    def __len__(self):
        # multiply by 10 to treat each crop as separate
        return len(self.samples) * 10

    def __getitem__(self, index):
        video_idx = index // 10   # which video
        crop_idx = index % 10     # which crop in that video

        vid_info, label = self.samples[video_idx]
        name = vid_info.split("/")[-1].split("_x264")[0]

        # load full [T,10,2048] feature
        video_feature = np.load(vid_info).astype(np.float32)  # shape [T,10,D]
        video_feature = video_feature[:, crop_idx, :]          # select single crop -> [T,D]

        # training: temporal averaging to num_segments
        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1]), dtype=np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype=np.int32)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i+1], :], axis=0)
                else:
                    new_feat[i, :] = video_feature[r[i], :]
            video_feature = new_feat

        if self.mode == "Test":
            return video_feature, label, f"{name}__{crop_idx}"
        else:
            return video_feature, label
 

class XDVideo(data.DataLoader):
    def __init__(self, root_dir, mode, modal, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path=root_dir
        self.mode=mode
        self.modal=modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        if self.modal == 'all':
            self.feature_path = []
            if self.mode == "Train":
                for _modal in ['RGB', 'Flow']:
                    self.feature_path.append(os.path.join(self.data_path, "i3d-features",_modal))
            else:
                for _modal in ['RGBTest', 'FlowTest']:
                    self.feature_path.append(os.path.join(self.data_path, "i3d-features",_modal))
        else:
            self.feature_path = os.path.join(self.data_path, modal)
        split_path = os.path.join("list",'XD_{}.list'.format(self.mode))
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[9525:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:9525]
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list=[]
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data,label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        label=0
        if "_label_A" not in vid_name:
            label=1  
        video_feature = np.load(os.path.join(self.feature_path[0],
                                vid_name )).astype(np.float32)
        if self.mode == "Train":
            new_feature = np.zeros((self.num_segments,self.len_feature)).astype(np.float32)
            sample_index = utils.random_perturb(video_feature.shape[0],self.num_segments)
            for i in range(len(sample_index)-1):
                if sample_index[i] == sample_index[i+1]:
                    new_feature[i,:] = video_feature[sample_index[i],:]
                else:
                    new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)
                    
            video_feature = new_feature
        return video_feature, label    
