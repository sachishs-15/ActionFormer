import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset

from data.data_utils import truncate_feats

from pdb import set_trace as bp
#sleep
from time import sleep

class THUMOS(Dataset):

    def __init__(self, 
                 split, 
                 feat_stride, 
                 num_frames,
                 max_seq_len
                 ):
        
        self.feat_num_frames = num_frames
        self.feat_stride = feat_stride
        self.datafolder = "/home/sachishs/Documents/ActionFormer/DATA/thumos"
        self.jsondb_folder = os.path.join(self.datafolder, "annotations")
        self.ft_folder = os.path.join(self.datafolder, "i3d_features")
        self.json_file_path = os.path.join(self.jsondb_folder, "thumos14.json") 
        self.max_seq_len = max_seq_len

        with open(self.json_file_path) as file:
            self.data = json.load(file)

        print("Reading data ",  ".....")

        self.data = self.data["database"]
        # print(len(self.data))
        
        if(split == "train"):
            split = "Validation"   
        else:
            split = "Test"


        self.cdata = [] # all the test data 
        self.datafilenames = []

        for key in self.data.keys():
            if self.data[key]["subset"] == split:

                self.cdata.append(self.data[key])
                self.datafilenames.append(key)

        self.data = self.cdata


    def convert(self, frame):

        return (frame - self.feat_num_frames/2)/self.feat_stride

    def ft_to_time_convert(self, time): # explicit stated fps = 30

        return (time*self.feat_stride + self.feat_num_frames/2)/30

    def process_segments(self, data_item):

        labels = []
        segments = []

        for seg in data_item['annotations']:

            labels.append(seg['label_id'])

            start = seg['segment(frames)'][0]
            end = seg['segment(frames)'][1]

            segments.append([self.convert(start), self.convert(end)])
        
        return labels, segments
    

    def clip_video(self, data_dict):
        print("Clipping video")
        sleep(2)

        data_copy = {}

        data_copy['fps'] = data_dict['fps']
        data_copy['duration'] = data_dict['duration']
        data_copy['feat_stride'] = data_dict['feat_stride']
        data_copy['feat_num_frames'] = data_dict['feat_num_frames']

        data_copy['duration'] = self.ft_to_time_convert(self.max_seq_len)
        data_copy['feats'] = data_dict['feats'][:, :self.max_seq_len]

        #only keep the segments that are within the max_seq_len
        data_copy['segments'] = []
        data_copy['labels'] = []

        for i in range(len(data_dict['segments'])):
            seg = data_dict['segments'][i]
            if seg[0] > self.max_seq_len:
                continue

            if seg[1] > self.max_seq_len:
                #print("Segment ", seg)
                seg[1] = self.max_seq_len
                #print("Segment mod ", seg)


            data_copy['segments'].append(seg.numpy())
            data_copy['labels'].append(data_dict['labels'][i])

        data_copy['segments'] = torch.tensor(np.array(data_copy['segments']))
        data_copy['labels'] = torch.tensor(data_copy['labels'])

        return data_copy

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        data_item = self.data[index]
        data_file_name = self.datafilenames[index]
        fts_file = os.path.join(self.ft_folder, data_file_name+'.npy')
        fts = np.load(fts_file)
        fts = torch.from_numpy(fts.transpose())

        labels, segments = self.process_segments(data_item)
        labels = torch.tensor(labels)
        segments = torch.FloatTensor(segments)

        datadict = {

            "video_id" : data_file_name,
            'feats': fts,
            'labels': labels,
            'fps': data_item['fps'],
            'duration': data_item['duration'],
            'feat_stride': self.feat_stride,
            'feat_num_frames': self.feat_num_frames,
            'segments': segments

        }

        if fts.shape[1] > self.max_seq_len:
            datadict = self.clip_video(datadict)

        return datadict
    
