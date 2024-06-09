import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset

class THUMOS(Dataset):

    def __init__(self, split, feat_stride, num_frames):
        
        self.feat_num_frames = num_frames
        self.feat_stride = feat_stride
        self.datafolder = "/home/sachishs/Documents/ActionFormer/DATA/thumos"
        self.jsondb_folder = os.path.join(self.datafolder, "annotations")
        self.ft_folder = os.path.join(self.datafolder, "i3d_features")
        self.json_file_path = os.path.join(self.jsondb_folder, "thumos14.json") 

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


    def process_segments(self, data_item):
        labels = []
        segments = []

        for seg in data_item['annotations']:

            labels.append(seg['label_id'])

            start = seg['segment(frames)'][0]
            end = seg['segment(frames)'][1]

            segments.append([self.convert(start), self.convert(end)])
        
        return labels, segments


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        data_item = self.data[index]
        data_file_name = self.datafilenames[index]
        fts_file = os.path.join(self.ft_folder, data_file_name+'.npy')
        fts = np.load(fts_file)

        labels, segments = self.process_segments(data_item)
        labels = torch.FloatTensor(labels)
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
        return datadict
    
