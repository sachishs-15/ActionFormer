import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset

class THUMOS(Dataset):

    def __init__(self, split):
        self.datafolder = "data/thumos"
        self.jsondb_folder = os.path.join(self.datafolder, "annotations")
        self.ft_folder = os.path.join(self.datafolder, "i3d_features")
        self.json_file_path = os.path.join(self.jsondb_folder, "thumos14.json") 

        with open(self.json_file_path) as file:
            self.data = json.load(file)

        print("Reading ", self.data["version"], ".....")

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

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        data_item = self.data[index]

        video_id = data_item["video_id"]
        video_duration = data_item["duration"]
        video_fps = data_item["fps"]
        video_subset = data_item["subset"]
        video_annotations = data_item["annotations"]

        


        return
    
