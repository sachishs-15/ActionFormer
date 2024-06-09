from data.datafile import THUMOS
from data.utils import get_data_loader

dataset = THUMOS(split = "train")

dataloader = get_data_loader(dataset, batch_size=1,  num_workers=1)

for i, data in enumerate(dataloader):
    
