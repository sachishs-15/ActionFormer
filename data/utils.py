import torch
from data.data_utils import trivial_batch_collator, truncate_feats

def get_data_loader(dataset, batch_size, num_workers, generator = None):
    
    dataloader = torch.utils.data.DataLoader(

        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True, 
        collate_fn=trivial_batch_collator

    )

    return dataloader
