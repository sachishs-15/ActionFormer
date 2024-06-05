import torch

def get_data_loader(dataset, batch_size, num_workers, generator = None):
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return dataloader
