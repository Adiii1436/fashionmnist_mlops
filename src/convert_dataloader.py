from torch.utils.data import DataLoader 
import torch

def convert_to_dataloader(data, batch_size)->torch.utils.data.dataloader.DataLoader:
    dataloader = DataLoader(dataset=data,
                            batch_size=batch_size,
                            shuffle=True)
    return dataloader