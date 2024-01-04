import logging
import pandas as pd
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_data_for_test()->DataLoader:
    try:
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
            target_transform=None
        )

        dataloader = DataLoader(dataset=test_data,
                            batch_size=32,
                            shuffle=True)
        
        return dataloader
    except Exception as e:
        logging.error(e)
        raise e
