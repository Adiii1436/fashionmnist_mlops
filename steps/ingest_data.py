from zenml import step
import logging
from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Annotated
from typing import Tuple
import torch

class InjestData:
    def __init__(self):
        pass
    
    def get_data(self):
        logging.info(f"Using torch version: {torch.__version__}")
        logging.info(f'Downloading data from pytorch server')
        
        train_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
            target_transform=None
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
            target_transform=None
        )

        logging.info(f'Downloaded data from pytorch server')

        return train_data,test_data


@step 
def ingest_df() -> Tuple[
    Annotated[datasets.mnist.FashionMNIST, "train_data"],
    Annotated[datasets.mnist.FashionMNIST, "test_data"],
    Annotated[list, "classes"]
]:
    try:
        ingest_data = InjestData()
        train_data, test_data = ingest_data.get_data()
        return train_data,test_data,train_data.classes
    except Exception as e:
        logging.error(f'Unable to download data from pytorch server')
        raise e
