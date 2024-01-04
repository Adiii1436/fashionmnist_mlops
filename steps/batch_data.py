import logging
from zenml import step
from torchvision import datasets
from torch.utils.data import DataLoader
from typing import Annotated
from typing import Tuple
from src.convert_dataloader import convert_to_dataloader

@step 
def batch_df(
    train_data: datasets.mnist.FashionMNIST, 
    test_data: datasets.mnist.FashionMNIST,
    batch_size: int = 32
    ) -> Tuple[
        Annotated[DataLoader, "train_dataloader"],
        Annotated[DataLoader, "test_dataloader"]
    ]:
    try:
        train_dataloader = convert_to_dataloader(train_data,batch_size)

        test_dataloader = convert_to_dataloader(test_data,batch_size)
        
        logging.info("Transformed data into dataloaders")

        return train_dataloader,test_dataloader

    except Exception as e:
        logging.error(f'Unable to clean data')
        raise e
