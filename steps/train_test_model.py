import torch
from zenml import step
from src.start_training_loop import start_training_loop
from src.FashionMNISTModel import FashionMNISTModel
from torch.utils.data import DataLoader
from zenml.client import Client
from typing import Annotated
from typing import Tuple

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_test_model(
        model_path:str, 
        model:FashionMNISTModel,
        train_dataloader: DataLoader, 
        test_dataloader: DataLoader, 
        hidden_units:int,
        classes:list,
        epochs:int = 1)->Tuple[
                Annotated[float, "train_loss"],
                Annotated[float, "train_acc"],
                Annotated[float, "test_loss"],
                Annotated[float, "test_acc"]
        ]:

        train_loss, train_acc, test_loss, test_acc = start_training_loop(
            epochs=epochs,
            model_path=model_path,
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            hidden_units=hidden_units,
            classes=classes
        )

        return train_loss, train_acc, test_loss, test_acc

   
        
        