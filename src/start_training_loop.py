from urllib.parse import urlparse
import torch 
from zenml import step 
import logging 
from src.train_step import trainData
from src.test_step import testData
from torch.utils.data import DataLoader
from src.FashionMNISTModel import FashionMNISTModel
import mlflow

def start_training_loop(
        epochs:int,
        model_path:str,
        model:FashionMNISTModel,
        hidden_units:int,
        classes:list,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader
):
    torch.manual_seed(42)
    logging.info("Starting training and testing loop")

    with mlflow.start_run(nested=True):
        for epoch in range(epochs):
            logging.info(f"Epoch: {epoch}\n-------")

            train_data = trainData()
            train_loss,train_acc=train_data.train_step(
                model_path=model_path,
                hidden_units=hidden_units,
                classes=classes,
                dataloader=train_dataloader
            )

            test_data = testData()
            test_loss, test_acc = test_data.test_step(
                model_path=model_path,
                hidden_units=hidden_units,
                classes=classes,
                dataloader=test_dataloader
            )

    mlflow.pytorch.log_model(model, "model", registered_model_name="CNN_MODEL", conda_env=mlflow.pytorch.get_default_conda_env())

    logging.info("Finished training and testing")

    return train_loss, train_acc, test_loss, test_acc