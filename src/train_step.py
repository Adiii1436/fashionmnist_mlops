import torch 
import logging 
from steps.helper_functions import accuracy_fn
from src.FashionMNISTModel import FashionMNISTModel
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import mlflow

class trainStep(ABC):

    @abstractmethod
    def train_step(
        self, 
        model_path:str, 
        hidden_units:int, 
        classes:list, 
        dataloader:DataLoader):
        pass 


class trainData(trainStep):

    def _load_model(self, model_path:str,hidden_units:int, classes:list):
        logging.info("Loading model for training")  
        model = FashionMNISTModel(
            input_shape=1,
            hidden_units=hidden_units,
            output_shape=len(classes)
        )

        # mlflow.pytorch.autolog()
        model.load_state_dict(torch.load(model_path))  # Load the state
        logging.info("Model loaded for training")

        return model
    
    def _load_loss_optim(self,model):
        lr=0.1
        logging.info("Initializing loss function and optimizer")
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=model.parameters(),lr=lr)

        mlflow.log_param("learning_rate",lr)

        return loss_fn,optimizer
    
    def _training_loop(
            self,
            model,
            dataloader:DataLoader,
            loss_fn,
            optimizer):

        logging.info("Starting training")

        train_loss, train_acc = 0,0

        model.train()

        for batch, (X,y) in enumerate(dataloader):
            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            train_loss+=loss
            train_acc+=accuracy_fn(y,y_pred.argmax(dim=1))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if batch%400 == 0:
                logging.info(f"Looked at {batch * len(X)}/{len(dataloader.dataset)} samples")
        
        train_loss/=len(dataloader)
        train_acc/=len(dataloader)

        logging.info("Finished training")
        mlflow.log_metric("train_loss",train_loss)
        mlflow.log_metric("train_acc",train_acc)

        return train_loss, train_acc

    def train_step(
            self,
            model_path:str,
            hidden_units:int,
            classes:list,
            dataloader:DataLoader
        ):

        mlflow.log_param("hidden_units",hidden_units)
        
        model = self._load_model(model_path,hidden_units,classes)

        loss_fn,optimizer = self._load_loss_optim(model)

        train_loss, train_acc = self._training_loop(model,dataloader,loss_fn,optimizer)

        logging.info(f"\nTrain loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

        return float(train_loss), float(train_acc)