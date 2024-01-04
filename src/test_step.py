import torch 
import logging 
from steps.helper_functions import accuracy_fn
from src.FashionMNISTModel import FashionMNISTModel
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import mlflow

class testStep(ABC):
    
    @abstractmethod
    def test_step(
        self, 
        model_path:str, 
        hidden_units:int, 
        classes:list, 
        dataloader:DataLoader):
        pass 


class testData(testStep):

    def _load_model(self, model_path:str,hidden_units:int, classes:list):
        logging.info("Loading model for testing")  
        model = FashionMNISTModel(
            input_shape=1,
            hidden_units=hidden_units,
            output_shape=len(classes)
        )
        model.load_state_dict(torch.load(model_path))  # Load the state
        logging.info("Model loaded for testing")

        return model
    
    def _load_loss(self):
        logging.info("Initializing loss function")
        loss_fn = torch.nn.CrossEntropyLoss()

        return loss_fn
    
    def _testing_model(
            self,
            model,
            dataloader:DataLoader,
            loss_fn):
        
        test_loss, test_acc = 0,0
        
        model.eval()

        with torch.inference_mode():
          for X, y in dataloader:
            test_pred = model(X)

            loss = loss_fn(test_pred, y)
            test_acc +=  accuracy_fn(y, test_pred.argmax(dim=1))

            test_loss += loss.detach()

        test_loss_copy = test_loss.clone()
        test_loss_copy /= len(dataloader)
        test_acc /= len(dataloader)

        mlflow.log_metric("test_loss",test_loss_copy)
        mlflow.log_metric("test_acc",test_acc)

        return test_loss_copy, test_acc

    def test_step(
            self,
            model_path:str,
            hidden_units:int,
            classes:list,
            dataloader:DataLoader
        ):
        
        model = self._load_model(model_path,hidden_units,classes)

        loss_fn = self._load_loss()

        test_loss, test_acc = self._testing_model(model,dataloader,loss_fn)
        
        logging.info(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}\n")

        return float(test_loss), float(test_acc)