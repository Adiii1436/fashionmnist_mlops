import torch 
from src.FashionMNISTModel import FashionMNISTModel

def create_model(hidden_units:int, class_names:list):
    torch.manual_seed(42)

    model = FashionMNISTModel(input_shape=1,
                              hidden_units=hidden_units,
                              output_shape=len(class_names))

    model_path = "saved_model/FashionMNIST_Model.pth"
    torch.save(model.state_dict(), model_path)

    return model, model_path    