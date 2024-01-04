from zenml import step
import logging
from src.create_model import create_model
from src.FashionMNISTModel import FashionMNISTModel
from typing import Tuple, Annotated

@step
def initialize_model(
    class_names:list,
    hidden_units:int = 10) -> Tuple[
        Annotated[FashionMNISTModel, "model"],
        Annotated[str, "model_path"],
    ]:

    logging.info("Initializing model")

    model, model_path = create_model(hidden_units, class_names)

    logging.info(f"Model created and saved to: {model_path}")

    return model,model_path
