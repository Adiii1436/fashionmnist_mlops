import torch
from steps.batch_data import batch_df
from steps.helper_functions import accuracy_fn
from steps.ingest_data import ingest_df
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from torch.utils.data import DataLoader
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters

from steps.initialize_model import initialize_model
from steps.train_test_model import train_test_model
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations={MLFLOW})

@step(enable_cache=False)
def dynamic_importer() -> DataLoader:
    data = get_data_for_test()
    return data

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = 70.0

@step
def deployment_trigger(accuracy:float,config:DeploymentTriggerConfig)->bool:
    return accuracy>=config.min_accuracy    

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]


@step
def predictor(
    service: MLFlowDeploymentService,
    data: DataLoader,
) -> float:
    
    test_acc = 0
    
    with torch.inference_mode():
        for X, y in data:
            test_pred = service(X)
            test_acc +=  accuracy_fn(y, test_pred.argmax(dim=1))
        
    test_acc /= len(data)
    return test_acc
    
    
@pipeline(enable_cache=False, settings={"docker":docker_settings})
def continous_deployment_pipeline(
    min_accuracy: float = 0,
    workers: int = 3,
    timeout: int = 6000000,
):
    train_data, test_data, classes = ingest_df()
    train_dataloader, test_dataloader = batch_df(train_data,test_data)
    model,model_path = initialize_model(class_names=classes, hidden_units=10)

    _, train_acc, _, _ = train_test_model(
        model_path=model_path,
        model=model, 
        train_dataloader=train_dataloader, 
        test_dataloader=test_dataloader,
        hidden_units=10,
        classes=classes
    )

    deployment_decision = deployment_trigger(train_acc)

    mlflow_model_deployer_step(
        model=model_path,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
        mlserver=True
    )


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)



