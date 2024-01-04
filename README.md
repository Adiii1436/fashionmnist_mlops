# COMPLETE MACHINE LEARNING PIPELINE: DEPLOYING A CNN MODEL WITH FASHIONMNIST DATASET

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

**Problem statement**: Crafting a personal MLOps project, this initiative tackles the challenge of organizing a machine learning workflow. Focused on the FashionMNIST dataset, the goal is to create a step-by-step process—from data handling to deploying a CNN model. The project aims to solve common issues in machine learning projects, like tangled code and tricky deployments. By structuring the project into clear stages, it simplifies the journey from idea to solution. Highlighting the importance of deploying a model, this project aims to showcase the practical impact of machine learning while providing a straightforward template for others diving into this field.

![FASHION_MNIST_DATASET](https://res.cloudinary.com/practicaldev/image/fetch/s---fNWEeWA--/c_imagga_scale,f_auto,fl_progressive,h_420,q_auto,w_1000/https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png)

## :snake: Python Requirements

Let's jump into the Python packages you need. Within the Python environment of your choice, run:

```bash
git clone https://github.com/zenml-io/zenml-projects.git
cd zenml-projects/customer-satisfaction
pip install -r requirements.txt
```

ZenML comes bundled with a dashboard. This dashboard allows you to observe your stacks, stack components and pipeline DAGs in a dashboard interface. To access this, you need to [launch the ZenML Server and Dashboard locally](https://docs.zenml.io/user-guide/starter-guide#explore-the-dashboard), but first you must install the optional dependencies for the ZenML server:

```bash
pip install zenml["server"]
zenml up
```

If you are running the `run_deployment.py` script, you will also need to install some integrations using ZenML:

```bash
zenml integration install mlflow -y
```

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. We'll be using Mlflow hosted in the [Dagshub](https://dagshub.com) Server. You need to configure Dagshub first and link your github repo to Dagshub. 

Configuring a new stack with the two components are as follows:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow --tracking_uri=<YOUR_TRACKING_URI> --tracking_username=<YOUR_USERNAME> --tracking_password=<YOUR_PASSWORD>
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow_deployer -e mlflow_experiment_tracker --set
```

## :thumbsup: The Solution

We're creating a system to analyze the images and predict its correct class. This system doesn't just train a model once; it continuously predicts and updates itself.

Our plan involves building a step-by-step process that can be used on the cloud, adjusting its size based on our needs. This process tracks all the important data and stages of the prediction, from raw data to the final results.

To make this happen, we're using ZenML, a tool that makes building this process easy yet powerful. One key part is the integration with MLflow, which helps us keep track of metrics and parameters and deploy our machine learning model.

### Training Pipeline

Our standard training pipeline consists of several steps:

- `ingest_data`: This step will ingest the data from torchvision library.
- `batch_data`: This step will convert datasets into dataloaders.
- `initialize_model`: This step will create a custom CNN model.
- `train_model`: This step will train and test the model and save the model using [MLflow autologging](https://www.mlflow.org/docs/latest/tracking.html).

### Deployment Pipeline

We've got another pipeline called deployment_pipeline.py, which builds on the training pipeline. This one handles a continuous deployment process. It takes in and processes input data, trains a model, and then sets up or updates the prediction server that delivers the model's predictions if it passes our evaluation criteria.

Now, to decide if the model is good enough, we've set a rule based on something called [CROSS ENTROPY LOSS](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) during training. If the loss meets a certain configurable threshold, we consider it good to go. The first four steps of this pipeline are the same as the training one, but we've added these extra steps to make sure our model is up to snuff before deploying it:

- `deployment_trigger`: The step checks whether the newly trained model meets the criteria set for deployment.
- `model_deployer`: This step deploys the model as a service using MLflow (if deployment criteria is met).

![training_and_deployment_pipeline](https://assets-global.website-files.com/65264f6bf54e751c3a776db1/6530058b791c6b6f8b260ed3_continuous-deployment.gif)

## :notebook: Diving into the code

You can run two pipelines as follows:

- Training pipeline:

```bash
python run_pipeline.py
```

- The continuous deployment pipeline:

```bash
python run_deployment.py
```