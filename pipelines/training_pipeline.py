from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.batch_data import batch_df
from steps.initialize_model import initialize_model
from steps.train_test_model import train_test_model

@pipeline(enable_cache=False,)
def training_pipeline(run_id:str):
    
    train_data, test_data, classes = ingest_df()
    train_dataloader, test_dataloader = batch_df(train_data,test_data)

    model,model_path = initialize_model(class_names=classes, hidden_units=10)

    train_loss, train_acc, test_loss, test_acc = train_test_model(
        model_path=model_path, 
        model=model,
        train_dataloader=train_dataloader, 
        test_dataloader=test_dataloader,
        hidden_units=10,
        classes=classes
    )




    

