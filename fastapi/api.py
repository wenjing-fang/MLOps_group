from fastapi import FastAPI,HTTPException
import uvicorn #web server
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
## add the path to make sure python could identify the modules

import ml.functions as f
import ml.models as models 
from valide_user_request import valide_request

def get_model_names():
        filenames=os.listdir('./checkpoints')
        modelnames=[f[:f.index('.')] for f in filenames]
        return modelnames

def predict(model_name,mode):
    # input data
    raw_data = f.read_dataset()
    data = f.preprocess_data(raw_data)
    X_train, X_test, y_train, y_test = f.split_data(data)
    model = models.get_model(model_name)
    if mode=='train':
        model.fit(X_train, y_train)
        f.save_model(model, model_name)
        return {"status": "success","message": f"The model '{model_name}' has been successfully trained and saved in the 'checkpoints' folder."}
    elif mode=='eval':
        loaded_model = f.load_model(model_name)
        metrics = f.evaluate_model(loaded_model, X_test, y_test)  
        return {"status": "success",
                "message": f"Model: {model_name}, F1 Score: {metrics[0]}, Precision: {metrics[1]}"}


app=FastAPI()

## access app using root 
@app.get("/")
#use asyncronous structure
async def root():
    return {"message":"Bonjour! Here is MLOPS Final Project"}


@app.get("/models/")
async def list_models():
        return get_model_names()

@app.api_route("/predict",methods=["GET","POST"])
async def predict(model_name: str, mode: str):
     #use valide_request to valide the input type
    try:
        result=await predict(model_name,mode)
        return result
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))




if __name__=='__main__':
    uvicorn.run(app="api:app",host="127.0.0.1",port=8080,reload=True)

