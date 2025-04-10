# MLOps course project

## Goal of the project
The objective of this project is to come up with a MLOPS tool that include the whole process of a machine learning model from data ingestion, training to deployment. In this case the user could easily reuse the trained model and better handle the CI/CD process.  

## Structure

<pre> ``` 
.
├── Dockerfile.streamlit
├── README.md
├── checkpoints
│   ├── logistic.pkl
│   ├── rf.pkl
│   └── svm.pkl
├── cli
│   └── cli_tool.py
├── dataset
│   └── bank.csv
├── ml
│   ├── __init__.py
│   ├── functions.py
│   └── models.py
├── pyproject.toml
├── requirements.txt
├── streamlit
│   └── app.py
└── tests
    ├── __init__.py
    ├── test_functions.py
    └── test_models.py
``` </pre>
## Dependencies
Make sure your current working directory is inside the folder `MLOps_group`.

This project includes both `requirements.txt` and `pyproject.toml`.

- `requirements.txt` is used mainly for deployment of the app (streamlit/app.py) and containerization using Docker.After creating your virtual environment, run:

```bash
pip install -r requirements.txt 
```

- `pyproject.toml` provides a more flexible and modern way to manage dependencies. We use **hatchling** as our build system. To install dependencies via `pyproject.toml`, run: 
``` bash
hatch install 
```


## Folder Description
### dataset
The data for machine learning modeling.

### ml
The `ml/` folder contains the core machine learning logic for the project, including data processing, model management, and evaluation utilities. It is structured to support a modular, testable, and reusable ML pipeline.
- **`functions.py`** provides functions for data ingestion, data preprocessing, data splitting, model evaluation and model saving/loading.
- **`models.py`** contains functions for ml models selection.

### checkpoints
The `checkpoints/` folder stores **trained models** (logistic, random forest, and SVM) as `.pkl` files using `pickle`.

### streamlit
A user-friendly streamlit-based app to interact with the models. To launch the app, run:
``` bash
streamlit run streamlit/app.py 
```

### FastApi
The `FastApi/` folder contains a api basically using fastapi and univorn to create a api that offers two choice: list and predict. For list, the method is "get" and for predict the method is "get" and "post".  

### CLI
The CLI tool supports two commands: 
1. **List available models:** 
```bash 
python cli/cli_tool.py list 
```
2. **Train or evaluate a model:**
```bash 
python cli/cli_tool.py predict --model [model_name] --mode [train|eval] 
```
  - `--model`: Choose a model (e.g., `logistic`, `rf`, or `svm`)
  - `--mode`:  `train`: Trains the model. You'll see: `"Your model {modelname} is saved successfully."`; `eval`: Evaluates the model. A performance score will be returned. 

### tests
The `tests/` folder includes unit tests written using Python's built-in `unittest` framework. These tests ensure the core machine learning logic and utility functions are working correctly.
- **`test_functions.py`** contains tests for data preprocessing and evaluation functions defined in `ml/functions.py`. It validates:
  - That user input is preprocessed without label leakage.
  - That data is correctly split into training and test sets.
  - That model evaluation metrics return expected data types (e.g., F1 score, precision).
- **`test_models.py`** verifies model selection and training routines in `ml/models.py`. Key aspects tested include:
  - Correct instantiation of models (`LogisticRegression`, `RandomForestClassifier`, `SVC`) via the `get_model()` function.
  - Error handling for unsupported model names.
- **Usage**: to run all unit tests:
  ```bash
   python -m unittest discover -s tests
   ```
  
**Note:** The Dockerfile is located one level above this folder, because Docker builds only from the current context,but the app requires access to files in other folders.
