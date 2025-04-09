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
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── __init__.cpython-313.pyc
│   │   ├── functions.cpython-312.pyc
│   │   ├── functions.cpython-313.pyc
│   │   └── models.cpython-312.pyc
│   ├── functions.py
│   └── models.py
├── pyproject.toml
├── requirements.txt
├── streamlit
│   └── app.py
└── tests
    ├── __init__.py
    ├── __pycache__
    │   ├── test_cli_tool.cpython-312.pyc
    │   ├── test_functions.cpython-312.pyc
    │   └── test_models.cpython-312.pyc
    ├── test_functions.py
    └── test_models.py
``` </pre>

## Dependencies
Make sure your current working directory is inside the folder `MLOps_group`.

This project includes both `requirements.txt` and `pyproject.toml`.

- `requirements.txt` is used mainly for deployment of the app (streamlit/app.py) and containerization using Docker.After creating your virtual environment, run:

```pip install -r requirements.txt ```

- `pyproject.toml` provides a more flexible and modern way to manage dependencies. We use **hatchling** as our build system. To install dependencies via `pyproject.toml`, run: ``` hatch install ```

## checkpoints
This folder stores **trained models** (logistic, random forest, and SVM) as `.pkl` files using `pickle`.

## dataset
The data for machine learning modeling

## project_code
This folder contrains all principle code for machine learning modeling. It also serves as a module that could be imported by other scripts

## streamlit
A user-friendly streamlit-based app to interact with the models. To launch the app, run:
``` streamlit run streamlit/app.py ```

**Note:** The Dockerfile is located one level above this folder, because Docker builds only from the current context,but the app requires access to files in other folders.

## CLI
The CLI tool supports two commands: 
1. **List available models:** ```bash python cli/cli_tool.py list ```
2. **Train or evaluate a model:**
```bash python cli/cli_tool.py predict --model [model_name] --mode [train|eval] ```
  - `--model`: Choose a model (e.g., `logistic`, `rf`, or `svm`)
  - `--mode`:  `train`: Trains the model. You'll see: `"Your model {modelname} is saved successfully."`; `eval`: Evaluates the model. A performance score will be returned. 
