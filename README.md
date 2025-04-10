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

```pip install -r requirements.txt ```

- `pyproject.toml` provides a more flexible and modern way to manage dependencies. We use **hatchling** as our build system. To install dependencies via `pyproject.toml`, run: ``` hatch install ```


## tests
The `tests/` folder includes unit tests written using Python's built-in `unittest` framework. These tests ensure the core machine learning logic and utility functions are working correctly.

- **`test_functions.py`**  
  Contains tests for data preprocessing and evaluation functions defined in `ml/functions.py`. It validates:
  - That target encoding is correctly applied (`deposit` column).
  - That user input is preprocessed without label leakage.
  - That data is correctly split into training and test sets.
  - That model evaluation metrics return expected data types (e.g., F1 score, precision).

- **`test_models.py`**  
  Verifies model selection and training routines in `ml/models.py`. Key aspects tested include:
  - Correct instantiation of models (`LogisticRegression`, `RandomForestClassifier`, `SVC`) via the `get_model()` function.
  - Error handling for unsupported model names.
  - Training validation via the presence of a `predict` method in trained models.

Each test uses small, simulated datasets to isolate logic and ensure reproducibility.

## checkpoints
This folder stores **trained models** (logistic, random forest, and SVM) as `.pkl` files using `pickle`.

## dataset
The data for machine learning modeling

## ml
This folder contains the core machine learning logic for the project, including data processing, model management, and evaluation utilities. It is structured to support a modular, testable, and reusable ML pipeline.
- `functions.py` provides functions for data ingestion, data preprocessing, data splitting, model evaluation and model saving/loading.
- `models.py` contains functions for ml models selection.

## FastApi
This folder contains a api basically using fastapi and univorn to create a api that offers two choice: list and predict. For list, the method is "get" and for predict the method is "get" and "post".  

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
