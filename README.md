# MLOps course project

## Goal of the project
The objective of this project is to come up with a MLOPS tool that include the whole process of a machine learning model from data ingestion, training to deployment. In this case the user could easily reuse the trained model and better handle the CI/CD process.  
## Structure

<pre lang="markdown"> ```plaintext D:. ├── .gitignore ├── Dockerfile.streamlit ├── EDA(do not submit this).ipynb ├── pyproject.toml ├── README.md ├── requirements.txt ├── __init__.py ├── checkpoints │ ├── logistic.pkl │ ├── rf.pkl │ └── svm.pkl ├── cli │ └── cli_tool.py ├── dataset │ └── bank.csv ├── project_code │ ├── functions.py │ ├── main.py │ ├── models.py │ ├── __init__.py │ └── __pycache__ │ ├── functions.cpython-311.pyc │ ├── models.cpython-311.pyc │ └── __init__.cpython-311.pyc └── streamlit └── app.py ``` </pre>
