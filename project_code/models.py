from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_model(name):
    if name == 'logistic':
        return LogisticRegression()
    elif name == 'rf':
        return RandomForestClassifier()
    elif name == 'svm':
        return SVC()
    else:
        raise ValueError(f"Unsupported model: {name}")
    
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

