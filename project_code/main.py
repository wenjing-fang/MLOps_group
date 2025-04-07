import functions as f
import models as models 

for model_name in ['logistic', 'rf','svm']:
    # hyper params
    # model_name = 'svm' # ['logistic', 'rf','svm']

    # data prep
    raw_data = f.read_dataset()
    data = f.preprocess_data(raw_data)
    X_train, X_test, y_train, y_test = f.split_data(data)


    # model selection, training, saving
    model = models.get_model(model_name)
    model.fit(X_train, y_train)
    f.save_model(model, model_name)

    # evaluation
    loaded_model = f.load_model(model_name)
    metrics = f.evaluate_model(loaded_model, X_test, y_test)  
    print("Model: ", model_name)
    print(f"F1 Score: {metrics[0]},Precision: {metrics[1]}")

# log experiment


