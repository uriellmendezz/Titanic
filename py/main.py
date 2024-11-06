from models import ClassifierModel, Mapper
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import time
from datetime import timedelta, datetime
import concurrent.futures
import numpy as np
from functools import partial
from utils import timeThis, timeFormat, shuffleList, continueProgram

@timeThis
def load_dataset(path):
    return pd.read_csv(path)

@timeThis
def train_test_division(df):
    x = df.drop('Survived', axis = 1)
    y = df.Survived

    return train_test_split(x, y, test_size = 0.25, random_state = 123)
    
@timeThis
def validation_division(x_train, y_train):
    return train_test_split(x_train, y_train, test_size = 0.3, random_state = 123)

def train_model(
    modelObj,
    modelName,
    hyperparameters,
    x_val_train,
    x_val_test,
    y_val_train,
    y_val_test):

    start = time.time()

    model = modelObj(**hyperparameters)

    try:
        model.fit(x_val_train, y_val_train)
        y_pred = model.predict(x_val_test)
        accuracy = accuracy_score(y_val_test, y_pred)

    except Exception:
        accuracy = np.nan

    hyperparameters['accuracy'] = accuracy
    end = time.time() - start
    total_time = timedelta(seconds = end)

    return (modelName, hyperparameters, end, accuracy)

@timeThis
def execute(modelName, modelObj, x_val_train, y_val_train, x_val_test, y_val_test, max_combinations = None):
    if model in ['tree', 'randomForest', 'knn', 'logisticRegression']:
        start = time.time()
        results = []
        i = 0

        hyperparameters = shuffleList(modelObj.combinations)
        max_combinations = continueProgram(model = modelName, combinaciones = len(hyperparameters))

        if max_combinations is not None:
            hyperparameters = hyperparameters[:max_combinations]

        futures = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for h in hyperparameters:
                partial_func = partial(train_model, modelObj.model, modelName, h, 
                                    x_val_train, y_val_train, x_val_test, y_val_test)
    
                futures.append(executor.submit(partial_func))

            for future in concurrent.futures.as_completed(futures):
                lap_time = time.time() - start
                i += 1
                results.append(future.result())
                print(f'{timeFormat(lap_time)} | {i}/{len(hyperparameters)}')

        return results

@timeThis
def download_results(results, output):
    df_results = pd.DataFrame(results, columns = ['model', 'hyperparameters', 'time', 'accuracy'])
    path = f'C:/Documents/Proyectos/proyecto-titanic/datasets/results/{output}.csv'

    df_results.to_csv(path, index = False)
    print(f'Terminado: {output}')


if __name__ == '__main__':

    path_dataset = "C:/Documents/Proyectos/proyecto-titanic/datasets/train-limpio.csv"
    df = load_dataset(path_dataset)
    x_train, x_test, y_train, y_test = train_test_division(df)
    x_val_train, x_val_test, y_val_train, y_val_test = validation_division(x_train, y_train)

    model, output = 'logisticRegression', 'lr-2-validation'
    modelName, modelObj = Mapper[model]['modelName'], Mapper[model]['modelSk']

    results = execute(modelName, modelObj, x_val_train, x_val_test, y_val_train, y_val_test)

    download_results(results, output)
    
    
    