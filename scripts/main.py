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
from utils import timeThis, timeFormat, shuffleList

@timeThis
def load_dataset():
    return pd.read_csv('C:/Documents/aprendizaje-automatico/datasets/winequality-white.csv', sep = ';')

@timeThis
def train_test_division(df):
    x = df.drop('quality', axis = 1)
    y = df.quality

    return train_test_split(x, y, test_size = 0.3, random_state = 123)
    
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

    return (modelName, hyperparameters, str(total_time), accuracy)

@timeThis
def execute(model, x_val_train, y_val_train, x_val_test, y_val_test, max_combinations = None):
    if model in ['tree', 'randomForest', 'knn', 'logisticRegresion']:
        start = time.time()
        results = []
        i = 0

        modelName = Mapper[model]['modelName']
        _modelSk = Mapper[model]['modelSk']
        hyperparameters = shuffleList(_modelSk.combinations)

        if max_combinations is not None:
            hyperparameters = hyperparameters[:max_combinations]

        futures = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for h in hyperparameters:
                partial_func = partial(train_model, _modelSk, modelName, h, 
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
    path = f'C:/Documents/aprendizaje-automatico/datasets/resultados-{output}.csv'

    df_results.to_csv(path, index = False)
    print(f'Terminado: {output}')


if __name__ == '__main__':

    df = load_dataset()
    x_train, x_test, y_train, y_test = train_test_division(df)

    x_val_train, x_val_test, y_val_train, y_val_test = validation_division(x_train, y_train)

    model_selection, output = 'knn', 'knn-validation'
    results = execute(model_selection, x_val_train, x_val_test, y_val_train, y_val_test,
                        max_combinations = 200)

    download_results(results, output)
    
    
    