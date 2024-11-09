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

# Función para cargar un dataset desde un archivo CSV
@timeThis
def load_dataset(path):
    """
    Carga el dataset desde el archivo CSV especificado en el path.
    """
    return pd.read_csv(path)

# Función para dividir el dataset en conjuntos de entrenamiento y prueba
@timeThis
def train_test_division(df):
    """
    Divide el dataframe en conjuntos de características (X) y etiquetas (y),
    luego divide los datos en entrenamiento y prueba.
    """
    x = df.drop('Survived', axis = 1)  # Características (X)
    y = df.Survived  # Etiqueta (y)

    return train_test_split(x, y, test_size = 0.25, random_state = 123)

# Función para dividir el conjunto de entrenamiento en entrenamiento y validación
@timeThis
def validation_division(x_train, y_train):
    """
    Divide el conjunto de entrenamiento en entrenamiento y validación.
    """
    return train_test_split(x_train, y_train, test_size = 0.3, random_state = 123)

# Función para entrenar un modelo, hacer predicciones y calcular la precisión
def train_model(
    modelObj,
    modelName,
    hyperparameters,
    x_val_train,
    x_val_test,
    y_val_train,
    y_val_test):
    """
    Entrena el modelo con los hiperparámetros proporcionados, evalúa la precisión
    sobre el conjunto de validación y retorna los resultados.
    """
    start = time.time()

    model = modelObj(**hyperparameters)  # Crear el modelo con los hiperparámetros

    try:
        model.fit(x_val_train, y_val_train)  # Entrenar el modelo
        y_pred = model.predict(x_val_test)  # Hacer predicciones
        accuracy = accuracy_score(y_val_test, y_pred)  # Calcular la precisión

    except Exception:
        accuracy = np.nan  # Si hay error, asignar precisión NaN

    hyperparameters['accuracy'] = accuracy
    end = time.time() - start
    total_time = timedelta(seconds = end)  # Calcular el tiempo total de entrenamiento

    return (modelName, hyperparameters, end, accuracy)

# Función para ejecutar el entrenamiento de modelos con múltiples combinaciones de hiperparámetros
@timeThis
def execute(modelName, modelObj, x_val_train, y_val_train, x_val_test, y_val_test, max_combinations = None):
    """
    Ejecuta el entrenamiento de un modelo con múltiples combinaciones de hiperparámetros,
    usando múltiples hilos para acelerar el proceso.
    """
    if model in ['tree', 'randomForest', 'knn', 'logisticRegression']:
        start = time.time()
        results = []
        i = 0

        hyperparameters = shuffleList(modelObj.combinations)  # Barajar las combinaciones de hiperparámetros
        max_combinations = continueProgram(model = modelName, combinaciones = len(hyperparameters))

        if max_combinations is not None:
            hyperparameters = hyperparameters[:max_combinations]  # Limitar el número de combinaciones

        futures = []  # Lista para almacenar los resultados de los hilos

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for h in hyperparameters:
                partial_func = partial(train_model, modelObj.model, modelName, h, 
                                    x_val_train, y_val_train, x_val_test, y_val_test)  # Función parcial con los parámetros

                futures.append(executor.submit(partial_func))  # Enviar la tarea al pool de hilos

            for future in concurrent.futures.as_completed(futures):  # Recoger resultados cuando estén listos
                lap_time = time.time() - start
                i += 1
                results.append(future.result())  # Almacenar el resultado
                print(f'{timeFormat(lap_time)} | {i}/{len(hyperparameters)}')

        return results

# Función para descargar los resultados en un archivo CSV
@timeThis
def download_results(results, output):
    """
    Guarda los resultados de las combinaciones de hiperparámetros y precisión
    en un archivo CSV.
    """
    df_results = pd.DataFrame(results, columns = ['model', 'hyperparameters', 'time', 'accuracy'])
    path = f'C:/Documents/Proyectos/proyecto-titanic/datasets/results/{output}.csv'

    df_results.to_csv(path, index = False)  # Guardar los resultados en un CSV
    print(f'Terminado: {output}')


if __name__ == '__main__':
    # Cargar el dataset
    path_dataset = "C:/Documents/Proyectos/proyecto-titanic/datasets/train-limpio.csv"
    df = load_dataset(path_dataset)
    
    # Dividir el dataset en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_division(df)
    
    # Dividir el conjunto de entrenamiento en entrenamiento y validación
    x_val_train, x_val_test, y_val_train, y_val_test = validation_division(x_train, y_train)

    # Configurar el modelo a utilizar
    model, output = 'logisticRegression', 'lr-2-validation'
    modelName, modelObj = Mapper[model]['modelName'], Mapper[model]['modelSk']

    # Ejecutar el entrenamiento con validación cruzada
    results = execute(modelName, modelObj, x_val_train, x_val_test, y_val_train, y_val_test)

    # Descargar los resultados
    download_results(results, output)
