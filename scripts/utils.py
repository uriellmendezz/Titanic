import time, random, os
from functools import wraps
from datetime import datetime, timedelta
from sklearn.model_selection import GridSearchCV, train_test_split

def timeThis(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs) 
        end = time.time()
        total = end - start

        print(f"{function.__name__}: {timeFormat(total)}\n")
        return result
    return wrapper

def timeFormat(seconds):
    total = datetime(1, 1, 1) + timedelta(seconds = seconds)
    return total.strftime("%H:%M:%S") + f".{int(total.microsecond / 1000):03d}"

def shuffleList(_list):
    new = _list.copy()
    random.shuffle(new)
    return new

def GridSearch(model, parameters, x_train, y_train, cv = 3, verbose = 1, jobs = -1):
    grid = GridSearchCV(
        estimator = model,
        param_grid = parameters,
        scoring = 'accuracy',
        cv = cv,
        verbose = verbose,
        n_jobs = jobs)
    
    grid.fit(x_train, y_train)
    return grid, grid.cv_results_

def continueProgram(model, combinaciones):
    while True:
        os.system('cls')
        print('Se encontraron {} combinaciones distinitas para el {}. ¿Quieres continuar?'.format(combinaciones, model))
        user_choice = input('>>> (s/n): ')
        if user_choice == "s":
            while True:
                os.system('cls')
                print('Combincaciones actuales: {}'.format(combinaciones))
                limit = input('Introduce el limite de combinaciones: ')
                if int(limit) <= int(combinaciones):
                    combinaciones = limit
                    print('Combincaciones actuales: {}'.format(combinaciones))
                    return int(combinaciones)
                
        elif user_choice == "n":
            return int(combinaciones)
        else:
            continue