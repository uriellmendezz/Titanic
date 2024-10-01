import time, random
from functools import wraps
from datetime import datetime, timedelta

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

