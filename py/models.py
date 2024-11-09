import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

'''
Este código define una clase ClassifierModel con modelos de clasificación como árboles de decisión, random forests, KNN y regresión logística.
Para cada modelo, creo combinaciones de hiperparámetros que usaré para buscar la mejor configuración.
Luego, el diccionario Mapper me ayuda a asociar cada modelo con su configuración, nombre y salida,
lo que me permite entrenarlos de manera eficiente con diferentes combinaciones de parámetros.
'''

class ClassifierModel:
    class tree:
        def __init__(self,
                    MAX_max_depth = 60):
            
            self.criterion = ['gini', 'entropy', 'log_loss']
            self.max_depth = list(range(2, MAX_max_depth))
            self.max_depth.append(None)
            self.model = DecisionTreeClassifier

            self.grid = {
                'criterion': self.criterion,
                'max_depth': self.max_depth
            }
            
            self.combinations = [
                {'criterion': c, 'max_depth': m}
                for c in self.criterion
                for m in self.max_depth
            ]
            

    class randomForestClassifier:
        def __init__(self, 
                    MAX_n_estimators = 310,
                    MAX_max_depth = 20,
                    MAX_min_samples_split = 5,
                    MAX_min_samples_leaf = 3):
            
            self.n_estimators = list(range(20, MAX_n_estimators, 10))
            self.criterion = ['gini', 'entropy']
            self.max_depth = list(range(2, MAX_max_depth))
            self.min_samples_split = list(range(2, MAX_min_samples_split))
            self.min_samples_leaf = list(range(2, MAX_min_samples_leaf))
            self.model = RandomForestClassifier

            self.grid = {
                'n_estimators': self.n_estimators,
                'criterion': self.criterion,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf
            }
            
            self.combinations = [
                {
                    'n_estimators': n_estimators,
                    'criterion': criterion,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf
                }
                for n_estimators in self.n_estimators
                for criterion in self.criterion
                for max_depth in self.max_depth
                for min_samples_split in self.min_samples_split
                for min_samples_leaf in self.min_samples_leaf
                if min_samples_split > min_samples_leaf
            ]

    class knn:
        def __init__(self,
                    MAX_n_neighbors=20,
                    MAX_leaf_size=35):
            
            self.n_neighbors = list(range(1, MAX_n_neighbors))
            self.weights = ['uniform', 'distance']
            self.algorithm = ['ball_tree', 'kd_tree', 'brute']
            self.leaf_size = list(range(10, MAX_leaf_size))
            self.p = [1, 2, 3]  # 1 = manhattan_distance (l1), 2 = euclidean_distance (l2), 3 = minkowski_distance (l_p)
            self.metric = ['euclidean', 'manhattan', 'minkowski']
            self.model = KNeighborsClassifier

            self.grid = {
                'n_neighbors': self.n_neighbors,
                'weights': self.weights,
                'algorithm': self.algorithm,
                'leaf_size': self.leaf_size,
                'p': self.p,
                'metric': self.metric
            }


            self.combinations = [
                {
                    'n_neighbors': n_neighbors,
                    'weights': weights,
                    'algorithm': algorithm,
                    'leaf_size': leaf_size,
                    'p': p,
                    'metric': metric
                }
                for n_neighbors in self.n_neighbors
                for weights in self.weights
                for algorithm in self.algorithm
                for leaf_size in self.leaf_size
                for p in self.p
                for metric in self.metric
                # Condiciones para evitar combinaciones no válidas
                if not (metric != 'minkowski' and p != 2)  # Si no es minkowski, p debe ser 2 (euclidean y manhattan no usan otros valores de p)
                if not (algorithm == 'kd_tree' and metric not in ['euclidean', 'manhattan'])  # kd_tree solo soporta euclidean y manhattan
            ]

    class logisticRegression:
        def __init__(self,
                    MAX_max_iter = 450):
            
            self.penalty = ['l1', 'l2', 'elasticnet', None]
            self.C = np.arange(0.5, 1.1, 0.1).tolist()
            self.max_iter = list(range(100, MAX_max_iter, 50))
            self.solver = ['lbfgsc', 'liblinear', 'saga']
            self.model = LogisticRegression

            self.grid = {
                'penalty': self.penalty,
                'C': self.C,
                'max_iter': self.max_iter,
                'solver': self.solver
            }

            self.combinations = [
                {
                    'penalty': penalty,
                    'C': C,
                    'max_iter': max_iter,
                    'solver': solver
                }
                for penalty in self.penalty
                for C in self.C
                for max_iter in self.max_iter
                for solver in self.solver
                if (solver == 'lbfgs' and penalty in ['l2', 'none']) or
                (solver == 'liblinear' and penalty in ['l1', 'l2']) or
                (solver == 'saga' and penalty in ['l1', 'l2', 'elasticnet'])
            ]

Mapper = {
        'tree': {
            'modelSk': ClassifierModel.tree(),
            'modelName': 'DecisionTreeClassifier',
            'output': 'decision_tree'
        },
        'randomForest': {
            'modelSk': ClassifierModel.randomForestClassifier(),
            'modelName': 'RandomForestClassifier',
            'output': 'random_forest'
        },
        'knn': {
            'modelSk': ClassifierModel.knn(),
            'modelName': 'KNeighborsClassifier',
            'output': 'knn'
        },
        'logisticRegression': {
            'modelSk': ClassifierModel.logisticRegression(),
            'modelName': 'LogisticRegression',
            'output': 'logistic_regression'
        }
    }