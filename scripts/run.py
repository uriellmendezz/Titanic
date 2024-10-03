from main import *
from utils import *

if __name__ == "__main__":
    path_dataset = "C:/Documents/Proyectos/proyecto-titanic/datasets/train-limpio.csv"
    df = load_dataset(path_dataset)
    x, y = df.drop('Survived', axis = 1), df.Survived

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 123)
    x_val_train, x_val_test, y_val_train, y_val_test = train_test_split(x_train, y_train, test_size = 0.3, random_state = 123)

    model = RandomForestClassifier()

    params = ClassifierModel.randomForestClassifier(MAX_n_estimators=40, MAX_max_depth=15).grid
    print(len(params))

    grid = GridSearchCV(
        estimator = model,
        scoring = 'accuracy',
        cv = 3,
        n_jobs = -1,
        param_grid = params
    ).fit(x_val_train, y_val_train)

    # Obtener los resultados en un DataFrame
    results_df = pd.DataFrame(grid.cv_results_)

    # Filtrar las columnas que te interesan (por ejemplo, params y mean_test_score)
    results_df = results_df[['params', 'mean_test_score']]

    # Renombrar la columna mean_test_score a accuracy
    results_df.rename(columns={'mean_test_score': 'accuracy'}, inplace=True)

    # Exportar a un archivo CSV (opcional)
    results_df.to_csv('grid_search_results.csv', index=False)