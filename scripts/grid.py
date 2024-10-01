from sklearn.model_selection import GridSearchCV, train_test_split

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