from main import load_dataset, train_test_division, validation_division, train_model, execute, download_results

if __name__ == '__main__':
    df = load_dataset()
    x_train, x_test, y_train, y_test = train_test_division(df)
    x_val_train, x_val_test, y_val_train, y_val_test = validation_division(x_train, y_train)

    model_selection, output = 'knn', 'knn-validation'
    results = execute(
        model_selection,
        x_val_train, x_val_test, y_val_train, y_val_test,
        max_combinations = 200
    )

    download_results(results, output)