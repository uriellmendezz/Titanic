import pandas as pd
import re
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

def Ingenieria(df_main, df_ext, transformer=None, dataset="train"):
    
    df = df_main.copy()

    # 1. Eliminar columnas irrelevantes (Ticket y Cabin).
    df = df.drop(['Ticket', 'Cabin'], axis=1)

    # 2. Realizar OneHotEncoding en variables categóricas.
    df = pd.get_dummies(df, columns=['Sex', 'Pclass', 'Embarked'])

    # 3. Crear nuevas columnas y eliminar otras.
    df.loc[:, 'Title'] = df.Name.str.extract(r' ([A-Za-z]+)\.')  # Extraer el título de la columna Name.
    df.loc[:, 'FamilySize'] = df.SibSp + df.Parch  # Crear una nueva columna 'FamilySize' con el tamaño de la familia.
    df.drop(['SibSp', 'Parch'], axis=1, inplace=True)  # Eliminar las columnas originales 'SibSp' y 'Parch'.

    # 4. Simplificar valores en la columna 'Title' para normalizar.
    df.Title = df.Title.replace({'Mlle': 'Miss', 'Mme': 'Mrs'})

    # 5. Combinar con datos adicionales y manejar valores nulos.
    df_num = df.drop(['Name'], axis=1).merge(df_ext[['PassengerId', 'Age_wiki']], how='inner', on='PassengerId')
    df_num.Age = df_num.Age.fillna(df_num.Age_wiki)  # Imputar valores nulos de 'Age' con 'Age_wiki'.

    # 6. Simplificar títulos y aplicar OneHotEncoding a los títulos más frecuentes.
    titulos_frecuentes = df_num.Title.value_counts().reset_index().Title[:4].values
    df_num_cortado = pd.get_dummies(df_num[df_num['Title'].isin(titulos_frecuentes)], columns=['Title'])
    df_num = pd.merge(df_num, df_num_cortado).drop(['Age_wiki', 'Title', 'PassengerId'], axis=1).astype(float)

    # 7. Crear nuevas features: 'FareXPassenger' y 'IsAlone'.
    df_num['FareXPassenger'] = df_num.Fare / (df_num.FamilySize + 1)  
    df_num['IsAlone'] = df_num.FamilySize.apply(lambda x: 1 if x == 0 else 0)
    df_num = pd.get_dummies(df_num, columns=['IsAlone'], prefix='IsAlone')

    # 8. Imputar valores nulos en 'Age' y realizar escalado si es el conjunto de entrenamiento.
    if dataset == "train":
        t1 = ColumnTransformer(transformers=[
            ('age_imputer', SimpleImputer(strategy='median'), ['Age'])  # Imputar valores nulos en 'Age' con la mediana.
        ], remainder='passthrough')

        t2 = ColumnTransformer(transformers=[
            ('minmax', MinMaxScaler(), ['Age', 'Fare', 'FareXPassenger'])  # Escalar las columnas 'Age', 'Fare' y 'FareXPassenger' entre 0 y 1.
        ], remainder='passthrough')

        df_num[['Age']] = t1.fit_transform(df_num[['Age']])
        df_num[['Age', 'Fare', 'FareXPassenger']] = t2.fit_transform(df_num[['Age', 'Fare', 'FareXPassenger']])

        transformers = (t1, t2)
        
    # 9. Si es el conjunto de prueba, solo aplicar transformaciones previamente ajustadas.
    elif dataset == "test" and transformer is not None:
        t1, t2 = transformer[0], transformer[1]
        df_num[['Age']] = t1.transform(df_num[['Age']])
        df_num[['Age', 'Fare', 'FareXPassenger']] = t2.transform(df_num[['Age', 'Fare', 'FareXPassenger']])

    # 10. Si no se proporcionan transformers en conjunto de prueba, lanzar error.
    else:
        raise ValueError("Para el test, se necesita un transformer previamente ajustado.")

    # Retornar los datos procesados, junto con los transformers si es el conjunto de entrenamiento.
    return (df_num, transformers) if dataset == 'train' else df_num