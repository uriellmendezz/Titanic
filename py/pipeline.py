import pandas as pd
import re
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def Ingenieria(df_main, df_ext, transformer=None, dataset="train"):
    df = df_main.copy()

    # Elimino duplicados y columnas irrelevantes.
    df = df.drop(['Ticket', 'Cabin'], axis=1)

    # Codificación de variables categóricas.
    df = pd.get_dummies(df, columns=['Sex', 'Pclass', 'Embarked'])

    # Crear nuevas columnas y eliminar algunas.
    df.loc[:, 'Title'] = df.Name.str.extract(r' ([A-Za-z]+)\.')
    df.loc[:, 'FamilySize'] = df.SibSp + df.Parch
    df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    # Convertir a valores numéricos y agregar columnas de df_ext.
    df_num = df.drop(['Name'], axis=1).merge(df_ext[['PassengerId', 'Age_wiki']], how='outer', on='PassengerId')
    df_num.Age = df_num.Age.fillna(df_num.Age_wiki)

    # Imputar valores nulos en Age
    if dataset == "train":
        transformer = ColumnTransformer(
            transformers=[
                ('age_imputer', SimpleImputer(strategy='mean'), ['Age'])
            ],
            remainder='passthrough'
        )

        df_num[['Age']] = transformer.fit_transform(df_num[['Age']])

    elif dataset == "test" and transformer is not None:
        df_num[['Age']] = transformer.transform(df_num[['Age']])

    else:
        raise ValueError("Para el test, se necesita un transformer previamente ajustado.")

    # Simplificar títulos y aplicar OneHotEncoding en los más frecuentes.
    titulos_frecuentes = df_num.Title.value_counts().reset_index().Title[:4].values
    df_num_cortado = pd.get_dummies(df_num[df_num['Title'].isin(titulos_frecuentes)], columns=['Title'])
    df_num = pd.merge(df_num, df_num_cortado).drop(['Age_wiki', 'Title', 'PassengerId'], axis=1).astype(float)

    if dataset == "train":
        return df_num.drop_duplicates().reset_index(drop=True), transformer
    else:
        return df_num.drop_duplicates().reset_index(drop=True)
