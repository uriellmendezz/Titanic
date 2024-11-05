import pandas as pd
import re

def proceso_ingenieria(df_main, df_ext):
    df = df_main.copy()

    # Elimino los duplicados y columnas irrelevantes.
    df = df.drop(['Ticket', 'Cabin'], axis = 1)

    # Creo columnas como si fuese un OneHotEncoder
    df = pd.get_dummies(df, columns=['Sex', 'Pclass', 'Embarked'])

    # Creo nuevas columnas que aportan valor y elimino otras.
    df.loc[:, 'Title'] = df.Name.str.extract(r' ([A-Za-z]+)\.')
    df.loc[:, 'FamilySize'] = df.SibSp + df.Parch
    df.drop(['SibSp', 'Parch'], axis = 1, inplace = True)

    # Empiezo a dejar las columnas con valores solamente numericos.
    df_num = df.drop(['Name'], axis = 1).merge(df_ext[['PassengerId', 'Age_wiki']], how = 'outer', on = 'PassengerId')
    df_num.Age = df_num.Age.fillna(df_num.Age_wiki)

    # A los valores nulos de la columna Age, les imputo la mediana.
    indexes = df_num.loc[df_num.Age.isna()].index
    df_num.loc[indexes, 'Age'] = df_num.Age.median()

    # Dejo solamente las columnas que de titulo solo tengan Miss, Master, Mr y Mrs.
    titulos_frecuentes = df_num.Title.value_counts().reset_index().Title[:4].values
    df_num_cortado = pd.get_dummies(df_num[df_num['Title'].isin(titulos_frecuentes)], ['Title'])
    df_num = pd.merge(df_num, df_num_cortado).drop(['Age_wiki', 'Title', 'PassengerId'], axis = 1).astype(float)

    return df_num.drop_duplicates().reset_index(drop=True)