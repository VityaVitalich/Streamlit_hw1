import numpy as np
import pandas as pd

from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

pd.options.mode.chained_assignment = None


def education_transform(val, inverse=False):
    mapping ={
        'Среднее специальное': 0, 'Среднее':1, 'Высшее':2, 'Неоконченное высшее':3,
    'Неполное среднее':4, 'Два и более высших образования':5, 'Ученая степень':6
    }
    if not inverse:
        return mapping[val]
    else:
        inv_map = {v: k for k, v in mapping.items()}
        return inv_map[val]
    
def marital_transform(val, inverse=False):
    mapping = {
        'Состою в браке': 0, 'Не состоял в браке': 1, 'Разведен(а)': 2, 'Вдовец/Вдова': 3,
    'Гражданский брак': 4
    }
    if not inverse:
        return mapping[val]
    else:
        inv_map = {v: k for k, v in mapping.items()}
        return inv_map[val]

def income_transform(val, inverse=False):
    mapping = {
        'от 10000 до 20000 руб.': 0, 'от 20000 до 50000 руб.': 1,
    'от 5000 до 10000 руб.': 2, 'свыше 50000 руб.': 3, 'до 5000 руб.': 4
    }

    if not inverse:
        return mapping[val]
    else:
        inv_map = {v: k for k, v in mapping.items()}
        return inv_map[val]

def encode_categories(df: pd.DataFrame):
    """ encodes categorical columns """

    df['EDUCATION'] = df['EDUCATION'].apply(education_transform)
    df['MARITAL_STATUS'] = df['MARITAL_STATUS'].apply(marital_transform)
    df['FAMILY_INCOME'] = df['FAMILY_INCOME'].apply(income_transform)

    try:
        dropping = ['AGREEMENT_RK', 'ID', 'REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE','POSTAL_ADDRESS_PROVINCE']
        df = df.drop(columns=dropping)
    except KeyError:
        pass 

    df_encoded = pd.get_dummies(df, columns=['EDUCATION', 'MARITAL_STATUS', 'FAMILY_INCOME'])

    print(df_encoded.columns)
    return df_encoded

def preprocess_data(df: pd.DataFrame):
    """ runs preprocessing on dataset """

    df_encoded_cats = encode_categories(df)

    # разделение данных
    X, y = df_encoded_cats.drop(['TARGET'], axis=1), df_encoded_cats['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # масштабирование
    # ss = MinMaxScaler()
    # ss.fit(X_train)

    # X_train = pd.DataFrame(ss.transform(X_train), columns=X_train.columns)
    # X_test = pd.DataFrame(ss.transform(X_test), columns=X_test.columns)

    return X_train, X_test, y_train, y_test


def fit_and_save_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,
                       path="data/model_weights.mw",
                       test_model=True,
                       metric='accuracy'):
    """ fits logistic regression model """
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    if test_model:
        preds = model.predict(X_test)
        if metric == 'accuracy':
            score = accuracy_score(y_test, preds)
        elif metric == 'recall':
            score = recall_score(y_test, preds)
        elif metric == 'precision':
            score = precision_score(y_test, preds)
        print(f'{metric.title()}: {round(score, 3)}')

    dump_model(model)
    save_importances(model, X_train.columns)


def dump_model(model, path="data/model_weights.mw"):
    """ saves model as pickle file """

    with open(path, "wb") as file:
        dump(model, file)

    print(f'Model was saved to {path}')


def save_importances(model, feature_names, path='data/importances.csv'):
    """ saves sorted feature weights as df """

    new_feature_names = []
    for name in feature_names:
        if ('EDUCATION' in name) or ('MARITAL_STATUS' in name) or ('FAMILY_INCOME' in name):
            category = int(name[-1])
            column_name = name[:-2]
            if column_name == 'EDUCATION':
                new_feature_names.append(column_name + '=' + education_transform(category, inverse=True))
            elif column_name == 'MARITAL_STATUS':
                new_feature_names.append(column_name + '=' + marital_transform(category, inverse=True))
            elif column_name == 'FAMILY_INCOME':
                new_feature_names.append(column_name + '=' + income_transform(category, inverse=True))
            else:
                raise TypeError()

        else:
            new_feature_names.append(name)
    
    importances = pd.DataFrame({'Признак': new_feature_names, 'Вес': model.coef_[0]})
    importances.sort_values(by='Вес', key=abs, ascending=False, inplace=True)

    importances.to_csv(path, index=False)
    print(f'Importances were saved to {path}')


def load_model(path="data/model_weights.mw"):
    """ load model from saved weights """

    with open(path, "rb") as file:
        model = load(file)

    return model


def get_importances(top_n=5, importance='most', path='data/importances.csv'):
    """ returns top n most important or least important weights """

    importances = pd.read_csv(path, encoding='utf-8')
    if importance == 'most':
        return importances.head(top_n)
    else:
        return importances.tail(top_n).iloc[::-1]


def predict_on_input(df: pd.DataFrame):
    """ loads model and returns prediction and probability """

    model = load_model()
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)

    return pred, proba

def main():
    df = pd.read_csv('data/result.csv')

    X_train, X_test, y_train, y_test = preprocess_data(df)
    fit_and_save_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()