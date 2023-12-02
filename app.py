import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotter import draw_features, draw_target
import model
import streamlit as st
import time


def highlight_weighs(s):
    """ generate colors to highlight weights """

    return ['background-color: #E6F6E4']*len(s) if s['Вес'] > 0 else ['background-color: #F6EBE4']*len(s)

def educational_choice(education):
    mapping ={
        'Среднее специальное': 0, 'Среднее':1, 'Высшее':2, 'Неоконченное высшее':3,
    'Неполное среднее':4, 'Два и более высших образования':5, 'Ученая степень':6
    }
    out = {}

    for k,v in mapping.items():
        if k == education:
            out['EDUCATION_' + str(v)] = 1
        else:
            out['EDUCATION_' + str(v)] = 0
    return out

def income_choice(income):
    mapping = {
        'от 10000 до 20000 руб.': 0, 'от 20000 до 50000 руб.': 1,
    'от 5000 до 10000 руб.': 2, 'свыше 50000 руб.': 3, 'до 5000 руб.': 4
    }
    out = {}

    for k,v in mapping.items():
        if k == income:
            out['FAMILY_INCOME_' + str(v)] = 1
        else:
            out['FAMILY_INCOME_' + str(v)] = 0
    return out


def marital_choice(marital):
    mapping = {
        'Состою в браке': 0, 'Не состоял в браке': 1, 'Разведен(а)': 2, 'Вдовец/Вдова': 3,
    'Гражданский брак': 4
    }
    out = {}
    for k,v in mapping.items():
        if k == marital:
            out['MARITAL_STATUS_' + str(v)] = 1
        else:
            out['MARITAL_STATUS_' + str(v)] = 0
    return out

def pack_input(sex, age, education, workstatus, FL, marital, pensstatus, car, child, income, pers_income, 
            dependance, loans, closed_loans):
    """ translate input values to pass to model """

    rule = {'Женский': 0,
            'Мужской': 1,
            "Да": 1,
            "Нет": 0}

    data = {'AGE': age,
            'GENDER': rule[sex],
            'CHILD_TOTAL': child,
            'DEPENDANTS': dependance,
            'SOCSTATUS_WORK_FL': rule[workstatus],
            'SOCSTATUS_PENS_FL': rule[pensstatus],
            'FL_PRESENCE_FL': rule[FL],
            'OWN_AUTO': car,
            #'FAMILY_INCOME': income,
            'PERSONAL_INCOME': pers_income,
            'NUM_LOANS': loans,
            'NUM_CLOSED_LOANS': closed_loans}
    
    data.update(educational_choice(education))
    data.update(marital_choice(marital))
    data.update(income_choice(income))

    df = pd.DataFrame(data, index=[0])
   # encoded = model.encode_categories(df)
    return df

if __name__ == '__main__':

    st.title('Исследование склонности к отклику клиентов банка')
    st.subheader('Разведочный анализ и моделирование')
    

    df = pd.read_csv('data/result.csv')

    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Численные статистики', 'Графики по признакам', 'Графики признаков с тагетом', 'Признаки в модель', "Предсказать"])
    with tab1:
        numerical_columns = ['AGE', 'CHILD_TOTAL', 'OWN_AUTO', 'PERSONAL_INCOME', 'NUM_LOANS', 'NUM_CLOSED_LOANS']
        statistics = df.describe()[numerical_columns]
        st.dataframe(statistics)
    with tab2:
       # st.write('test')
        draw_features(df)
    with tab3:
        st.write('Ниже приведены графики по признакам в связи с таргетом. Видно, что почти везде они почти независимы, кроме наверно статуса работы и статуса пенсионера')
        draw_target(df)
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.write('**Какой человек скорее всего согласится на наше предложение?**')
            st.dataframe(model.get_importances(5, 'most').style.apply(highlight_weighs, axis=1))
        with col2:
            st.write('**А что практически не важно?**')
            st.dataframe(model.get_importances(5, 'least').style.apply(highlight_weighs, axis=1))
    with tab5:
        st.write('Введите данные клиента:')


        col1, col2, col3 = st.columns(3)
        with col1:
            sex = st.selectbox('Пол', ['Женский', 'Мужской'])
            education = st.selectbox('Образование', ['Среднее специальное', 'Среднее', 'Высшее',
             'Неоконченное высшее', 'Неполное среднее', 'Два и более высших образования', 'Ученая степень'])
            workstatus = st.radio('Работаете в настоящее время', ["Нет", "Да"])
            FL = st.radio('Есть квартира в собственности', ["Нет", "Да"])
        with col2:
            age = st.slider('Возраст', min_value=0, max_value=100)
            marital = st.selectbox('Семейный статус', ['Состою в браке', 'Не состоял в браке', 'Разведен(а)',
            'Вдовец/Вдова', 'Гражданский брак'])
            pensstatus = st.radio('Являетесь ли пенсионером', ["Нет", "Да"])
            car = st.slider('Количество машин', min_value=0, max_value=100)
        with col3:
            child = st.slider('Количество детей', min_value=0, max_value=10)
            income = st.selectbox('Семейный доход', ['от 10000 до 20000 руб.', 'от 20000 до 50000 руб.', 'от 5000 до 10000 руб.',
            'свыше 50000 руб.', 'до 5000 руб.'])
            pers_income = st.slider('Персональный Доход', min_value=0, max_value=500000, step=1000)
            dependance = st.slider('Количество иждевенцев', min_value=0, max_value=10)

        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            loans = st.slider('Количество кредитов за все время', min_value=0, max_value=10)
        with col2:
            closed_loans = st.slider('Количество закрытых кредитов за все время', min_value=0, max_value=10)
        st.divider()

        col1, col2, col3 = st.columns(3)
        if col2.button('Рассчитать'):
            with st.spinner('Считаем'):
                time.sleep(1)
                inputs = pack_input(sex, age, education, workstatus, FL, marital, pensstatus, car, child, income, pers_income, 
                                    dependance, loans, closed_loans)

                pred, proba = model.predict_on_input(inputs)
                if pred == 1:
                    st.success('Этому клиенту стоит направить предложение :thumbsup: :thumbsup:')
                    with st.expander('Подробнее'):
                        st.write(f'Вероятность согласия: **`{round(max(proba[0]), 3)}`**')
                elif pred == 0:
                    st.error('Этому клиенту не стоит направлять предложение :thumbsdown: :thumbsdown:')
                    with st.expander('Подробнее'):
                        st.write(f'Вероятность согласия: **`{round(max(proba[0]), 3)}`**')
                else:
                    st.error('Что-то пошло не так...')
    

