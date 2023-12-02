import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_gender_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='GENDER', data=df, ax=ax)
    ax.set_title('Distribution of Gender')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Female', 'Male'])
    return fig, ax

def plot_working_status_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='SOCSTATUS_WORK_FL', data=df, ax=ax)
    ax.set_title('Distribution of Working Status')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Unemployed', 'Employed'])
    ax.set_xlabel('Working Status')
    return fig, ax

def plot_retirement_status_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='SOCSTATUS_PENS_FL', data=df, ax=ax)
    ax.set_title('Distribution of Pensioner Status')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Retired', 'Retired'])
    ax.set_xlabel('Retirement status')
    return fig, ax

def plot_top_regions_distribution(df):
    top_regions = df['REG_ADDRESS_PROVINCE'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_regions.index, y=top_regions.values, ax=ax)
    ax.set_title('Top 10 Regions of Living')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Region')
    ax.set_ylabel('Count')
    return fig, ax

def plot_flat_ownership_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='FL_PRESENCE_FL', data=df, ax=ax)
    ax.set_title('Distribution of Flat Ownership')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not owning', 'Owning'])
    ax.set_xlabel('Flat ownership')
    return fig, ax

def plot_autos_vs_income_boxplot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='OWN_AUTO', y='PERSONAL_INCOME', data=df, ax=ax)
    ax.set_title('Boxplot of Autos vs Personal Income')
    ax.set_xlabel('Number of owned autos')
    return fig, ax

def plot_family_income_distribution(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='FAMILY_INCOME', data=df, ax=ax)
    ax.set_title('Distribution of Family Income')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    return fig, ax

def plot_loans_scatter(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='NUM_LOANS', y='NUM_CLOSED_LOANS', data=df, ax=ax)
    ax.set_title('Scatter Plot of Number of Loans vs Number of Closed Loans')
    ax.set_xlabel('Number of Loans')
    ax.set_ylabel('Number of closed loans')
    return fig, ax

def plot_correlation_heatmap(df):
    numerical_columns = ['AGE', 'CHILD_TOTAL', 'PERSONAL_INCOME', 'NUM_LOANS', 'NUM_CLOSED_LOANS', 'OWN_AUTO']
    correlation_matrix_all = df[numerical_columns].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = sns.heatmap(correlation_matrix_all, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
    ax.set_title('Correlation Heatmap for Numerical Columns')
    return fig, ax

def plot_family_income_with_working_status(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='FAMILY_INCOME', hue='SOCSTATUS_WORK_FL', data=df, ax=ax)
    ax.set_title('Family Income Distribution with Working Status')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Working Status', labels=['Not Working', 'Working'])
    return fig, ax

def draw_features(df):
    # Plot a histogram of the AGE column
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['AGE'], bins=20, kde=True, ax=ax)
    plt.title('Distribution of Age')
    st.pyplot(fig)
    st.write('Видим что возраст распределен относительно равномерно с правым хвостом после 60')
    st.divider()

    # Plot Gender Distribution
    fig, ax = plot_gender_distribution(df)
    st.pyplot(fig)
    st.write('Количество мужчин в нашей выборке вдвое больше')
    st.divider()

    # Plot Working Status Distribution
    fig, ax = plot_working_status_distribution(df)
    st.pyplot(fig)
    st.write('Почти все люди работают')
    st.divider()

    # Plot Retirement Status Distribution
    fig, ax = plot_retirement_status_distribution(df)
    st.pyplot(fig)
    st.write('Также почти все не пенсионеры')
    st.divider()

    # Plot Top Regions Distribution
    fig, ax = plot_top_regions_distribution(df)
    st.pyplot(fig)
    st.write('Регионы распределены относительно равномерно, Краснодарский край и Кемеровская область слегка превалируют')
    st.divider()

    # Plot Flat Ownership Distribution
    fig, ax = plot_flat_ownership_distribution(df)
    st.pyplot(fig)
    st.write('Вдвое больше людей у которых нет квартиры в собственности')
    st.divider()

    # Plot Autos vs Income Boxplot
    fig, ax = plot_autos_vs_income_boxplot(df)
    st.pyplot(fig)
    st.write('Видим что есть небольшая связь между доходом и количеством машин, но не очень высокая')
    st.divider()

    # Plot Family Income Distribution
    fig, ax = plot_family_income_distribution(df)
    st.pyplot(fig)
    st.write('В основном люди зарабатывают до 50т рублей, весьма грустно')
    st.divider()

    # Plot Loans Scatter Plot
    fig, ax = plot_loans_scatter(df)
    st.pyplot(fig)
    st.write('Высокая взаимосвязь между количеством взятых кредитов и закрытых. Удивительно, что не наблюдается даже небольших выбросов')
    st.divider()

    # Plot Correlation Heatmap
    fig, ax = plot_correlation_heatmap(df)
    st.pyplot(fig)
    st.write('В целом все корреляции между признаками маленькие, кроме количества взятых кредитов и закрытых. Можно еще выделить количество детей и возраст, что тоже логично')
    st.divider()

    # Plot Family Income with Working Status
    fig, ax = plot_family_income_with_working_status(df)
    st.pyplot(fig)
    st.write('В основных группах почти нет неработающих, в группе с доходом выше 50т вообще нет, а вот в группах с низким доходом треть и выше')
    st.divider()

def draw_target(df):
    target_column = 'TARGET'

    # Numerical columns
    numerical_columns = ['AGE', 'CHILD_TOTAL', 'OWN_AUTO', 'PERSONAL_INCOME', 'NUM_LOANS', 'NUM_CLOSED_LOANS']

    # fig, ax = plt.subplots(figsize=(8, 6))
    # sns.countplot(x='FL_PRESENCE_FL', data=df, ax=ax)
    # ax.set_title('Distribution of Flat Ownership')
    # ax.set_xticks([0, 1])
    # ax.set_xticklabels(['Not owning', 'Owning'])
    # ax.set_xlabel('Flat ownership')
    for column in numerical_columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=target_column, y=column, data=df, ax=ax)
        ax.set_title(f'{column} Distribution with Respect to {target_column}')
        st.pyplot(fig)

    # Categorical columns
    categorical_columns = ['GENDER', 'MARITAL_STATUS', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL',
                            'FL_PRESENCE_FL', ]

    for column in categorical_columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x=column, hue=target_column, data=df, ax=ax)
        ax.set_title(f'Distribution of {column} with Respect to {target_column}')
        st.pyplot(fig)


    columns = ['EDUCATION', 'FAMILY_INCOME']
    for column in columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x=column, hue=target_column, data=df, ax=ax)
        ax.set_title(f'Distribution of {column} with Respect to {target_column}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)


    fig, ax = plt.subplots(figsize=(14, 8))
    sns.countplot(x='REG_ADDRESS_PROVINCE', hue=target_column, data=df, order=df['REG_ADDRESS_PROVINCE'].value_counts().head(10).index, ax=ax)
    ax.set_title(f'Distribution of Top 10 Regions with Respect to {target_column}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

def draw_all():  

    df = pd.read_csv('result.csv')
    draw_features(df)



