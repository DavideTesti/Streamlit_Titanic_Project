## STREAMLIT ORGANIZATION

# Import the Streamlit library and the necessary data exploration 
# and DataVizualization libraries.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a dataframe called df to read the file train.csv.
df=pd.read_csv("train.csv")

# Create 3 pages called "Exploration", "DataVizualization" and "Modelling" on Streamlit.
st.title("Titanic : binary classification project")
st.sidebar.title("Table of contents")
pages=["Exploration", "DataVizualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)


## DATA EXPLORATION

# Write "Presentation of data" at the top of the first page.
if page == pages[0] : 
    st.write("### Presentation of data")

    # Display the first 10 lines of df on the web application Streamlit.
    st.dataframe(df.head(10))

    # Display informations about the dataframe on the Streamlit web application.
    st.write(df.shape)
    st.dataframe(df.describe())

    # Create a checkbox to choose whether to display the number of missing values or not.
    if st.checkbox("Show NA") :
        st.dataframe(df.isna().sum())


## DATA VISUALIZATION

# Write "DataVizualization" at the top of the second page.
if page == pages[1] : 
    st.write("### DataVizualization")

    # Display in a plot the distribution of the target variable.
    fig = plt.figure()
    plt.title("Distribution of the target variable")
    sns.countplot(x = 'Survived', data = df)
    st.pyplot(fig)

    # Display plots to describe the Titanic passengers. Add titles to the plots.
    gig = plt.figure()
    sns.countplot(x = 'Sex', data = df)
    plt.title("Distribution of the passengers gender")
    st.pyplot(gig)
    fig = plt.figure()
    sns.countplot(x = 'Pclass', data = df)
    plt.title("Distribution of the passengers class")
    st.pyplot(fig)
    fig = sns.displot(x = 'Age', data = df)
    plt.title("Distribution of the passengers age")
    st.pyplot(fig)

    # Display a countplot of the target variable according to the gender.
    # Display a plot of the target variable according to the classes.
    # Display a plot of the target variable according to the age.
    fig = plt.figure()
    sns.countplot(x = 'Survived', hue='Sex', data = df)
    plt.title("Countplot of the target variable according to the gender")
    st.pyplot(fig)
    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    plt.title("Plot of the target variable according to the classes")
    st.pyplot(fig)
    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    plt.title("Plot of the target variable according to the age")
    st.pyplot(fig)

    # Display the correlation matrix of the explanatory variables.
    # Drop specified columns to create a new DataFrame
    fig, ax = plt.subplots()
    columns_to_drop = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
    new_df = df.drop(columns=columns_to_drop)
    plt.title("Correlation matrix of some explanatory variables")
    sns.heatmap(new_df.corr(), ax=ax)
    st.write(fig)


## MODELLING
    
# Write "Modelling" at the top of the third page.
if page == pages[2] : 
    st.write("### Modelling")

    # Remove the irrelevant variables (PassengerID, Name, Ticket, Cabin).
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Create a variable y containing the target variable. 
    # Create a dataframe X_cat containing the categorical explanatory variables 
    # and a dataframe X_num containing the numerical explanatory variables.
    y = df['Survived']
    X_cat = df[['Pclass', 'Sex',  'Embarked']]
    X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

    # Replace the missing values for categorical variables by the mode 
    # and replace the missing values for numerical variables by the median.
    # Encode the categorical variables.
    # Concatenate the encoded explanatory variables without missing values to obtain a clean X dataframe.
    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())
    X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
    X = pd.concat([X_cat_scaled, X_num], axis = 1)

    # Separate the data into a train set 
    # and a test set using the train_test_split function from the Scikit-Learn model_selection package.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Standardize the numerical values using the StandardScaler function from the Preprocessing package of Scikit-Learn.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    # Import of libraries for the models.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    # Create a function called prediction which takes the name of a classifier as an argument 
    # and which returns the trained classifier.
    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            # Saving of the model
            import joblib
            joblib.dump(clf, "random_forest_model")
        elif classifier == 'SVC':
            clf = SVC()
            clf.fit(X_train, y_train)
            # Saving of the model
            import joblib
            joblib.dump(clf, "svc_model")
        elif classifier == 'Logistic Regression':
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            # Saving of the model
            import joblib
            joblib.dump(clf, "logistic_regression_model")
        return clf

    # Since the classes are not unbalanced, it is interesting to look at the accuracy of the predictions.  
    # Creates a function which returns either the accuracy or the confusion matrix.
    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))
        
    # Allow to choose between the RandomForest classifier, the SVM classifier, and the LogisticRegression classifier.
    choice = ['Random Forest', 'SVC', 'Logistic Regression']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is:', option)

    # Load pre-trained models
    import joblib
    if option == 'Random Forest':
        clf = joblib.load("random_forest_model")
    elif option == 'SVC':
        clf = joblib.load("svc_model")
    elif option == 'Logistic Regression':
        clf = joblib.load("logistic_regression_model")

    # Displays checkboxes to choose between many options.
    display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))