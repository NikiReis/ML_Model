# importing the necessaries libraries
import pandas as pd
import numpy as np

# importing the random forest algorith to aplay to our model and the base of test and training to uor algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# read the CSV file that contains the data
dados = pd.read_csv('/content/train2.csv')
# Show a pre-view of the data
dados.head()
# drop some columns from the database that are useless to our model
dados = dados.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
# show the new formatting
dados.head()

dados = dados.set_index(['PassengerId']) # setting "PassengerId" as a strong key index in our base
dados = dados.rename(columns={'Survived': 'target'}, inplace=False) # setting "Survived" as target variable

dados.head() # show the database formatted and reduced

dados.describe() # Do an Analysis from numeric variables
dados.describe(include=['O']) # Add a non-numeric variable to our Analysis

dados['Sex_F'] = np.where(dados['Sex'] == 'female', 1, 0) # Transform the non-numeric data to a numeric data (Sex) to
# the Analysis

# Most of the Machine Learning doesn't accept categorical data, so to solve this problem we have to transform into
# numeric data
dados['Pclass_1'] = np.where(dados['Pclass'] == 1, 1, 0) # Transforming the passenger's class categorical data into numerical data
dados['Pclass_2'] = np.where(dados['Pclass'] == 2, 1, 0) # Transforming the passenger's class categorical data into numerical data
dados['Pclass_3'] = np.where(dados['Pclass'] == 3, 1, 0) # Transforming the passenger's class categorical data into numerical data

dados = dados.drop(['Pclass', 'Sex'], axis=1) # deleting the original columns after the data transformation

dados.head() # show the new db model

dados.isnull().sum() # Verifies where is lost data

dados.fillna(0, inplace=True) # Fill the columns with lost data with the number zero

dados.isnull().sum() # Variables summary after the substitutions

x_train, x_test, y_train, y_test = train_test_split(
    dados.drop(['target'], axis=1), # show the sample division in conformation to the target variable
    dados['target'], # The parameter of the division is Survived variable
    test_size=0.3, # set the db division in 30/70. 70% training and 30% test
    random_state=1234 # fix the model's precision result to late reproduction
)

# demonstrate the division in training and test with number of lines and columns for which db
[
    {'treino': x_train.shape},
    {'teste': x_test.shape}
]

rdnforest = RandomForestClassifier(
    n_estimators=1000, # create a model with a thousand trees
    criterion='gini', # Fix an Analysis criteria, gini isolates 1 branch of the tree to the most frequent class
    max_depth=5 # Set the max depth of the nodes as 5
)
rdnforest.fit(x_train, y_train) # Calculates the model according to the passed parameters

probabilidade = rdnforest.predict_proba(dados.drop('target', axis=1))[:, 1] # prediction with probability regression of survive
classificacao = rdnforest.predict(dados.drop('target', axis=1)) # prediction with survivor or not survivor classification

dados['probabilidade'] = probabilidade # Defines the column with probability result
dados['classificacao'] = classificacao # Defines the column with classification result

dados # Final ML Analysis with survive probability rate
