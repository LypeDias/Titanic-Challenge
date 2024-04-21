import pandas as pd
import numpy as np

train = pd.read_csv(".\\train.csv")
test = pd.read_csv(".\\test.csv")
train.head()
print(train.head())

from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
variaveis = ['Sex_binario', 'Age']
train['Sex'].value_counts()
print(train['Sex'].value_counts())
def transformar_sexo(valor):
    if valor == 'female':
        return 1
    else:
        return 0
train ['Sex_binario'] = train ['Sex'].map(transformar_sexo)    
train.head()
print(train.head())


variaveis = ['Sex_binario', 'Age']
X = train[variaveis]
Y = train['Survived']

X.head()
print(X.head())

Y.head()
print(Y.head())

modelo.fit(X,Y)
print(modelo.fit(X,Y))

X = X.fillna(-1)

modelo.fit(X,Y)
print(modelo.fit(X,Y))


test['Sex_binario'] = test['Sex'].map(transformar_sexo)
X_prev = test[variaveis]
X_prev = X_prev.fillna(-1)
X_prev.head()
print(X_prev.head())

p = modelo.predict(X_prev)
print(p)

print(test.head())
sub = pd.Series(p, index=test['PassengerId'], name = 'Survived')
print(pd.Series(p, index=test['PassengerId']))
sub.shape
print(sub.shape)
sub.to_csv("primeiro_modelo.csv", header=True)
df = pd.read_csv("primeiro_modelo.csv", nrows=10)
print(df)







