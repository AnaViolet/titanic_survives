import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('C:/Users/paula/Desktop/PyWorkplace/titanic_survives/data/train.csv')
test_data = pd.read_csv("C:/Users/paula/Desktop/PyWorkplace/titanic_survives/data/test.csv")

women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women) / len(women)
print("{:.2%} of women survived".format(rate_women))

men = train_data.loc[train_data.Sex == 'male']['Survived']
rate_men = sum(men) / len(men)
print("{:.2%} of men survived".format(rate_men))

#select the survived column
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]

#convert Series to dummy (0|1) codes. 
X = pd.get_dummies(train_data[features])
X_teste = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
#Build a forest of trees from the training set (X, y).
model.fit(X, y)
#Predict class for X_teste
predictions = model.predict(X_teste)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('prediction.csv', index=False)
print('Your submission was sucessfullt saved!')
