import pandas as pd
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv("./data/test.csv")

#gender prediction
women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women) / len(women)
print("{:.2%} of women survived".format(rate_women))

men = train_data.loc[train_data.Sex == 'male']['Survived']
rate_men = sum(men) / len(men)
print("{:.2%} of men survived".format(rate_men))

#We can see that a person in Pclass = 1 has a higher probability of survival
grid = sn.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Sex', alpha=.5, bins=20)
grid.add_legend();
grid.set_xlabels("Sex")
plt.show()

#Here we will build a random forest model
#select the survived column
y = train_data["Survived"]

#We will look for patterns in five differents columns that looks to
#contribute to a person's chance of survival
features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

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

print("Accuracy: {}%".format(round(model.score(X, y) * 100, 2)))
