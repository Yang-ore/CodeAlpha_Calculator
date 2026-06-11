import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


iris= load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target     


print("First 5 rows of the dataset:")
print(iris_df.head())

print("\nDataset summary:")
print(iris_df.describe())
print("\nBasic Statisitcs:")
print(iris_df.info())

sns.pairplot(iris_df, hue='target')
plt.suptitle("Pairplot of Iris Dataset", y=1.02)    
plt.savefig("iris_pairplot.png")
plt .show()

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
printz("Classification Report:")
print ('An Accuracy of {:.2f}%'.format(model.score(X_test, y_test) * 100))
