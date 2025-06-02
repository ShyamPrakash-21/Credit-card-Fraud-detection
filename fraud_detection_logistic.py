import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# %matplotlib inline  # Not used in scripts

df = pd.read_csv('creditcard.csv')
print(df.shape)
df.head()
df.info()
df.describe()
class_names = {0:'Not Fraud', 1:'Fraud'}
print(df.Class.value_counts().rename(index = class_names))

fig = plt.figure(figsize = (15, 12))
for i in range(1, 29):
    plt.subplot(5, 6, i)
    plt.plot(df[f'V{i}'])
plt.subplot(5, 6, 29)
plt.plot(df.Amount)
plt.tight_layout()
plt.savefig("features_plot.png")

from sklearn.model_selection import train_test_split
feature_names = df.iloc[:, 1:30].columns
target = df.iloc[:1, 30:].columns
print(feature_names)
print(target)

data_features = df[feature_names]
data_target = df[target]
X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=1)
print(f"Length of X_train: {len(X_train)}")
print(f"Length of X_test: {len(X_test)}")
print(f"Length of y_train: {len(y_train)}")
print(f"Length of y_test: {len(y_test)}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score

model = LogisticRegression()
model.fit(X_train, y_train.values.ravel())
pred = model.predict(X_test)

class_names = ['not_fraud', 'fraud']
matrix = confusion_matrix(y_test, pred)
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt='g')
plt.title("Confusion Matrix")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

f1 = round(f1_score(y_test, pred), 2)
recall = round(recall_score(y_test, pred), 2)
print(f"Sensitivity/Recall for Logistic Regression Model: {recall}")
print(f"F1 Score for Logistic Regression Model: {f1}")
