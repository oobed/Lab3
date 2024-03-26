
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score

# 1a) Load the data
data = pd.read_csv("data/dataR2.csv")

# Exploratory data analysis
sns.pairplot(data, hue="Classification")
plt.show()

# 1b) Select training and test sets
class_0_train = data[data["Classification"] == 0].iloc[:40]
class_1_train = data[data["Classification"] == 1].iloc[:48]
training_set = pd.concat([class_0_train, class_1_train])
test_set = data.drop(training_set.index)

# 1c) Classification using KNN
train_error_rates = []
test_error_rates = []
k_range = range(1, 21)  # Consider different values of k

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(training_set.drop("Classification", axis=1), training_set["Classification"])

    train_pred = knn.predict(training_set.drop("Classification", axis=1))
    train_error = 1 - accuracy_score(training_set["Classification"], train_pred)
    train_error_rates.append(train_error)

    test_pred = knn.predict(test_set.drop("Classification", axis=1))
    test_error = 1 - accuracy_score(test_set["Classification"], test_pred)
    test_error_rates.append(test_error)

# Plot training and test error rates
plt.plot(k_range, train_error_rates, label='Training Error Rate')
plt.plot(k_range, test_error_rates, label='Test Error Rate')
plt.xlabel('k')
plt.ylabel('Error Rate')
plt.title('Training and Test Error Rates vs. k')
plt.legend()
plt.show()

# Find the best k
best_k = k_range[np.argmin(test_error_rates)]
print("Best k:", best_k)

# 1d) Replace the Euclidean metric with the Minkowski distance
min_test_error_rate = float('inf')
best_k = None
best_p = None

for k in range(1, 21):
    for p in range(1, 6):
        knn = KNeighborsClassifier(n_neighbors=k, p=p)
        knn.fit(training_set.drop("Classification", axis=1), training_set["Classification"])
        test_pred = knn.predict(test_set.drop("Classification", axis=1))
        test_error_rate = 1 - accuracy_score(test_set["Classification"], test_pred)

        if test_error_rate < min_test_error_rate:
            min_test_error_rate = test_error_rate
            best_k = k
            best_p = p

print("Best k:", best_k)
print("Best p:", best_p)
print("Misclassification Error Rate on Test Set:", min_test_error_rate)

#Extra Credit
k_range = range(1, 21)
p_range = range(1, 6)

min_test_error_rate = float('inf')
best_k = None
best_p = None

for k in k_range:
    for p in p_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='minkowski', p=p)
        knn.fit(training_set.drop("Classification", axis=1), training_set["Classification"])

        train_pred = knn.predict(training_set.drop("Classification", axis=1))
        train_error_rate = 1 - accuracy_score(training_set["Classification"], train_pred)

        test_pred = knn.predict(test_set.drop("Classification", axis=1))
        test_error_rate = 1 - accuracy_score(test_set["Classification"], test_pred)

        if test_error_rate < min_test_error_rate:
            min_test_error_rate = test_error_rate
            best_k = k
            best_p = p

print("Misclassification Error Rate on Training Set:", train_error_rate)
print("Misclassification Error Rate on Test Set:", min_test_error_rate)
print("Best k:", best_k)
print("Best p:", best_p)
