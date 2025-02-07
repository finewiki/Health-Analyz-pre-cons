import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Read dataset
health_data = pd.read_csv('health_data.csv')  # Yeni veri setinizi buraya ekleyin

# Create your map
plt.figure(figsize=(12,10))
p = sns.heatmap(health_data.corr(), annot=True, cmap='RdYlGn')
plt.show()

# Get information about the dataset
health_data.info(verbose=True)
print(health_data.describe())
print(health_data.shape)
print(health_data.groupby('HealthStatus').size())

# Visualize your health status
p = health_data.HealthStatus.value_counts().plot(kind="bar")
plt.show()

# Separate the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(health_data.loc[:, health_data.columns != 'HealthStatus'], 
                                                    health_data['HealthStatus'], 
                                                    stratify=health_data['HealthStatus'], 
                                                    random_state=66)

# Train and test and improve the KNN model
training_accuracy = []
test_accuracy = []
knneighbors_settings = range(1, 11)

for n_neighbors in knneighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

plt.title("Başarı Analizi")
plt.plot(knneighbors_settings, training_accuracy, label="Öğrenim Seti")
plt.plot(knneighbors_settings, test_accuracy, label="Test Seti")
plt.ylabel("Doğruluk yüzdemiz")
plt.xlabel("Komşu sayımız")
plt.legend()
plt.show()

# Choose the best number of neighbors and train the model
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

print('Eğitim setinde K-NN sınıflandırıcısının doğruluk oranı: {:.2f}'.format(knn.score(X_train, y_train)))
print('Test setinde K-NN sınıflandırıcısının doğruluk oranı: {:.2f}'.format(knn.score(X_test, y_test)))

# Create the confusion matrix
y_pred = knn.predict(X_test)
confusion_matrix(y_test, y_pred)
print(pd.crosstab(y_test, y_pred, rownames=['Dogruluk'], colnames=['Tahmin'], margins=True))

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.title('Karışıklık Matrisi', y=1.1)
plt.ylabel('Gerçekleşen')
plt.xlabel('Tahmin')
plt.show()

print(classification_report(y_test, y_pred))