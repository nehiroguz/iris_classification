# Iris Classification
## English
This project aims to create a classification model using the Iris dataset. Logistic regression is used to predict the species of flowers.
### Requirements
- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

To install the requried packages:
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Setup
1. Clone the project repository:
```Bash
git clone <repository-url>
cd iris_classification
```

2. Install the required libraries:
```Bash
pip install -r requirements.txt
```

### Usage
Run the main script to train and test the model:
```Bash
python main.py
```

### Code Explanation
#### Loading Data
```Python
from sklearn import datasets
import pandas as pd

#load the Iris dataset
iris = datasets.load_iris()
data = pd.DataFrame(data= iris.data, columns= iris.feature_names)
data['target'] = iris.target

```

This code loads the Iris dataset from scikit-learn and converts it into a Pandas DataFrame.

#### Data Processing
```Python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features and labels
X = data.iloc[:, :-1] # All columns except the last one 
Y = data['target']

#Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state=42)

#Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test)

```
This code splits the data into training and testing sets and standardizes the features.

#### Model Training and Prediction
```Python 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create and train the logistic regression model
lr = LogisticRegression() 
lr.fit(x_train, y_train) 

# Make predictions on the test data 
predictions = lr.predict(x_test)

# Accuracy score
accuracy = accuracy_score(y_test, predictions)

#Confusion matrix
cm = confusion_matrix(y_test, predictions)

# Clssification report
report = classification_report(y_test, predictions)

print(f'Accuracy score: {accuracy}')
print(f'Confusion matrix:\n {cm}')
print(f'Clssification report:\n {report}')
```
This code trains a logistic regression model, makes predictions on the test data, and evaluates the model's performance.

### Contributors
Contributors to this Projects
Elif Nehir OĞUZ

### License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---------------------------------------------------------------------------------------------------------------------------------------------------
## Türkçe

Bu proje, İris veri setini kullanarak bir sınıflandırma modeli oluşturmayı amaçlar. Lojistik regresyon kullanarak çiçek türlerinin tahmini yapılır.

### Gereksinimler
- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Gereksinimleri yüklemek için:
```Bash
pip install numpy pandas matplotlib scikit-learn
```
### Kurulum
1. Proje dosyalarını indirin:
```Bash
git clone <repository-url>
cd iris_classification
```

2. Gereken kütüphaneleri yükleyin:
```Bash
pip install -r requirements.txt
```

### Kullanım
Projenin ana dosyasını çalıştırarak modeli eğitip test edebilirsiniz:
```Bash
python main.py
```
### Kod Açıklaması
#### Veri Yükleme
```Python
from sklearn import datasets
import pandas as pd

#İris veri setini yükle
iris = datasets.load_iris()
data = pd.DataFrame(data= iris.data, columns= iris.feature_names)
data['target'] = iris.target

```

Bu kod, iris veri setini scikit-learn kütüphanesinden yükler ve bir Pandas DataFrame'e dönüştürür.

#### Veriyi İşleme
```Python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Özellikler ve etiketler
X = data.iloc[:, :-1] # Son sütun hariç tüm sütunları seçer
Y = data['target']

# Eğiyim ve test verilerini ayır
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state=42)

# Özellikleri ölçeklendir
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test)

```
Bu kod, veriyi eğitim ve test setlerine ayırır ve özellikleri standartlaştırır. 

#### Model Eğitimi ve Tahmin
```Python 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Lojistik regresyon modelini oluştur ve eğit
lr = LogisticRegression() 
lr.fit(x_train, y_train) 

# Test verisi üzerinde tahmin yap
predictions = lr.predict(x_test)

# Doğruluk skoru
accuracy = accuracy_score(y_test, predictions)

# Karışıklık matrisi
cm = confusion_matrix(y_test, predictions)

# Sınıflandırma raporu
report = classification_report(y_test, predictions)

print(f'Accuracy score: {accuracy}')
print(f'Confusion matrix:\n {cm}')
print(f'Clssification report:\n {report}')
```

Bu kod, lojistik regresyon modelini eğitir, test verisi üzerinde tahminler yapar ve performansı değerlendirir.

### Katkıda Bulunanlar
Bu projede katkıda bulunanlar:
- Elif Nehir OĞUZ

### Lisans
Bu proje [MIT Lisansı](https://opensource.org/licenses/MIT) altında lisanslanmıştır.

### NOT
Bu örnekte, README dosyasını hem İngilizce hem de Türkçe olarak iki dilde hazırladım. Bölümler arasında ayırıcı çizgiler ('-------') kullanarak iki dili ayırt edebilirsiniz.
