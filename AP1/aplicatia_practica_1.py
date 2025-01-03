import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error

file_path = 'Grafic_SEN.xlsx'
data = pd.read_excel(file_path)

print(data.info())
print(data.describe())

columns_to_clean = ['Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]', 
                    'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]', 'Sold[MW]']

for col in columns_to_clean:
    data[col] = data[col].replace(r'[^\d.-]', '', regex=True).astype(float)

print(data.describe())

data['Data'] = pd.to_datetime(data['Data'], format='%d-%m-%Y %H:%M:%S', errors='coerce')

data = data.dropna()

print("Distributia lunilor în setul de date:")
print(data['Data'].dt.month.value_counts())

train_data = data[data['Data'].dt.month != 12]
test_data = data[data['Data'].dt.month == 12]

if train_data.empty:
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

features = ['Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 
            'Carbune[MW]', 'Hidrocarburi[MW]', 'Ape[MW]', 
            'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]']
target = 'Sold[MW]'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

id3_model = DecisionTreeRegressor(max_depth=5, random_state=42)
id3_model.fit(X_train, y_train)

y_pred_id3 = id3_model.predict(X_test)

rmse_id3 = np.sqrt(mean_squared_error(y_test, y_pred_id3))
mae_id3 = mean_absolute_error(y_test, y_pred_id3)

print(f"ID3 - RMSE: {rmse_id3:.2f}, MAE: {mae_id3:.2f}")

bins = np.linspace(X_train.min().min(), X_train.max().max(), 10)
X_train_binned = np.digitize(X_train, bins=bins)
X_test_binned = np.digitize(X_test, bins=bins)

bayes_model = GaussianNB()
bayes_model.fit(X_train_binned, y_train)

y_pred_bayes = bayes_model.predict(X_test_binned)

rmse_bayes = np.sqrt(mean_squared_error(y_test, y_pred_bayes))
mae_bayes = mean_absolute_error(y_test, y_pred_bayes)

print(f"Bayesian - RMSE: {rmse_bayes:.2f}, MAE: {mae_bayes:.2f}")

max_sold_december = test_data['Sold[MW]'].max()
max_sold_date = test_data[test_data['Sold[MW]'] == max_sold_december]['Data'].iloc[0]

print(f"Cel mai mare sold pentru luna decembrie 2024 este {max_sold_december} MW, la data {max_sold_date.strftime('%d-%m-%Y %H:%M:%S')}")

plt.figure(figsize=(10, 6))
plt.plot(test_data['Data'], y_test, label='Valori reale', linestyle='-', color='blue')
plt.plot(test_data['Data'], y_pred_id3, label='Predicții ID3', linestyle='-', color='green')
plt.plot(test_data['Data'], y_pred_bayes, label='Predicții Bayesian', linestyle='-', color='orange')
plt.xlabel('Data')
plt.ylabel('Sold [MW]')
plt.legend()
plt.title('Comparație Predicții pentru luna Decembrie 2024')
plt.xticks(rotation=45)
plt.tight_layout()  
plt.show()
