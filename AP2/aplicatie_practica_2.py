import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('tourism_dataset.csv')

print(data.head())
print(data.info())

selected_country = "USA" 
data_country = data[data['Country'] == selected_country]

if data_country.empty:
    raise ValueError(f"Nu exista date pentru tara selectata: {selected_country}")

data_country = data_country[['Category', 'Revenue', 'Visitors']]

data_country['Revenue_per_Visitor'] = data_country['Revenue'] / data_country['Visitors']
data_country = pd.get_dummies(data_country, columns=['Category'], drop_first=True)
data_country.columns = [col.replace('Category_', '') for col in data_country.columns]

X = data_country.drop(['Revenue', 'Revenue_per_Visitor'], axis=1)
y = data_country['Revenue'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

print(f"Random Forest - Mean Squared Error: {rf_mse}")
print(f"Random Forest - R² Score: {rf_r2}")

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_y_pred = knn_model.predict(X_test)

knn_mse = mean_squared_error(y_test, knn_y_pred)
knn_r2 = r2_score(y_test, knn_y_pred)

print(f"K-Nearest Neighbors - Mean Squared Error: {knn_mse}")
print(f"K-Nearest Neighbors - R² Score: {knn_r2}")

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

categories = [col for col in X.columns if col in ['Nature', 'Historical', 'Cultural', 'Beach', 'Adventure', 'Urban']]
ranking = feature_importances[feature_importances['Feature'].isin(categories)]
ranking = ranking.sort_values(by='Importance', ascending=False)

print("Ranking al categoriilor tematice pentru tara selectata (Random Forest):")
print(ranking)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=ranking)
plt.title(f'Importanta categoriilor tematice pentru maximizarea veniturilor in {selected_country}')
plt.xlabel('Importanta')
plt.ylabel('Categorie')
plt.tight_layout()
plt.show()

models_results = {
    'Random Forest': {'MSE': rf_mse, 'R²': rf_r2},
    'K-Nearest Neighbors': {'MSE': knn_mse, 'R²': knn_r2}
}

results_df = pd.DataFrame(models_results).T
print("\nComparatia modelelor:")
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index, y='R²', data=results_df.reset_index())
plt.title('Comparatia performantelor modelelor (R² Score)')
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.tight_layout()
plt.show()
