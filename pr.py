import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Загрузка данных
df = pd.read_csv("ml_structure_training_data_500.csv")

# Целевая переменная
y = df['best_structure']
X = df.drop(columns='best_structure')

# Преобразуем булевы признаки в int
X['need_order'] = X['need_order'].astype(int)
X['unique_keys'] = X['unique_keys'].astype(int)
X['access_by_key'] = X['access_by_key'].astype(int)

# Кодируем категориальные (строковые) признаки
label_encoders = {}
for col in ['insert_freq', 'delete_freq', 'search_freq']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Кодируем целевую переменную
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Делим на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Обучаем модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Выводим точность
print(f"Accuracy: {model.score(X_test, y_test):.2f}")

# Сохраняем всё в один .pkl
with open("structure_predictor_model.pkl", "wb") as f:
    joblib.dump((model, label_encoders, target_encoder), f)
