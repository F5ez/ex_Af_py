from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

# Загружаем модель, энкодеры и энкодер цели
model, label_encoders, target_encoder = joblib.load("structure_predictor_model.pkl")

# Список нужных признаков
expected_columns = [
    'insert_freq',
    'delete_freq',
    'search_freq',
    'need_order',
    'unique_keys',
    'access_by_key'
]

app = Flask(__name__)
CORS(app)
print("📦 Label encoders keys:", list(label_encoders.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json

    # Преобразуем JSON в DataFrame
    df = pd.DataFrame([input_data])
    print("✅ DataFrame to predict:\n", df)

    # Преобразуем булевы значения в int
    df['need_order'] = df['need_order'].astype(int)
    df['unique_keys'] = df['unique_keys'].astype(int)

    # Кодируем access_by_key вручную
    access_map = {'low': 0, 'medium': 1, 'high': 2}
    df['access_by_key'] = df['access_by_key'].map(access_map)
    if df['access_by_key'].isnull().any():
        return jsonify({"error": "Invalid value in 'access_by_key'"}), 400

    # Проверим наличие всех колонок
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    # Кодируем строковые признаки
    for col in label_encoders:
        if col in df.columns:
            encoder = label_encoders[col]
            try:
                df[col] = encoder.transform(df[col])
            except Exception as e:
                print(f"❌ Ошибка кодирования '{col}':", e)
                return jsonify({"error": f"Encoding failed for '{col}': {str(e)}"}), 400

    print("✅ Encoded DataFrame:\n", df)

    # Предсказание
    try:
        probs = model.predict_proba(df)[0]
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    class_probs = {
        target_encoder.inverse_transform([i])[0]: float(probs[i])
        for i in range(len(probs))
    }

    sorted_structures = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)

    result = {
        "best_structure": sorted_structures[0][0],
        "alternatives": [
            {"structure": name, "confidence": round(score, 2)}
            for name, score in sorted_structures[:3]
        ]
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
