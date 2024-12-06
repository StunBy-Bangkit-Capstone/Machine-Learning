from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Muat model dan data yang disimpan
with open('food_recommendation_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

df = model_data['df']
similarity_matrix = model_data['similarity_matrix']
scaler = model_data['scaler']

# Fungsi untuk mendapatkan rekomendasi
def get_recommendations(age_months, daily_needs, budget, top_n=5):
    features_to_normalize = ['Calcium(mg)', 'Protein(g)', 'Carbohydrate(g)',
                             'Fat(g)', 'Calorie(kcal)', 'Price_per_product(IDR)']

    # Filter berdasarkan usia
    filtered_df = df[
        (df['Recommended_Age_Start(months)'] <= age_months) &
        (df['Recommended_Age_End(months)'] >= age_months)
    ]

    # Filter berdasarkan budget
    price_index = features_to_normalize.index('Price_per_product(IDR)')
    original_prices = scaler.inverse_transform(filtered_df[features_to_normalize])[:, price_index]
    filtered_df = filtered_df[original_prices <= budget]

    if filtered_df.empty:
        return "Tidak ada makanan yang sesuai dengan kriteria"

    # Normalisasi kebutuhan nutrisi harian
    daily_needs_normalized = {
        'Calcium(mg)': 0,  # nilai default untuk Calcium
        'Protein(g)': daily_needs['protein'],
        'Carbohydrate(g)': daily_needs['carb'],
        'Fat(g)': daily_needs['fat'],
        'Calorie(kcal)': daily_needs['calorie'],
        'Price_per_product(IDR)': 0  # nilai default untuk Price
    }

    daily_needs_df = pd.DataFrame([daily_needs_normalized])[features_to_normalize]
    daily_needs_scaled = scaler.transform(daily_needs_df)

    # Hitung similarity dengan kebutuhan nutrisi
    nutrition_features = ['Calorie(kcal)', 'Protein(g)', 'Carbohydrate(g)', 'Fat(g)']

    # Dapatkan indeks yang sesuai untuk nutrition features
    nutrition_indices = [features_to_normalize.index(feat) for feat in nutrition_features]

    similarities = cosine_similarity(
        filtered_df[nutrition_features],
        daily_needs_scaled[:, nutrition_indices]
    )

    # Dapatkan indeks makanan dengan similarity tertinggi
    similar_indices = similarities.flatten().argsort()[::-1][:top_n]

    # Buat DataFrame hasil rekomendasi
    recommendations = filtered_df.iloc[similar_indices].copy()

    # Denormalisasi nilai untuk ditampilkan
    recommendations[features_to_normalize] = scaler.inverse_transform(
        recommendations[features_to_normalize]
    )

    return recommendations[['Food(per 100g)', 'Calorie(kcal)', 'Protein(g)',
                            'Carbohydrate(g)', 'Fat(g)', 'Price_per_product(IDR)', 'Notes']].to_dict(orient='records')

# Route untuk halaman utama (home)
@app.route('/')
def home():
    return "Fitur Rekomendasi Makanan Bayi StunBy!"

# Endpoint untuk mendapatkan rekomendasi
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Ambil data dari request JSON
        data = request.get_json()

        age_months = data['age_months']
        daily_needs = data['daily_needs']
        budget = data['budget']

        # Panggil fungsi get_recommendations
        recommendations = get_recommendations(age_months, daily_needs, budget, top_n=5)

        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)