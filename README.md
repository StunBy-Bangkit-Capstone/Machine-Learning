# Machine Learning

This repository contains machine learning models and implementations for the StunBy application, which focuses on combating stunting through innovative technology.

## Table of Contents

- [Baby Height Detection](#baby-height-detection)
- [Nutrition Prediction](#nutrition-prediction)
- [Z-Score Calculation & Classification](#z-score-calculation--classification)
- [Food Recommendation & Tracking](#food-recommendation--tracking)
- [Chatbot with Vertex AI & RAG (Retrieval Augmented Generation)](#chatbot-with-vertex-ai--rag-retrieval-augmented-generation)
- [API Machine Learning Documentation](#api-machine-learning-documentation)
  - [How to Run](#how-to-run)
  - [Measure Classify](#measure-classify)
  - [Predict Nutrition](#predict-nutrition)
  - [Add Food Tracking](#add-food-tracking)
  - [Recommendation Food](#recommendation-food)
- [Contributors](#contributors)

## Computer Vision to Baby Height Measurement Detection

- **Model**: Fine-tuned YOLO on COCO dataset (people and coin classes).
- **Dataset**: [Coco dataset](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml) [Coin dataset roboflow](https://universe.roboflow.com/coinsrec/coin-recognition-u1ebb)
- **Notebook**: [Notebook People detection](https://github.com/StunBy-Bangkit-Capstone/Machine-Learning/blob/main/notebook/Keypoint_YOLOv11.ipynb) [Notebook Coin detection](https://github.com/StunBy-Bangkit-Capstone/Machine-Learning/blob/main/notebook/Coin_YOLOv11.ipynb)
- **Purpose**: Detect baby's height from images with reference to a coin for scale.
- **Input**: Vertical images capturing the baby from head to toe with a coin visible.
- **Output**: Estimated height of the baby in centimeters.
- **Dependencies**:
  - TensorFlow
  - OpenCV
  - ultralytics
  - YOLOv11

## Nutrition Prediction

- **Model**: TensorFlow-based model for predicting nutrient requirements.
- **Dataset**: [WHO child growth standards dataset](https://github.com/StunBy-Bangkit-Capstone/Machine-Learning/blob/main/data/kebutuhan_nutrisi_baby_2th.csv)
- **Notebook**: [Notebook](https://github.com/StunBy-Bangkit-Capstone/Machine-Learning/blob/main/notebook/predict-tracking-nutrition-baby2.ipynb)
- **Purpose**: Calculate the daily caloric, protein, fat, and carbohydrate needs of a baby.
- **Input**: Age, gender, weight, height, activity level, and breastfeeding status.
- **Output**: Recommended daily intake of nutrients.
- **Dependencies**:
  - TensorFlow
  - NumPy
  - Pandas
  - Scikit-learn
- **Model Evaluation**: </br>
  <img width="150" alt="image" src="https://github.com/user-attachments/assets/4f92038b-89b2-4054-8f5a-4494e261e034" />

  <img width="200" alt="image" src="https://github.com/user-attachments/assets/2e5d49c2-b7b5-4e3f-b59d-15d9ab535b73" />

## Z-Score Calculation & Classification

- **Purpose**: Classify the child's growth status based on WHO Z-Score standards.
- **Dataset**: [Kementerian Kesehatan RI](https://github.com/StunBy-Bangkit-Capstone/Machine-Learning/tree/main/data/datasetbaby_klasifikasi_status)
- **Notebook**: [Notebook](https://github.com/StunBy-Bangkit-Capstone/Machine-Learning/blob/main/notebook/NUTRITIONAL%20STATUS%20CLASSIFICATION%20Z-SCORE%20INDEX%20FOR%20INFANT.ipynb)
- **Input**: Baby's height, weight, and age.
- **Output**: Growth classification (e.g., normal, stunted).
- **Dependencies**:
  - NumPy
  - WHO child growth standards dataset

## Food Recommendation & Tracking

- **Model**: Content-Based and Collaborative Filtering.
- **Dataset**: [Baby food dataset](https://github.com/StunBy-Bangkit-Capstone/Machine-Learning/blob/main/data/rekomendasi_data_baby_food.csv)
- **Notebook**: [Notebook](https://github.com/StunBy-Bangkit-Capstone/Machine-Learning/blob/main/notebook/REKOMENDASI_MAKANAN_Content_Based_and_Collaborative_Filtering_LinearProgramming.ipynb)
- **Purpose**: Recommend suitable baby foods and track daily food intake.
- **Input**: Baby's dietary preferences, allergies, and nutrient needs.
- **Output**: Personalized food recommendations and intake tracking.
- **Dependencies**:
  - Scikit-Learn
  - Pandas
  - Surprise (Collaborative Filtering library)

## Chatbot with Vertex AI & RAG (Retrieval Augmented Generation)

- **Model**: Retrieval-Augmented Generation using Vertex AI.
- **Notebook**: [Notebook](https://github.com/StunBy-Bangkit-Capstone/Machine-Learning/blob/main/notebook/Generative-RAG-vertex_bangkit_stunby.ipynb)
- **Purpose**: Provide quick responses to parenting and nutrition-related queries.
- **Input**: User queries about stunting, nutrition, or app features.
- **Output**: Conversational responses enriched with relevant information.
- **Dependencies**:
  - Vertex AI SDK
  - Flask (for API wrapping)

## API Machine Learning Documentation

### How to Run

1. Clone this repository.
2. Go to the `ML_API_Integration` directory.
3. Install Dependencies

```bash
pip install -r requirements.txt
```

4. Run the API

```bash
python app.py
```

### Measure Classify

Untuk mengukur panjang badan dan mengklasifikasikan status gizi berdasarkan panjang badan, berat badan, umur, dan jenis kelamin.

POST http://127.0.0.1:5000/measure-classify
Request:

```json
{
  "url": "https://storage.googleapis.com/stunby_bucket/baby_testing/baby_3.jpeg",
  "weight": 5.5,
  "age": 6,
  "gender": "male"
}
```

Response:

```json
{
  "baby_length": 45.79,
  "imt": 25.99243856332703,
  "nutritional_status_length": "Sangat Pendek", // "Sangat Pendek", "Pendek", "Normal", "Tinggi", "Tidak Diketahui"
  "nutritional_status_weight": "Gizi Kurang", // "Gizi Buruk", "Gizi Kurang", "Gizi Baik", "Gizi Lebih", "Kategori Tidak DIketahui"
  "status_bb_tb": "Gemuk", // "Sangat Kurus", "Kurus", "Normal", "Gemuk", "Kategori Tidak Diketahui"
  "status_imt": "Obesitas (Obese)", // "Gizi Buruk (Severely Wasted)", "Gizi Kurang (Wasted)", "Gizi Baik (Normal)", "Berisiko Gizi Lebih (Possible Risk of Overweight)", "Gizi Lebih (Overweight)", "Obesitas (Obese)", "Kategori Tidak Diketahui"
  "z_score_bb_tb": 9.666666666666671,
  "z_score_length": -10.285714285714311,
  "z_score_weight": -2.999999999999998
}
```

### Predict Nutrition

Untuk memprediksi kebutuhan nutrisi berdasarkan umur, jenis kelamin, berat badan, tinggi badan, tingkat aktivitas, dan status pemberian ASI.

POST 127.0.0.1:5000/predict_nutrition
request:

```json
{
    "usia_bulan": 12,           # int: 0-24 bulan
    "gender": "L",              # str: "L" atau "P"
    "berat_kg": 9.5,           # float: berat dalam kg
    "tinggi_cm": 75.0,         # float: tinggi dalam cm
    "aktivitas_level": "Sedang", # str: "Rendah"/"Sedang"/"Aktif"/"Sangat_Aktif"
    "status_asi": "ASI+MPASI"   # str: "ASI_Eksklusif"/"ASI+MPASI"/"MPASI"
}
```

response:

```json
{
  "status": "success",
  "data": {
    "calories_needed": 850.5, // kalori dalam kkal
    "proteins_needed": 20.1, // protein dalam gram
    "fat_needed": 30.2, // lemak dalam gram
    "carbohydrate_needed": 120.5 // karbohidrat dalam gram
  }
}
```

### Add Food Tracking

Untuk menambahkan makanan yang dikonsumsi oleh anak.

POST http://127.0.0.1:5000/api/tracking/add-food

request:

```json
{
  "food_name": "ASI (Air Susu Ibu)",
  "portion": 200
}
```

response:

```json
{
  "data": {
    "name": "ASI (Air Susu Ibu)",
    "notes": "ASI Eksklusif direkomendasikan oleh WHO sebagai sumber nutrisi eksklusif.",
    "nutrients": {
      "calcium": 68.0,
      "calories": 140.0,
      "carbohydrate": 14.0,
      "fat": 8.4,
      "proteins": 2.4
    },
    "portion": 200.0
  },
  "status": "success"
}
```

### recommendation food

Untuk merekomendasikan makanan yang sesuai dengan kebutuhan nutrisi anak.

POST http://localhost:5000/recommend-food

REQUEST

```json
{
  "age_months": 3,
  "daily_needs": {
    "calorie": 450,
    "protein": 9.5,
    "carb": 31,
    "fat": 48
  },
  "daily_budget": 50000,
  "user_preferences": ["ASI"] // optional, bisa berupa kata kunci yang diinginkan
}
```

```json
{
  "age_months": 7,
  "daily_needs": {
    "calorie": 700.5,
    "protein": 20.1,
    "carb": 120.5,
    "fat": 30.2
  },
  "daily_budget": 30000,
  "user_preferences": ["Susu", "Buah"] // optional, bisa berupa kata kunci yang diinginkan
}
```

RESPONSE

```json
{
  "recommendations": [
    {
      "Banyak_produk": 1.0,
      "Berat_per_Produk(gr)": 150,
      "Harga_per_Porsi(IDR)": 15800.000000000002,
      "Kalori_per_Porsi(kcal)": 108.0,
      "Karbohidrat_per_Porsi(gr)": 12.0,
      "Lemak_per_Porsi(gr)": 5.700000000000001,
      "Nama_Makanan": "Susu Formula Lanjutan 6-12 bulan",
      "Notes": "Contoh merek: Bebelac Gold 2 SGM 2 Dancow 2. Diperkaya dengan zat besi dan vitamin esensial.",
      "Protein_per_Porsi(gr)": 2.6999999999999997,
      "Total_Harga": 15800.000000000002
    },
    {
      "Banyak_produk": 2.0,
      "Berat_per_Produk(gr)": 100,
      "Harga_per_Porsi(IDR)": 8000.0,
      "Kalori_per_Porsi(kcal)": 160.0,
      "Karbohidrat_per_Porsi(gr)": 8.5,
      "Lemak_per_Porsi(gr)": 14.699999999999996,
      "Nama_Makanan": "Pure Alpukat",
      "Notes": "Alpukat matang dilumatkan. Sumber lemak baik untuk perkembangan otak bayi.",
      "Protein_per_Porsi(gr)": 2.0,
      "Total_Harga": 16000.0
    },
    {
      "Banyak_produk": 1.0,
      "Berat_per_Produk(gr)": 500,
      "Harga_per_Porsi(IDR)": 10000.0,
      "Kalori_per_Porsi(kcal)": 384.99999999999994,
      "Karbohidrat_per_Porsi(gr)": 85.0,
      "Lemak_per_Porsi(gr)": 0.5,
      "Nama_Makanan": "Pure Kentang",
      "Notes": "Kentang kukus yang dihaluskan. Sumber karbohidrat yang baik.",
      "Protein_per_Porsi(gr)": 10.0,
      "Total_Harga": 10000.0
    }
  ],
  "summary": {
    "remaining_budget": -11800.0,
    "total_nutrients": {
      "Kalori": 813.0,
      "Karbohidrat": 114.0,
      "Lemak": 35.599999999999994,
      "Protein": 16.7
    }
  }
}
```

```json
{
  "error": "Tidak ada solusi yang ditemukan. Coba lagi dengan budget yang lebih tinggi."
}
```

## Contributors

- [Ridlo Abdullah Ulinnuha](https://github.com/Ridlo543)
- [Ahmad Dzulfikar Ubaidillah](https://github.com/dzulfikarubaid)
- [Azzahra Athifah Dzaki](https://github.com/AzzahraDzaki)
