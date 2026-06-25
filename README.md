# AI Food Donation Network

A Flask + Machine Learning web application that enables donors to submit surplus food,
tracks donations in a SQLite database, provides a live management dashboard, and uses
a trained **Random Forest Classifier** to predict whether donated food is safe for distribution.

---

## Project Structure

```
AI food donation network/
├── app.py                      ← Flask app (routes, DB, ML prediction)
├── train_model.py              ← Model training script
├── generate_dataset.py         ← Synthetic dataset generator
├── food_quality_dataset.csv    ← 5 200-record training dataset
├── food_quality_model.pkl      ← Trained RandomForestClassifier
├── label_encoder.pkl           ← LabelEncoder for Food_Type column
├── requirements.txt            ← Python dependencies
├── README.md
├── instance/
│   └── donations.db            ← SQLite database (auto-created)
├── static/
│   ├── css/
│   │   └── style.css           ← Custom stylesheet
│   └── js/
│       └── main.js             ← Client-side helpers
└── templates/
    ├── base.html               ← Master layout (navbar, footer, flash messages)
    ├── index.html              ← Home page
    ├── donate.html             ← Food donation form
    ├── dashboard.html          ← Donation management dashboard
    ├── predict.html            ← AI prediction input form
    └── prediction_result.html  ← AI prediction result page
```

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the ML model

```bash
python train_model.py
```

This will:
- Load `food_quality_dataset.csv` (5 200 records)
- Encode the `Food_Type` column with `LabelEncoder`
- Train a `RandomForestClassifier` (200 trees, balanced class weights)
- Print accuracy, classification report, and confusion matrix
- Save `food_quality_model.pkl` and `label_encoder.pkl`

Expected output:
```
Test Accuracy : ~80%+
```

### 4. Run the Flask application

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## Flask Routes

| Method | Path | Description |
|---|---|---|
| GET | `/` | Home page with live stats |
| GET / POST | `/donate` | Food donation form |
| GET | `/dashboard` | All donations table |
| POST | `/update_status/<id>` | Change donation status |
| POST | `/delete/<id>` | Delete a donation |
| GET / POST | `/predict` | **AI food safety prediction** |

---

## AI Prediction Page

Navigate to **AI Prediction** in the navbar (or `/predict`).

Enter:
- **Food Type** — Rice, Curry, Bread, Fruits, Vegetables, Dairy Products, Fast Food
- **Quantity** — kg or portions
- **Preparation Time** — minutes
- **Storage Duration** — hours since preparation
- **Temperature** — current storage temperature in °C

The Random Forest model returns:
- **Safe for Donation** ✅ or **Unsafe for Donation** ❌
- **Confidence %** (how many of the 200 trees agreed)
- **Recommendation** message based on confidence band

---

## Machine Learning Details

| Property | Value |
|---|---|
| Algorithm | RandomForestClassifier |
| Estimators | 200 trees |
| Class weight | balanced |
| Features | Food_Type (encoded), Quantity, Preparation_Time, Storage_Hours, Temperature |
| Target | Food_Safe (Yes = 1, No = 0) |
| Train / Test split | 80% / 20% |
| Test accuracy | ~80% |

### Feature Importance (approximate)

| Feature | Importance |
|---|---|
| Temperature | ~32% |
| Storage_Hours | ~26% |
| Preparation_Time | ~19% |
| Quantity | ~12% |
| Food_Type | ~11% |

---

## Dataset

`food_quality_dataset.csv` — 5 200 synthetic records generated with realistic
per-food-type safety rules (temperature bands, storage windows, prep times).

To regenerate:
```bash
python generate_dataset.py
```

---

## Requirements

```
flask>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

---

## SDG Alignment

| Goal | Contribution |
|---|---|
| **SDG 2 — Zero Hunger** | Safely routes surplus food to those in need |
| **SDG 3 — Good Health** | Prevents unsafe food from reaching vulnerable people |
| **SDG 12 — Responsible Consumption** | Reduces food waste |
| **SDG 17 — Partnerships** | Connects donors, charities, and volunteers |

---

## License

MIT
