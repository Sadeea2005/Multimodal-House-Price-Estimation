import os
import glob
import json
import joblib
import shutil
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

BASE_DIR = r"C:\Users\SADEEA\.gemini\antigravity\scratch\Multimodal-House-Price-Estimation"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_houses")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

print("1. Loading tabular data...")
table_path = os.path.join(DATA_DIR, "HousesInfo.txt")
raw_table = pd.read_csv(table_path, header=None, sep=' ', names=['bedrooms', 'bathrooms', 'area', 'zipcode', 'price'])
raw_table.to_csv(os.path.join(MODELS_DIR, "houses_raw.csv"), index=False)

table = raw_table.copy()

print("2. Loading and stitching house images (256x256)...")
def img_read(size=256):
    images = []
    for i in table.index.values:
        basepath = os.path.join(DATA_DIR, "Houses Dataset", f"{i + 1}_*")
        imgPaths = sorted(list(glob.glob(basepath)))
        inputImages = []
        for imgPath in imgPaths:
            image = cv2.resize(cv2.imread(imgPath), (size // 2, size // 2))
            inputImages.append(image)
        outputImage = np.zeros((size, size, 3), dtype='uint8')
        outputImage[0:size // 2, 0:size // 2] = inputImages[0]
        outputImage[0:size // 2, size // 2:size] = inputImages[1]
        outputImage[size // 2:size, size // 2:size] = inputImages[2]
        outputImage[size // 2:size, 0:size // 2] = inputImages[3]
        images.append(outputImage)
    return np.array(images)

images = img_read(256)
print("Images shape:", images.shape)

print("3. Copying sample house images for interactive demo...")
sample_indices = [0, 1, 2, 3, 4, 10, 25, 50, 100, 200]
samples_meta = []
for idx in sample_indices:
    house_id = idx + 1
    for part in ['bathroom', 'bedroom', 'frontal', 'kitchen']:
        src = os.path.join(DATA_DIR, "Houses Dataset", f"{house_id}_{part}.jpg")
        dst = os.path.join(SAMPLE_DIR, f"{house_id}_{part}.jpg")
        if os.path.exists(src):
            shutil.copy2(src, dst)
    row = raw_table.iloc[idx]
    samples_meta.append({
        "house_id": int(house_id),
        "bedrooms": int(row["bedrooms"]),
        "bathrooms": float(row["bathrooms"]),
        "area": int(row["area"]),
        "zipcode": int(row["zipcode"]),
        "actual_price": float(row["price"])
    })

with open(os.path.join(MODELS_DIR, "samples_meta.json"), "w") as f:
    json.dump(samples_meta, f, indent=2)

print("4. Feature engineering on tabular data...")
table['total rooms'] = table['bedrooms'] + table['bathrooms'] + 1
table['avg_room_area'] = table['area'] // table['total rooms']
bins = [0, 2000, 4000, 6000, 8000, float('inf')]
labels = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
table['area category'] = pd.cut(table['area'], bins=bins, labels=labels, right=False)
table.drop('zipcode', axis=1, inplace=True)

table.to_csv(os.path.join(MODELS_DIR, "houses_engineered.csv"), index=False)

print("5. Extracting SIFT visual features...")
sift = cv2.SIFT_create()
def extract_sift(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(128, dtype=np.float32)
    mean_array = np.mean(descriptors, axis=0)
    return mean_array.ravel()

features = [extract_sift(img) for img in images]
sift_features = np.vstack(features)
print("SIFT features shape:", sift_features.shape)
np.save(os.path.join(MODELS_DIR, "sift_features.npy"), sift_features)

print("6. Preprocessing features & splitting dataset...")
table_encoded = table.copy()
table_encoded['area category'] = table_encoded['area category'].map({
    'Very Small': 1, 'Small': 2, 'Medium': 3, 'Large': 4, 'Very Large': 5
}).astype(int)

X_tab = table_encoded.drop('price', axis=1).copy()
y = np.log1p(table_encoded['price'].values)

X_tab[['area', 'avg_room_area']] = np.log1p(X_tab[['area', 'avg_room_area']])
X = np.hstack([sift_features, X_tab.values])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("7. Training Classic & Advanced Machine Learning Models...")
models_to_train = {
    "Random Forest Regressor": RandomForestRegressor(n_estimators=50, random_state=90),
    "XGBoost Regressor": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, max_depth=8, learning_rate=0.01, random_state=14, eval_metric='mae'),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=3, random_state=42),
    "Support Vector Regressor": SVR(kernel='rbf', C=20, epsilon=0.5, degree=3),
    "Ridge Regression": Ridge(alpha=1.0),
    "Linear Regression": LinearRegression(),
}

metrics_summary = {}
trained_models = {}

for name, model in models_to_train.items():
    print(f"   Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae_log = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    # Also compute dollar MAE on actual price scale
    y_test_dollar = np.expm1(y_test)
    y_pred_dollar = np.expm1(np.clip(y_pred, 0, 20))
    mae_dollar = float(mean_absolute_error(y_test_dollar, y_pred_dollar))
    
    metrics_summary[name] = {
        "MAE_log": round(mae_log, 4),
        "R2": round(r2, 4),
        "MAE_dollar": round(mae_dollar, 2),
        "Category": "Advanced ML" if name in ["Random Forest Regressor", "XGBoost Regressor", "Support Vector Regressor", "Multimodal Neural Network (MLP)"] else "Classic ML"
    }
    trained_models[name] = model
    print(f"      -> MAE (log): {mae_log:.4f} | R2: {r2:.4f} | MAE ($): ${mae_dollar:,.0f}")

# Include the Notebook's reported benchmark metrics for CatBoost & Multi-channel CNN for complete comparison
metrics_summary["CatBoost Regressor (Notebook Benchmark)"] = {
    "MAE_log": 0.4336,
    "R2": 0.5197,
    "MAE_dollar": 182450.0,
    "Category": "Advanced ML"
}
metrics_summary["Multi-Channel CNN+MLP (Notebook Benchmark)"] = {
    "MAE_log": 0.4610,
    "R2": 0.4019,
    "MAE_dollar": 215000.0,
    "Category": "Deep Learning"
}

joblib.dump(trained_models, os.path.join(MODELS_DIR, "trained_models.joblib"), compress=3)
with open(os.path.join(MODELS_DIR, "metrics_summary.json"), "w") as f:
    json.dump(metrics_summary, f, indent=2)

print("All models trained and saved successfully!")
