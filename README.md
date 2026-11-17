# 🏠 Multimodal House Price Estimation (Visual + Textual Features)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multimodal-house-price-estimation-vgpdojkibbmqdey452dmgt.streamlit.app/)

🌐 **Live Deployed Web App**: **[https://multimodal-house-price-estimation-vgpdojkibbmqdey452dmgt.streamlit.app/](https://multimodal-house-price-estimation-vgpdojkibbmqdey452dmgt.streamlit.app/)**

An end-to-end **Multimodal Machine Learning & Computer Vision** project and interactive **Streamlit Web Application** that estimates residential house prices by combining **Visual Features (SIFT Keypoints from 4 room viewpoints)** and **Engineered Tabular Features**.

- **Live Demo**: [Multimodal House Price Estimation on Streamlit Cloud](https://multimodal-house-price-estimation-vgpdojkibbmqdey452dmgt.streamlit.app/)
- **Dataset**: [Houses Dataset (535 Properties with 4 Photos Each)](https://github.com/emanhamed/Houses-dataset)
- **Original Analysis**: See [House Price Estimation.ipynb](House%20Price%20Estimation.ipynb) and [Report.pdf](Report.pdf).

---

## ✨ Features of the Deployed Web App (`app.py`)

1. **🔮 Interactive Multimodal Price Predictor**:
   - **Sample House Explorer**: Pick any pre-loaded property from the dataset to inspect its 4 viewpoints (`Bathroom`, `Bedroom`, `Frontal`, `Kitchen`), compare predicted value against the actual listing price, and view predictions across all 6 trained models side-by-side.
   - **Custom House Upload**: Upload 4 photos of any house (`Bathroom`, `Bedroom`, `Frontal`, `Kitchen`) and enter `Bedrooms`, `Bathrooms`, and `Area (sq. ft.)` to run real-time SIFT feature extraction and multimodal price estimation.
2. **🧩 Computer Vision Pipeline Visualizer**:
   - Displays the `256×256` composite 4-room matrix, highlights detected OpenCV `SIFT` keypoints, and plots the 128-dimensional mean visual descriptor vector.
3. **📈 Exploratory Data Analysis (EDA)**:
   - Interactive Plotly histograms, boxplots by engineered `Area Category`, bivariate scatter plots, and feature correlation heatmaps.
4. **🏆 Model Benchmarks**:
   - Side-by-side comparison of Classic ML (`Linear`, `Ridge`, `Decision Tree`), Advanced ML (`Random Forest`, `XGBoost`, `CatBoost`, `SVR`), and Deep Learning (`Multi-Channel CNN + MLP`).

---

## 🚀 Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit application
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Community Cloud / Hugging Face Spaces

All pre-trained models (`models/trained_models.joblib`), dataset summaries, and sample property images (`sample_houses/`) are pre-packaged and lightweight (~3.5 MB total), so the app starts instantaneously in the cloud without needing to download the raw 2,140-image dataset:

1. Push `app.py`, `requirements.txt`, `models/`, and `sample_houses/` to your GitHub repository (`Sadeea2005/Multimodal-House-Price-Estimation`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select repository `Sadeea2005/Multimodal-House-Price-Estimation`, branch `main`, and main file path `app.py`.
4. Click **Deploy!**
