import os
import json
import joblib
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Multimodal House Price Estimation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for cards and metrics
st.markdown("""
<style>
    .price-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 24px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        margin-bottom: 16px;
    }
    .price-card h2 {
        margin: 0;
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
    }
    .price-card p {
        margin: 4px 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    .sub-metric {
        background: #f8f9fa;
        border-left: 4px solid #2a5298;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_houses")

@st.cache_resource
def load_artifacts():
    models = joblib.load(os.path.join(MODELS_DIR, "trained_models.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics_summary.json"), "r") as f:
        metrics = json.load(f)
    with open(os.path.join(MODELS_DIR, "samples_meta.json"), "r") as f:
        samples = json.load(f)
    df_engineered = pd.read_csv(os.path.join(MODELS_DIR, "houses_engineered.csv"))
    df_raw = pd.read_csv(os.path.join(MODELS_DIR, "houses_raw.csv"))
    return models, metrics, samples, df_engineered, df_raw

def stitch_four_images(img_dict, size=256):
    """
    Replicates the exact notebook img_read() pipeline:
    Sorted order of filenames: [bathroom, bedroom, frontal, kitchen]
    Top-Left: bathroom
    Top-Right: bedroom
    Bottom-Right: frontal
    Bottom-Left: kitchen
    """
    half = size // 2
    order = ['bathroom', 'bedroom', 'frontal', 'kitchen']
    resized = []
    for k in order:
        img_bgr = img_dict[k]
        resized.append(cv2.resize(img_bgr, (half, half)))
    output = np.zeros((size, size, 3), dtype='uint8')
    output[0:half, 0:half] = resized[0]
    output[0:half, half:size] = resized[1]
    output[half:size, half:size] = resized[2]
    output[half:size, 0:half] = resized[3]
    return output

def extract_sift_features(stitched_bgr):
    """
    Replicates the exact notebook SIFT() feature extraction:
    - Converts BGR image to Grayscale (via RGB2GRAY as in notebook)
    - Detects SIFT keypoints and 128-dim descriptors
    - Returns mean descriptor vector (128,) and keypoint visualization image
    """
    gray_image = cv2.cvtColor(stitched_bgr, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    kp_image = cv2.drawKeypoints(stitched_bgr, keypoints, None)
    kp_image_rgb = cv2.cvtColor(kp_image, cv2.COLOR_BGR2RGB)
    if descriptors is None or len(descriptors) == 0:
        mean_array = np.zeros(128, dtype=np.float32)
    else:
        mean_array = np.mean(descriptors, axis=0).ravel()
    return mean_array, kp_image_rgb, len(keypoints)

def engineer_tabular_features(bedrooms, bathrooms, area):
    total_rooms = bedrooms + bathrooms + 1
    avg_room_area = area // total_rooms if total_rooms > 0 else area
    if area < 2000:
        area_cat_label = "Very Small"
        area_cat_code = 1
    elif area < 4000:
        area_cat_label = "Small"
        area_cat_code = 2
    elif area < 6000:
        area_cat_label = "Medium"
        area_cat_code = 3
    elif area < 8000:
        area_cat_label = "Large"
        area_cat_code = 4
    else:
        area_cat_label = "Very Large"
        area_cat_code = 5

    log_area = np.log1p(area)
    log_avg_room_area = np.log1p(avg_room_area)
    tab_vector = np.array([bedrooms, bathrooms, log_area, total_rooms, log_avg_room_area, area_cat_code], dtype=np.float64)
    info = {
        "total_rooms": total_rooms,
        "avg_room_area": avg_room_area,
        "area_category": area_cat_label,
        "area_category_code": area_cat_code,
        "log_area": round(float(log_area), 4),
        "log_avg_room_area": round(float(log_avg_room_area), 4)
    }
    return tab_vector, info

# Header
st.title("🏠 Multimodal House Price Estimation")
st.markdown(
    "Estimate residential property prices by fusing **Visual Features (SIFT Computer Vision Keypoints)** from 4 house viewpoints "
    "(`Bathroom`, `Bedroom`, `Frontal`, `Kitchen`) with **Engineered Tabular Features** (`Bedrooms`, `Bathrooms`, `Area`, `Total Rooms`, `Avg Room Area`)."
)

if not os.path.exists(os.path.join(MODELS_DIR, "trained_models.joblib")):
    st.warning("Models are not trained yet. Running `train_models.py` first...")
    st.stop()

models, metrics, samples, df_engineered, df_raw = load_artifacts()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Model & Input Settings")
    selected_model_name = st.selectbox(
        "Primary Prediction Model",
        options=list(models.keys()),
        index=0,
        help="Random Forest Regressor achieved the best R² (0.5836) and lowest MAE in the notebook."
    )
    st.divider()
    input_mode = st.radio(
        "Property Input Mode",
        options=["🏘️ Select Sample House from Dataset", "📤 Upload Custom House Photos"],
        index=0
    )
    st.divider()
    st.markdown("### 📊 Dataset Overview")
    st.caption(f"Total Properties: **{len(df_raw)}** houses")
    st.caption(f"Price Range: **${df_raw['price'].min():,.0f} – ${df_raw['price'].max():,.0f}**")
    st.caption("Visual Descriptor: **128-D SIFT Vector**")
    st.caption("Tabular Features: **6 Engineered Features**")

tab_predict, tab_eda, tab_benchmarks = st.tabs([
    "🔮 Multimodal Price Predictor",
    "📈 Exploratory Data Analysis (EDA)",
    "🏆 Model Benchmarks & Architecture"
])

with tab_predict:
    col_inputs, col_visuals = st.columns([1, 1.3], gap="large")

    img_dict = {}
    actual_price = None

    with col_inputs:
        st.subheader("1️⃣ Property Specifications")
        if input_mode == "🏘️ Select Sample House from Dataset":
            sample_options = {
                f"House #{s['house_id']} — {s['bedrooms']} Bed, {s['bathrooms']} Bath, {s['area']:,} sqft (${s['actual_price']:,.0f})": s
                for s in samples
            }
            chosen_label = st.selectbox("Choose a Sample House", options=list(sample_options.keys()))
            chosen_sample = sample_options[chosen_label]
            actual_price = chosen_sample["actual_price"]
            house_id = chosen_sample["house_id"]

            c1, c2 = st.columns(2)
            with c1:
                bedrooms = st.number_input("Bedrooms", min_value=1, max_value=15, value=int(chosen_sample["bedrooms"]), step=1)
                bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=float(chosen_sample["bathrooms"]), step=0.5)
            with c2:
                area = st.number_input("Living Area (sq. ft.)", min_value=300, max_value=20000, value=int(chosen_sample["area"]), step=50)
                zipcode = st.text_input("Zipcode (Informational)", value=str(chosen_sample["zipcode"]), disabled=True)

            for part in ['bathroom', 'bedroom', 'frontal', 'kitchen']:
                img_path = os.path.join(SAMPLE_DIR, f"{house_id}_{part}.jpg")
                img_dict[part] = cv2.imread(img_path)

        else:
            c1, c2 = st.columns(2)
            with c1:
                bedrooms = st.number_input("Bedrooms", min_value=1, max_value=15, value=4, step=1)
                bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
            with c2:
                area = st.number_input("Living Area (sq. ft.)", min_value=300, max_value=20000, value=3500, step=50)

            st.markdown("##### Upload 4 Viewpoints (or leave blank to use default house photos)")
            u1, u2 = st.columns(2)
            with u1:
                up_bath = st.file_uploader("Bathroom Photo", type=["jpg", "jpeg", "png"], key="u_bath")
                up_bed = st.file_uploader("Bedroom Photo", type=["jpg", "jpeg", "png"], key="u_bed")
            with u2:
                up_front = st.file_uploader("Frontal Photo", type=["jpg", "jpeg", "png"], key="u_front")
                up_kitch = st.file_uploader("Kitchen Photo", type=["jpg", "jpeg", "png"], key="u_kitch")

            uploads = {'bathroom': up_bath, 'bedroom': up_bed, 'frontal': up_front, 'kitchen': up_kitch}
            for part, up_file in uploads.items():
                if up_file is not None:
                    file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
                    img_dict[part] = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                else:
                    fallback_path = os.path.join(SAMPLE_DIR, f"1_{part}.jpg")
                    img_dict[part] = cv2.imread(fallback_path)

        # Feature engineering preview
        tab_vector, eng_info = engineer_tabular_features(bedrooms, bathrooms, area)
        st.markdown("#### 🛠️ Engineered Tabular Features")
        fcol1, fcol2, fcol3 = st.columns(3)
        fcol1.metric("Total Rooms (+Kitchen)", f"{eng_info['total_rooms']:g}")
        fcol2.metric("Avg Room Area", f"{eng_info['avg_room_area']:,.0f} sqft")
        fcol3.metric("Area Category", eng_info["area_category"])

    # Process images & run inference
    stitched_bgr = stitch_four_images(img_dict, size=256)
    sift_vec, kp_image_rgb, num_keypoints = extract_sift_features(stitched_bgr)
    full_features = np.hstack([sift_vec, tab_vector]).reshape(1, -1)

    primary_model = models[selected_model_name]
    pred_log = float(primary_model.predict(full_features)[0])
    pred_price = float(np.expm1(np.clip(pred_log, 0, 20)))

    with col_inputs:
        st.divider()
        st.markdown(f"""
        <div class="price-card">
            <p>Estimated Market Value ({selected_model_name})</p>
            <h2>${pred_price:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)

        if actual_price is not None:
            diff = pred_price - actual_price
            pct_err = abs(diff) / actual_price * 100
            m1, m2 = st.columns(2)
            m1.metric("Actual Dataset Price", f"${actual_price:,.0f}")
            m2.metric("Prediction Difference", f"${diff:+,.0f}", f"{pct_err:.1f}% error", delta_color="inverse")

        # Multi-model consensus expander
        with st.expander("📊 Compare Predictions Across All Trained Models", expanded=True):
            all_preds = []
            for m_name, m_obj in models.items():
                p_log = float(m_obj.predict(full_features)[0])
                p_val = float(np.expm1(np.clip(p_log, 0, 20)))
                all_preds.append({
                    "Model": m_name,
                    "Estimated Price ($)": f"${p_val:,.0f}",
                    "R² Score (Test)": metrics[m_name]["R2"]
                })
            st.dataframe(pd.DataFrame(all_preds), use_container_width=True, hide_index=True)

    with col_visuals:
        st.subheader("2️⃣ Visual Pipeline & SIFT Keypoint Extraction")
        vtab1, vtab2, vtab3 = st.tabs([
            "🖼️ 4 Property Viewpoints",
            "🧩 256×256 Stitched Matrix & SIFT Keypoints",
            "📊 128-D SIFT Descriptor Profile"
        ])

        with vtab1:
            g1, g2 = st.columns(2)
            with g1:
                st.image(cv2.cvtColor(img_dict['bathroom'], cv2.COLOR_BGR2RGB), caption="Top-Left: Bathroom", use_container_width=True)
                st.image(cv2.cvtColor(img_dict['kitchen'], cv2.COLOR_BGR2RGB), caption="Bottom-Left: Kitchen", use_container_width=True)
            with g2:
                st.image(cv2.cvtColor(img_dict['bedroom'], cv2.COLOR_BGR2RGB), caption="Top-Right: Bedroom", use_container_width=True)
                st.image(cv2.cvtColor(img_dict['frontal'], cv2.COLOR_BGR2RGB), caption="Bottom-Right: Frontal", use_container_width=True)

        with vtab2:
            sc1, sc2 = st.columns(2)
            with sc1:
                st.image(cv2.cvtColor(stitched_bgr, cv2.COLOR_BGR2RGB), caption="Combined 256×256 Visual Matrix", use_container_width=True)
            with sc2:
                st.image(kp_image_rgb, caption=f"SIFT Feature Keypoints ({num_keypoints} detected)", use_container_width=True)
            st.info(
                f"**Visual Feature Extraction:** OpenCV `SIFT_create()` detected **{num_keypoints}** scale-invariant keypoints "
                f"across the 4 rooms and aggregated them into a **128-dimensional mean visual descriptor vector**."
            )

        with vtab3:
            fig_sift = px.bar(
                x=list(range(1, 129)),
                y=sift_vec,
                labels={"x": "SIFT Descriptor Dimension (1–128)", "y": "Mean Magnitude"},
                title="128-D Mean SIFT Feature Vector Extracted from House Image",
                color=sift_vec,
                color_continuous_scale="Blues"
            )
            fig_sift.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_sift, use_container_width=True)

with tab_eda:
    st.subheader("📈 Exploratory Data Analysis & Feature Correlations")
    st.markdown("Interactive visualizations reproducing the notebook's univariate, bivariate, and correlation analyses across all 535 houses.")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig_price = px.histogram(
            df_engineered, x="price", nbins=30,
            title="House Price Distribution (Target)",
            labels={"price": "Price ($)"},
            color_discrete_sequence=["#2a5298"]
        )
        st.plotly_chart(fig_price, use_container_width=True)

    with r1c2:
        fig_box = px.box(
            df_engineered, x="area category", y="price",
            category_orders={"area category": ["Very Small", "Small", "Medium", "Large", "Very Large"]},
            title="House Price vs. Engineered Area Category",
            color="area category",
            labels={"price": "Price ($)", "area category": "Area Category"}
        )
        st.plotly_chart(fig_box, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        scatter_col = st.selectbox("Select Feature for Bivariate Scatter vs. Price", ["area", "bathrooms", "bedrooms", "total rooms", "avg_room_area"])
        fig_scatter = px.scatter(
            df_engineered, x=scatter_col, y="price",
            color="area category",
            title=f"Price vs. {scatter_col.title()}",
            opacity=0.75
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with r2c2:
        num_cols = ["bedrooms", "bathrooms", "area", "total rooms", "avg_room_area", "price"]
        corr_matrix = df_engineered[num_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Feature Correlation Heatmap"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

with tab_benchmarks:
    st.subheader("🏆 Model Performance Benchmarks (Classic ML, Advanced ML & Deep Learning)")
    metrics_df = pd.DataFrame([
        {
            "Model": k,
            "Category": v["Category"],
            "MAE (Log Scale)": v["MAE_log"],
            "R² Score": v["R2"],
            "Approx. Dollar MAE ($)": f"${v['MAE_dollar']:,.0f}"
        }
        for k, v in metrics.items()
    ]).sort_values(by="R² Score", ascending=False)

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        fig_r2 = px.bar(
            metrics_df, x="Model", y="R² Score", color="Category",
            title="Model Comparison by R² Score (Higher is Better)",
            text_auto=".3f"
        )
        fig_r2.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_r2, use_container_width=True)

    with bcol2:
        fig_mae = px.bar(
            metrics_df.sort_values(by="MAE (Log Scale)", ascending=True),
            x="Model", y="MAE (Log Scale)", color="Category",
            title="Model Comparison by Mean Absolute Error (Lower is Better)",
            text_auto=".3f"
        )
        fig_mae.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_mae, use_container_width=True)

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("""
    ### 🧠 Architecture Notes from Notebook
    1. **Visual Feature Engineering (`SIFT`)**: Each house's 4 photos (`bathroom`, `bedroom`, `frontal`, `kitchen`) are resized to `128×128` and tiled into a `256×256` composite image. Scale-Invariant Feature Transform (`SIFT`) keypoints and descriptors are extracted and averaged into a `128-D` visual vector.
    2. **Textual/Tabular Feature Engineering**: Adds `total rooms = bedrooms + bathrooms + 1` (for kitchen), `avg_room_area = area // total rooms`, and ordinal `area category`, followed by `np.log1p` transformation on skewed area features and target price.
    3. **Multi-Channel Neural Network**: Combines a 3-block Conv2D + BatchNorm + MaxPool CNN branch (on `64×64×3` composite images) with a 3-layer MLP branch on standardized tabular features.
    """)
