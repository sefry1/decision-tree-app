# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib
from modeling import train_and_evaluate, predict_single
from utils import load_data_auto, auto_encode_dataframe, download_joblib

st.set_page_config(page_title="Decision Tree Playground", layout="wide")

st.title("🧠 Decision Tree — Tugas Modul")
st.markdown(
    """
Aplikasi interaktif untuk membangun, mengevaluasi, dan mencoba model Decision Tree.
Gunakan file Excel/CSV kamu (atau gunakan file contoh). Pilih kolom target, atur parameter, lalu latih model.
"""
)

# --- Sidebar: dataset input & settings ---
st.sidebar.header("1) Data")
uploaded = st.sidebar.file_uploader("Upload file .xlsx / .csv (opsional). Jika kosong, pakai contoh.", type=["xlsx", "csv"])
use_example = False
if uploaded is None:
    st.sidebar.info("Tidak ada file. Aplikasi akan mencoba memuat file `BlaBla.xlsx` jika tersedia, atau contoh acak.")
    use_example = st.sidebar.checkbox("Paksa gunakan contoh acak (synthetic)", value=False)

# Load data
df = load_data_auto(uploaded, use_example=use_example)

st.sidebar.markdown("---")
st.sidebar.header("2) Preprocessing")
fill_na_method = st.sidebar.selectbox("Isi nilai kosong dengan", ["mean_for_numeric", "median_for_numeric", "mode_for_all", "constant_empty"], index=0)
encode_categorical = st.sidebar.checkbox("Encode categorical secara otomatis", value=True)

st.sidebar.markdown("---")
st.sidebar.header("3) Training params")
target_col = st.sidebar.selectbox("Pilih kolom target (label)", options=df.columns.tolist())
test_size = st.sidebar.slider("Proporsi test set", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"], index=0)
max_depth = st.sidebar.slider("Max depth (0 = None)", 0, 20, 5)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.header("Run")
run_train = st.sidebar.button("🟢 Train Model")

# --- Main layout: show data & actions ---
with st.expander("Preview Data (first 10 baris)"):
    st.dataframe(df.head(10))

with st.expander("Dataset Summary"):
    if not df.empty:
        st.write("Shape:", df.shape)
        st.write(df.describe(include="all"))
    else:
        st.warning("Dataset masih kosong. Upload file atau pastikan BlaBla.xlsx tersedia.")


# Preprocessing controls & apply
if encode_categorical:
    df_encoded, encode_map = auto_encode_dataframe(df.copy(), strategy=fill_na_method)
else:
    # just fill missing numerics
    df_encoded = df.copy()
    if fill_na_method == "mean_for_numeric":
        for c in df_encoded.select_dtypes(include=[np.number]).columns:
            df_encoded[c] = df_encoded[c].fillna(df_encoded[c].mean())
    elif fill_na_method == "median_for_numeric":
        for c in df_encoded.select_dtypes(include=[np.number]).columns:
            df_encoded[c] = df_encoded[c].fillna(df_encoded[c].median())
    elif fill_na_method == "mode_for_all":
        for c in df_encoded.columns:
            df_encoded[c] = df_encoded[c].fillna(df_encoded[c].mode().iloc[0] if not df_encoded[c].mode().empty else 0)
    else:
        df_encoded = df_encoded.fillna("")

st.write("Preview after preprocessing:")
st.dataframe(df_encoded.head(5))

# Train
model = None
results = None
if run_train:
    with st.spinner("Melatih model..."):
        model, results = train_and_evaluate(
            df_encoded, target_col=target_col, test_size=test_size,
            criterion=criterion, max_depth=(None if max_depth==0 else max_depth),
            random_state=int(random_state)
        )
    st.success("Training selesai ✅")

    # show metrics
    st.header("Hasil Evaluasi")
    st.metric("Accuracy", f"{results['accuracy']:.4f}")
    st.subheader("Classification Report")
    st.text(results["report_str"])

    st.subheader("Confusion Matrix")
    cm = results["confusion_matrix"]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    # tick labels
    labels = results["labels"]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="w")
    st.pyplot(fig)

    # Decision tree plot
    st.subheader("Visualisasi Decision Tree")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    plot_tree(results["model"], feature_names=results["feature_names"], class_names=[str(x) for x in results["labels"]], filled=True, ax=ax2, rounded=True)
    st.pyplot(fig2)

    # Save model button
    st.markdown("---")
    if st.button("💾 Unduh model (.joblib)"):
        file_bytes = download_joblib(model, filename="decision_tree_model.joblib")
        st.download_button("Download .joblib", data=file_bytes, file_name="decision_tree_model.joblib", mime="application/octet-stream")

    # Manual input prediction
    st.markdown("---")
    st.subheader("Coba Prediksi Manual")
    st.write("Masukkan nilai fitur untuk memprediksi kelas.")
    input_vals = {}
    feature_names = results["feature_names"]
    # for each numeric feature create input; for simplicity use text_input then convert
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            val = st.text_input(f"{feat}", key=f"input_{i}")
            input_vals[feat] = val

    if st.button("Prediksi"):
        # convert inputs to numeric where possible; non-convertible -> 0
        x = []
        for feat in feature_names:
            raw = input_vals[feat]
            try:
                v = float(raw)
            except:
                v = 0.0
            x.append(v)
        pred = predict_single(results["model"], np.array(x).reshape(1, -1))
        st.info(f"Prediksi kelas: **{pred[0]}**")

st.markdown("---")
st.caption("Dibuat otomatis berdasarkan modul Decision Tree. Hubungi saya jika mau versi Flask atau tambahan fitur (hyperparam tuning, cross-validation, pipeline, dll).")
