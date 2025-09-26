# utils.py
import pandas as pd
import numpy as np
from io import BytesIO
import joblib

def load_data_auto(uploaded_file, use_example=False):
    """
    Jika uploaded_file ada -> baca,
    jika tidak, coba baca '/mnt/data/BlaBla.xlsx' (yang kamu upload ke session).
    jika tidak ada dan use_example True -> buat synthetic contoh.
    """
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            print("Error membaca uploaded file:", e)
    # try default path
    try:
        df = pd.read_excel("/mnt/data/BlaBla.xlsx")
        return df
    except Exception:
        pass

    if use_example:
        # buat dataset contoh (synthetic)
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=200, n_features=6, n_informative=4, n_classes=3, random_state=42)
        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        df["target"] = y
        return df

    # fallback: dataframe kosong
    return pd.DataFrame()

def auto_encode_dataframe(df, strategy="mean_for_numeric"):
    """
    - Isi missing values sesuai strategy.
    - Encode categorical (object/str) via factorize (simple).
    Returns: df_encoded, mapping_info
    """
    df2 = df.copy()
    mapping = {}
    # fillna
    if strategy == "mean_for_numeric":
        for c in df2.select_dtypes(include=[np.number]).columns:
            df2[c] = df2[c].fillna(df2[c].mean())
        for c in df2.select_dtypes(exclude=[np.number]).columns:
            df2[c] = df2[c].fillna(df2[c].mode().iloc[0] if not df2[c].mode().empty else "")
    elif strategy == "median_for_numeric":
        for c in df2.select_dtypes(include=[np.number]).columns:
            df2[c] = df2[c].fillna(df2[c].median())
        for c in df2.select_dtypes(exclude=[np.number]).columns:
            df2[c] = df2[c].fillna(df2[c].mode().iloc[0] if not df2[c].mode().empty else "")
    elif strategy == "mode_for_all":
        for c in df2.columns:
            df2[c] = df2[c].fillna(df2[c].mode().iloc[0] if not df2[c].mode().empty else 0)
    else:
        df2 = df2.fillna("")

    # encode categorical
    for c in df2.select_dtypes(include=["object", "category"]).columns:
        vals, uniques = pd.factorize(df2[c])
        mapping[c] = dict(enumerate(uniques))
        df2[c] = vals

    return df2, mapping

def download_joblib(model, filename="model.joblib"):
    bio = BytesIO()
    joblib.dump(model, bio)
    bio.seek(0)
    return bio.read()
