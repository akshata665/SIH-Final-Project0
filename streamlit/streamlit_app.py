import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ====== UI STYLE FIX FOR PADDING =======
st.markdown(
    """<style>
    .css-18e3th9, .css-1d391kg { padding: 0rem 1rem 0rem 1rem; }
    </style>""",
    unsafe_allow_html=True
)

st.set_page_config(page_title="FRA AI Diagnostic System", layout="wide")

st.title("⚡🔧 Power Transformer FRA Fault Detection AI 🔌")

tab1, tab2, tab3 = st.tabs([
    "📁 Dataset",
    "🧠 Model",
    "🔎 Prediction"
])

# ======================================
# COMMON PREPROCESS FUNCTIONS
# ======================================
class_names = ['Normal','Interturn','Radial','Shorted','Axial','Core']

keyword_map = {
    "Normal":"Normal","Interturn":"Interturn","Fault":"Interturn",
    "Radial":"Radial","Deformation":"Radial",
    "Shorted":"Shorted","Turns":"Shorted",
    "Axial":"Axial","Displacement":"Axial",
    "Core":"Core","Looseness":"Core"
}

def extract_label(path):
    fname = os.path.basename(path).lower()
    for k,v in keyword_map.items():
        if k.lower() in fname:
            return v
    return None


def load_test_dataset():
    base = "full_clean_dataset"
    X = []
    Y = []

    max_len = 500
    max_cols = 4

    for root,_,files in os.walk(base):
        for f in files:
            if not f.endswith(".csv"):
                continue
            if f == "labels.csv":
                continue

            path = os.path.join(root, f)

            label = extract_label(path)
            if label is None:
                continue

            df = pd.read_csv(path).apply(pd.to_numeric, errors='coerce').fillna(0)
            arr = df.values

            if arr.shape[0] >= max_len:
                arr = arr[:max_len]
            else:
                arr = np.pad(arr, ((0, max_len-arr.shape[0]), (0,0)))

            if arr.shape[1] >= max_cols:
                arr = arr[:,:max_cols]
            else:
                arr = np.pad(arr, ((0,0),(0,max_cols-arr.shape[1])))

            X.append(arr)
            Y.append(class_names.index(label))

    return np.array(X,dtype=float), np.array(Y,dtype=int)


# ======================================
# TAB 1
# ======================================
with tab1:
    st.header("📁 Dataset Overview")

    df = pd.read_csv("full_clean_dataset/labels.csv")

    st.markdown("Our dataset contains measurements from multiple vendors, "
                "multiple fault types and multiple testing conditions.")

    st.markdown("---")
    
    st.subheader("📊 Dataset Size & Categories")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Fault Classes", df["fault_class"].nunique())
    col3.metric("Vendors", df["vendor"].nunique())

    st.markdown("---")

    u1, u2 = st.columns(2)
    with u1:
        st.markdown("### 🧩 Fault Classes Present")
        st.write(df["fault_class"].unique())

    with u2:
        st.markdown("### 🏢 Vendors Present")
        st.write(df["vendor"].unique())

    st.markdown("---")

    st.markdown("### 📈 Dataset Distribution Overview")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("#### Fault Class")
        st.bar_chart(df["fault_class"].value_counts())

    with c2:
        st.write("#### Vendor")
        st.bar_chart(df["vendor"].value_counts())

    with c3:
        st.write("#### Test Type")
        st.bar_chart(df["test_type"].value_counts())

    st.markdown("---")

    st.markdown("### 🔍 Representative Samples")
    sample_diverse = df.drop_duplicates(
        subset=["fault_class", "vendor", "test_type"]
    )
    st.dataframe(sample_diverse, use_container_width=True)


# ======================================
# TAB 2 (UNCHANGED)
# ======================================
with tab2:
    st.header("🧠 Model Summary")

    model = tf.keras.models.load_model("fra_model.h5")
    st.success("Model Loaded Successfully!")

    st.write("### Strong Hybrid CNN + BiLSTM + Autoencoder Model")

    with st.expander("📌 Click to view model architecture"):
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        summary = "\n".join(stringlist)
        st.code(summary)

    X, Y = load_test_dataset()
    st.write(f"Loaded Test Samples: {len(X)}")

    mean = X.mean(axis=(0,1), keepdims=True)
    std  = X.std(axis=(0,1), keepdims=True) + 1e-8
    X_norm = (X - mean) / std

    preds = model.predict(X_norm)
    y_pred = np.argmax(preds, axis=1)

    accuracy = (y_pred == Y).mean() * 100

    st.subheader("📌 Model Accuracy")
    st.metric("Accuracy", f"{accuracy:.2f}%")

    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(Y, y_pred)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm,
                annot=True,
                cmap="Blues",
                fmt="d",
                linewidths=0.5,
                cbar=False,
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel("Predicted", fontsize=12, fontweight='bold')
    plt.ylabel("Actual", fontsize=12, fontweight='bold')
    st.pyplot(fig)

    st.subheader("📌 Classification Report")

    report = classification_report(Y, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.style.format("{:.2f}"),
        use_container_width=True
    )


# ======================================
# TAB 3 (UNCHANGED)
# ======================================
with tab3:
    st.header("🔎 Upload FRA File for Prediction")

    uploaded_file = st.file_uploader("Upload FRA CSV File", type=["csv"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        df = pd.read_csv(uploaded_file).apply(pd.to_numeric, errors='coerce').fillna(0)
        arr = df.values

        max_len = 500
        max_cols = 4

        if arr.shape[0] >= max_len:
            arr = arr[:max_len]
        else:
            arr = np.pad(arr, ((0, max_len-arr.shape[0]), (0,0)))

        if arr.shape[1] >= max_cols:
            arr = arr[:,:max_cols]
        else:
            arr = np.pad(arr, ((0,0),(0,max_cols-arr.shape[1])))

        arr = np.array(arr, dtype=float)

        # FIXED SHAPE: NOW IT'S (1,500,4)
        x = np.expand_dims(arr, axis=0)
        x = (x - mean) / std

        pred = model.predict(x)[0]
        pred_idx = np.argmax(pred)
        confidence = pred[pred_idx] * 100

        st.subheader("🧠 Prediction Result")
        st.write(f"Predicted Fault Class: **{class_names[pred_idx]}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.subheader("📌 Class Probabilities")
        prob_df = pd.DataFrame({"Class": class_names, "Probability": pred})
        st.bar_chart(prob_df.set_index("Class"))
        # ==============================
        # XAI / Explainability Section
        # ==============================

        st.subheader("🧠 Why this Fault was Predicted? (Explainability)")

        # compute gradient for explainability
        import tensorflow as tf

        x_var = tf.Variable(x, dtype=float)
        with tf.GradientTape() as tape:
            preds = model(x_var)
            loss = preds[0, pred_idx]

        grads = tape.gradient(loss, x_var).numpy()[0]

        # reduce gradients per time step into a single importance score
        importance = np.mean(np.abs(grads), axis=1)

        # normalize for better visualization
        importance = (importance - importance.min()) / (importance.max() + 1e-8)

        # show as saliency graph
        st.write("Higher peaks mean those FRA frequency regions contributed more to this prediction.")

        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.plot(importance, color="red", linewidth=2)
        ax2.set_title("Feature Importance (Saliency)")
        ax2.set_xlabel("Frequency Sweep Index")
        ax2.set_ylabel("Importance")
        st.pyplot(fig2)
                