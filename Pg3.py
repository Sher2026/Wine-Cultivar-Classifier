import streamlit as st
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------
# Load and Train Model
# ----------------------------------
wine = load_wine()
# Using 4 key chemical features for clean interactive input
selected_features = [0, 1, 6, 9]  # Alcohol, Malic Acid, Flavanoids, Color Intensity
X = wine.data[:, selected_features]
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# ----------------------------------
# Streamlit Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Wine Cultivar Classifier",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Wine Cultivar Classifier")
st.write("Enter the wine's chemical measurements below and click **Predict** to identify the cultivar.")
st.write(f"### Model Accuracy: **{accuracy:.2%}**")
st.divider()

# ----------------------------------
# User Inputs
# ----------------------------------
alcohol = st.number_input("Alcohol (%)", min_value=10.0, max_value=16.0, value=13.0)
malic_acid = st.number_input("Malic Acid (g/L)", min_value=0.5, max_value=6.0, value=2.3)
flavanoids = st.number_input("Flavanoids (mg/L)", min_value=0.1, max_value=6.0, value=2.0)
color_intensity = st.number_input("Color Intensity", min_value=1.0, max_value=14.0, value=5.0)

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("Predict Cultivar"):
    user_input = [[alcohol, malic_acid, flavanoids, color_intensity]]
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input)
    cultivar = wine.target_names[prediction[0]]

    st.success(f"Predicted Cultivar: **{cultivar.upper()}**")

    st.subheader("Prediction Confidence")
    st.write({
        wine.target_names[i]: f"{probability[0][i] * 100:.2f}%"
        for i in range(len(wine.target_names))
    })

    st.progress(float(max(probability[0])))

st.divider()
st.caption("Developed using Streamlit and Scikit-Learn")