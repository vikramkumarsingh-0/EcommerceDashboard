import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# File paths
DATA_FILE = "Filtered_Data (5).csv"
MODEL_FILE = "trained_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODERS_FILE = "label_encoders.pkl"

# Load dataset
if "filtered_data" in st.session_state and not st.session_state.filtered_data.empty:
    filtered_df = st.session_state.filtered_data
# Data Preprocessing
st.write("### Data Preprocessing")
df = filtered_df.dropna()

# Encoding categorical variables
categorical_columns = ['Booking Type', 'Class Type', 'Time Slot', 'Facility', 'Theme', 'Service Type', 'Instructor']
label_encoders = {}

for col in categorical_columns:
    df[col] = df[col].astype(str)  # Ensure it's string type
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Encode column
    label_encoders[col] = le  # Store encoder

# Save label encoders
joblib.dump(label_encoders, ENCODERS_FILE)

# Feature Selection
features = categorical_columns + ['Duration (mins)']
target = 'Price'

X = df[features]
y = df[target]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, SCALER_FILE)

# Train Model
st.write("### Training the Model...")
model = XGBRegressor(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, MODEL_FILE)

# Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Prediction Interface in Streamlit
st.write("### Predict Booking Price")

# Load saved model & encoders
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(ENCODERS_FILE):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    label_encoders = joblib.load(ENCODERS_FILE)

input_data = {}

for feature in features:
    if feature in categorical_columns:
        original_categories = list(label_encoders[feature].classes_)
        selected_category = st.selectbox(feature, original_categories)
        input_data[feature] = selected_category
    else:
        input_data[feature] = st.number_input(feature, min_value=0.0, value=30.0)

# Convert user input to DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical inputs safely
for col in categorical_columns:
    input_df[col] = input_df[col].astype(str)
    if input_df[col][0] in label_encoders[col].classes_:
        input_df[col] = label_encoders[col].transform(input_df[col])
    else:
        st.warning(f"⚠️ Unseen label '{input_df[col][0]}' in '{col}'. Using default value (-1).")
        input_df[col] = -1  # Assign -1 for unknown labels

# Scale input
input_df = scaler.transform(input_df)

# Predict Price
if st.button("Predict Price"):
    predicted_price = model.predict(input_df)
    st.write(f"### Predicted Price: { predicted_price[0]:.2f}")
