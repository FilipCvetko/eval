import streamlit as st
import pickle
import torch
from src.ageclocks.blood.deepctrl.model import DeepCTRLModel, Encoder, Net

# with open("/home/filip/IT/Longevize/machine-learning/src/ageclocks/blood/deepctrl/saved_models/model1.pkl", "rb") as file:
    # model = pickle.load(file)

model = torch.load("/home/filip/IT/Longevize/machine-learning/src/ageclocks/blood/deepctrl/saved_models/model1")

features = [
    "albumin",
    "hbA1c%",
    "cholesterol",
    "SHBG",
    "urea",
    "apolipoproteinB",
    "gender_male",
    "creatinine",
]

st.title("DeepCTRL model evaluation")

st.write("This is a simple demonstration of using rules to enhance deep learning. Currently only one rule has been applied - higher cholesterol, result in higher biological age. If alpha is set to 1.0 only rule will be followed, if alpha is 0.0 only the data will be used for inference - classical deep learning.")

col1, col2 = st.columns(2)

albumin = col1.number_input("Albumin: ")
hbA1c = col1.number_input("hbA1c%: ")
chol = col1.number_input("LDL")
shbg = col1.number_input("SHBG")
urea = col2.number_input("Urea")
apoB = col2.number_input("Apolipoprotein B")
creatinine = col2.number_input("Creatinine")
gender = col2.radio("Gender", options=["M", "F"])

gender_male = 1.0 if gender == "M" else 0.0

alpha = st.slider("Alpha (1 = rule): ", 0.0, 1.0, 0.0)

print(model.predict([54.0, 5.7, 2.85, 13.6, 4.5, 1.01, 1.0, 101], alpha=1.0))

if st.button("Predict"):
    pred = model.predict([albumin, hbA1c, chol, shbg, urea, apoB, gender_male, creatinine], alpha=alpha)
    st.write("Predicted age: ", pred)


