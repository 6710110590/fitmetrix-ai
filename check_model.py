import json

import joblib
import pandas as pd
import sklearn

with open("models/model_meta.json", encoding="utf-8") as f:
    meta = json.load(f)

trained_ver = meta["versions"]["scikit-learn"]
print(f"scikit-learn ตอนเทรน: {trained_ver} | บนเครื่องนี้: {sklearn.__version__}")
if trained_ver != sklearn.__version__:
    print("คำเตือน: เวอร์ชันไม่ตรงกัน ให้ pip install scikit-learn==" + trained_ver)

model = joblib.load("models/fitmetrix_model.pkl")
print("โหลดโมเดลสำเร็จ:", meta["model_name"])
print("คอลัมน์ที่โมเดลต้องการ:", meta["input_columns"])

sample = pd.DataFrame([{
    "Gender": "male", "Age": 28, "Height": 175.0, "Weight": 72.0,
    "Duration": 25.0, "Heart_Rate": 110.0,
}])
pred = model.predict(sample)[0]
print(f"ผลทำนาย: {pred:.2f} kcal  (ตอนรันบน Colab ได้ 156.79 kcal)")