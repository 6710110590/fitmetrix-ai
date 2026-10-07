# =============================================================================
# FitMetrix AI - Training Script (ขั้นที่ 0)
#   - เทรน 3 โมเดล x 2 ชุดฟีเจอร์ ("full" มี Body_Temp / "no_temp" ไม่มี Body_Temp)
#   - เปรียบเทียบว่าตัด Body_Temp แล้วคะแนนลดลงเท่าไร
#   - เซฟ fitmetrix_model.pkl + model_meta.json (ช่วงค่าฝึก, เวอร์ชันไลบรารี, คะแนน)
#
# วิธีรันบน Google Colab:
#   1) วางสคริปต์นี้ในเซลล์เดียว แล้วกดรัน
#   2) เมื่อมีหน้าต่างอัปโหลด ให้เลือก 3 ไฟล์พร้อมกัน:
#        exercise.csv, calories.csv, features.py
# =============================================================================

import importlib
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------- CONFIG ---------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
RUN_CV = True                    # ทำ Cross-Validation ด้วย (ช้าลงนิดหน่อย)
CV_FOLDS = 5
EXPORT_FEATURE_SET = "no_temp"   # "no_temp" = ไม่ใช้ Body_Temp (เหมาะกับ Real-time Tracker)
MODEL_PATH = "fitmetrix_model.pkl"
META_PATH = "model_meta.json"
RESULTS_PATH = "model_comparison.csv"

CAT_COLS = ["Gender"]
BASE_NUM = ["Age", "Height", "Weight", "Duration", "Heart_Rate"]
ENGINEERED = ["BMI", "BMR"]
TARGET = "Calories"
FEATURE_SETS = {
    "full": BASE_NUM + ["Body_Temp"],
    "no_temp": BASE_NUM,
}


# ----------------------------- STEP 0: เตรียมไฟล์ ----------------------------
REQUIRED = ["exercise.csv", "calories.csv", "features.py"]
missing = [f for f in REQUIRED if not os.path.exists(f)]
if missing:
    try:
        from google.colab import files  # type: ignore
        print(f"ไม่พบไฟล์: {missing}\nกรุณาอัปโหลด {REQUIRED} พร้อมกัน")
        files.upload()
    except ImportError:
        raise FileNotFoundError(f"ไม่พบไฟล์ที่จำเป็น: {missing}")
    still = [f for f in REQUIRED if not os.path.exists(f)]
    if still:
        raise FileNotFoundError(f"ยังขาดไฟล์: {still}")

# import FeatureEngineer จากโมดูลแยก เพื่อให้ .pkl อ้างอิง features.FeatureEngineer
sys.path.insert(0, os.getcwd())
importlib.invalidate_caches()
from features import FeatureEngineer  # noqa: E402

print("=" * 72)
print("เวอร์ชันไลบรารี (ต้องติดตั้งให้ตรงกันบนเครื่อง/แอป)")
print(f"  python       : {platform.python_version()}")
print(f"  scikit-learn : {sklearn.__version__}")
print(f"  numpy        : {np.__version__}")
print(f"  pandas       : {pd.__version__}")
print(f"  joblib       : {joblib.__version__}")


# ----------------------------- STEP 1: โหลด + Merge --------------------------
exercise_df = pd.read_csv("exercise.csv")
calories_df = pd.read_csv("calories.csv")
df = exercise_df.merge(calories_df, on="User_ID", how="inner", validate="one_to_one")

print("\n" + "=" * 72)
print(f"STEP 1: exercise {exercise_df.shape} + calories {calories_df.shape} -> merged {df.shape}")

# ----------------------------- STEP 2: Missing Values ------------------------
print("\nSTEP 2: Missing Values")
print(df.isnull().sum().to_string())
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

# ----------------------------- STEP 3: Train/Test Split ----------------------
ALL_INPUT_COLS = CAT_COLS + FEATURE_SETS["full"]
X_all, y = df[ALL_INPUT_COLS], df[TARGET]
X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\nSTEP 3: Train = {len(X_train_all)} | Test = {len(X_test_all)} (ใช้ชุดแบ่งเดียวกันทุกโมเดล เพื่อเทียบอย่างยุติธรรม)")


# ----------------------------- STEP 4: Pipeline builder ----------------------
def build_pipeline(model, num_cols):
    """FeatureEngineer -> (Imputer+Scaler / Imputer+OneHot) -> Model  (สร้างใหม่ทุกครั้ง)"""
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols + ENGINEERED),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT_COLS),
    ])
    return Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def make_models():
    return {
        "Linear Regression (Baseline)": LinearRegression(),
        "Random Forest (Main)": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "MLP Neural Network": TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=(128, 64), max_iter=1000,
                early_stopping=True, random_state=RANDOM_STATE,
            ),
            transformer=StandardScaler(),
        ),
    }


# ----------------------------- STEP 5: Train + Evaluate ----------------------
print("\n" + "=" * 72)
print("STEP 5: เทรนและประเมินผล (3 โมเดล x 2 ชุดฟีเจอร์)")

rows, pipelines = [], {}
for fs_name, num_cols in FEATURE_SETS.items():
    cols = CAT_COLS + num_cols
    X_tr, X_te = X_train_all[cols], X_test_all[cols]
    for model_name, model in make_models().items():
        pipe = build_pipeline(model, num_cols)

        t0 = time.time()
        pipe.fit(X_tr, y_train)
        train_time = time.time() - t0

        pred = pipe.predict(X_te)
        row = {
            "Feature Set": fs_name,
            "Model": model_name,
            "R2-Score": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        }
        if RUN_CV:
            cv = cross_val_score(
                build_pipeline(make_models()[model_name], num_cols),
                X_tr, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=-1,
            )
            row["CV R2 (mean)"] = cv.mean()
            row["CV R2 (std)"] = cv.std()
        row["Train Time (s)"] = train_time
        rows.append(row)
        pipelines[(fs_name, model_name)] = pipe
        print(f"  [{fs_name:<7}] {model_name:<30} R2={row['R2-Score']:.4f}  ({train_time:.1f}s)")

results_df = (
    pd.DataFrame(rows)
    .sort_values(["Feature Set", "R2-Score"], ascending=[True, False])
    .reset_index(drop=True)
)
results_df.to_csv(RESULTS_PATH, index=False)

print("\n" + "=" * 72)
print("ตารางเปรียบเทียบทั้งหมด")
print("=" * 72)
print(results_df.round(4).to_string(index=False))

# ----------------------------- STEP 6: ตัด Body_Temp แล้วเสียไปเท่าไร ---------
print("\n" + "=" * 72)
print("STEP 6: ผลของการตัด Body_Temp (no_temp เทียบกับ full)")
print("=" * 72)
full = results_df[results_df["Feature Set"] == "full"].set_index("Model")
nt = results_df[results_df["Feature Set"] == "no_temp"].set_index("Model")
impact = pd.DataFrame({
    "R2 (full)": full["R2-Score"],
    "R2 (no_temp)": nt["R2-Score"],
    "R2 change": nt["R2-Score"] - full["R2-Score"],
    "MAE (full)": full["MAE"],
    "MAE (no_temp)": nt["MAE"],
    "MAE change": nt["MAE"] - full["MAE"],
})
print(impact.round(4).to_string())
print("\nแนวทางตัดสินใจ (heuristic): ถ้า R2 ลดลงน้อยมาก (เช่น < 0.01) และ MAE เพิ่มเล็กน้อย")
print("ใช้ชุด no_temp ได้เลย เพราะ Real-time Tracker ไม่มีค่าอุณหภูมิร่างกาย")

# ----------------------------- STEP 7: เลือกโมเดล + Export -------------------
cand = results_df[results_df["Feature Set"] == EXPORT_FEATURE_SET].sort_values(
    ["R2-Score", "RMSE"], ascending=[False, True]
)
best_row = cand.iloc[0]
best_name = best_row["Model"]
best_pipeline = pipelines[(EXPORT_FEATURE_SET, best_name)]
export_num_cols = FEATURE_SETS[EXPORT_FEATURE_SET]
input_cols = CAT_COLS + export_num_cols

joblib.dump(best_pipeline, MODEL_PATH)

print("\n" + "=" * 72)
print(f"STEP 7: Export -> '{best_name}' | feature set = {EXPORT_FEATURE_SET}")
print(f"        R2={best_row['R2-Score']:.4f}  MAE={best_row['MAE']:.3f}  RMSE={best_row['RMSE']:.3f}")
print(f"        input columns ที่แอปต้องส่งให้โมเดล: {input_cols}")


# ----------------------------- STEP 8: model_meta.json -----------------------
def range_info(s: pd.Series) -> dict:
    return {
        "min": float(s.min()), "max": float(s.max()),
        "p01": float(s.quantile(0.01)), "p99": float(s.quantile(0.99)),
        "mean": float(s.mean()),
    }


X_train_used = X_train_all[input_cols]
meta = {
    "project": "FitMetrix AI",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "model_file": MODEL_PATH,
    "model_name": best_name,
    "feature_set": EXPORT_FEATURE_SET,
    "input_columns": input_cols,
    "categorical_values": {"Gender": sorted(df["Gender"].astype(str).str.lower().unique().tolist())},
    "feature_ranges": {c: range_info(X_train_used[c]) for c in export_num_cols},
    "target": TARGET,
    "target_range": range_info(y_train),
    "metrics_test": {
        "R2": float(best_row["R2-Score"]),
        "MAE": float(best_row["MAE"]),
        "RMSE": float(best_row["RMSE"]),
    },
    "split": {
        "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
        "n_train": int(len(X_train_all)), "n_test": int(len(X_test_all)),
    },
    "all_results": json.loads(results_df.round(5).to_json(orient="records")),
    "versions": {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    },
    "notes": [
        "feature_ranges คือช่วงค่าของข้อมูลที่ใช้เทรน ถ้า input อยู่นอกช่วง p01-p99 ควรเตือนผู้ใช้ว่าความแม่นยำอาจต่ำ",
        "FeatureEngineer อยู่ในโมดูล features.py ต้อง import ได้ตอน joblib.load()",
    ],
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"        บันทึก '{META_PATH}' และ '{RESULTS_PATH}' แล้ว")

print("\nช่วงค่าข้อมูลฝึก (ใช้เตือนผู้ใช้ใน Tracker):")
for c, r in meta["feature_ranges"].items():
    print(f"  {c:<11} min={r['min']:.1f}  p01={r['p01']:.1f}  p99={r['p99']:.1f}  max={r['max']:.1f}")

# ----------------------------- STEP 9: Sanity checks -------------------------
print("\n" + "=" * 72)
print("STEP 9: ตรวจสอบไฟล์ที่เซฟ")
loaded = joblib.load(MODEL_PATH)
X_check = X_test_all[input_cols].head(200)
same = np.allclose(loaded.predict(X_check), best_pipeline.predict(X_check))
fe_module = loaded.named_steps["feature_engineering"].__class__.__module__
print(f"  โหลดกลับแล้วทำนายตรงกับตัวเดิม : {same}")
print(f"  FeatureEngineer อ้างอิงโมดูล   : '{fe_module}'  (ต้องเป็น 'features' ไม่ใช่ '__main__')")
assert same and fe_module == "features", "Export ผิดพลาด ตรวจสอบ features.py"

example = {"Gender": "male", "Age": 28, "Height": 175.0, "Weight": 72.0,
           "Duration": 25.0, "Heart_Rate": 110.0}
if "Body_Temp" in input_cols:
    example["Body_Temp"] = 40.5
print(f"  ตัวอย่างทำนาย {example}\n  -> {loaded.predict(pd.DataFrame([example]))[0]:.2f} kcal")

print("\n" + "=" * 72)
print("สิ่งที่ต้องจดไปใช้ในโปรเจกต์ VS Code:")
print(f"  pip install scikit-learn=={sklearn.__version__} numpy=={np.__version__} "
      f"pandas=={pd.__version__} joblib=={joblib.__version__}")

# ----------------------------- STEP 10: แพ็กไฟล์ดาวน์โหลด --------------------
import zipfile  # noqa: E402

ZIP_PATH = "fitmetrix_artifacts.zip"
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for p in [MODEL_PATH, META_PATH, RESULTS_PATH, "features.py"]:
        z.write(p)
print(f"\nแพ็กไฟล์ทั้งหมดไว้ที่ '{ZIP_PATH}'")

try:
    from google.colab import files  # type: ignore
    files.download(ZIP_PATH)   # เบราว์เซอร์จะดาวน์โหลดไฟล์ zip ให้อัตโนมัติ
except ImportError:
    pass
