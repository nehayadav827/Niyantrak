from src.preprocessing.clean import load_data
from src.preprocessing.create_target import create_target
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.advanced_features import add_advanced_features

from src.training.train_catboost import train_catboost
from src.training.feature_importance import feature_importance
from src.training.shap_analysis import run_shap_analysis

# =====================================================
# DATA PATH
# =====================================================

DATA_PATH = "data.csv"

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading Dataset...")

df = load_data(DATA_PATH)

print("Dataset Shape:", df.shape)

# =====================================================
# TARGET CREATION
# =====================================================

print("\nCreating Severity Target...")

df = create_target(df)

print(
    df["severity_class"]
    .value_counts()
)

# =====================================================
# BASIC FEATURE ENGINEERING
# =====================================================

print("\nBuilding Features...")

df = engineer_features(df)

# =====================================================
# ADVANCED FEATURES
# =====================================================

print("\nBuilding Advanced Features...")

df = add_advanced_features(df)

print(
    "\nFinal Dataset Shape:",
    df.shape
)

# =====================================================
# TRAIN MODEL
# =====================================================

model, X = train_catboost(df)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

feature_importance(
    model,
    X
)

# =====================================================
# SHAP ANALYSIS
# =====================================================

run_shap_analysis(
    model,
    X
)

print("\nPipeline Complete.")