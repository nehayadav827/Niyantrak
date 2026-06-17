from src.preprocessing.clean import load_data

from src.forecasting.build_risk_datset import (
    build_risk_dataset
)

from src.forecasting.cross_validation_risk import (
    cross_validate_risk
)

from src.forecasting.train_risk_regressor import (
    train_risk_regressor
)

from src.forecasting.risk_predictor import (
    predict_risk
)

DATA_PATH = "data.csv"

df = load_data(DATA_PATH)

risk_df = build_risk_dataset(df)


cross_validate_risk(
    risk_df
)

train_risk_regressor(
    risk_df
)

predict_risk()