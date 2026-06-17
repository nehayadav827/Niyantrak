import pandas as pd
import matplotlib.pyplot as plt


def forecast_feature_importance(model, features):

    importance = model.get_feature_importance()

    imp_df = pd.DataFrame({
        "feature": features,
        "importance": importance
    })

    imp_df = imp_df.sort_values(
        by="importance",
        ascending=False
    )

    print(
        "\n"
        + "=" * 60
    )
    print(
        "FORECAST FEATURE IMPORTANCE"
    )
    print(
        "=" * 60
    )

    print(
        imp_df.head(25)
    )

    # ==========================================
    # SAVE PLOT
    # ==========================================

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        imp_df["feature"],
        imp_df["importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Forecast Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        "forecast_feature_importance.png"
    )

    print(
        "\nForecast feature importance plot saved as "
        "forecast_feature_importance.png"
    )

    return imp_df