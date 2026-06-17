import pandas as pd
import matplotlib.pyplot as plt


def forecast_feature_importance(
    model,
    features
):

    print("\n" + "=" * 60)
    print("FORECAST FEATURE IMPORTANCE")
    print("=" * 60)

    # =====================================================
    # NEW HURDLE MODEL BUNDLE
    # =====================================================

    if isinstance(model, dict) and model.get("model_type") == "zero_inflated_hurdle_v1":

        classifier = model["classifier"]
        regressor = model["regressor"]

        classifier_importance = classifier.get_feature_importance()

        if regressor is not None:

            regressor_importance = regressor.get_feature_importance()

        else:

            regressor_importance = [
                0.0
                for _ in features
            ]

        imp_df = pd.DataFrame({
            "feature": features,
            "classifier_importance": classifier_importance,
            "regressor_importance": regressor_importance
        })

        imp_df["combined_importance"] = (
            0.55 * imp_df["classifier_importance"]
            +
            0.45 * imp_df["regressor_importance"]
        )

        imp_df = imp_df.sort_values(
            by="combined_importance",
            ascending=False
        )

        print(
            imp_df.head(25)
        )

        plt.figure(
            figsize=(11, 8)
        )

        plot_df = imp_df.head(25).sort_values(
            by="combined_importance",
            ascending=True
        )

        plt.barh(
            plot_df["feature"],
            plot_df["combined_importance"]
        )

        plt.xlabel(
            "Combined Importance"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            "Zero-Inflated Forecast Feature Importance"
        )

        plt.tight_layout()

        plt.savefig(
            "forecast_feature_importance.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "\nForecast feature importance plot saved as "
            "forecast_feature_importance.png"
        )

        return imp_df

    # =====================================================
    # OLD SINGLE REGRESSOR MODEL
    # =====================================================

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
        imp_df.head(25)
    )

    plt.figure(
        figsize=(10, 8)
    )

    plot_df = imp_df.head(25).sort_values(
        by="importance",
        ascending=True
    )

    plt.barh(
        plot_df["feature"],
        plot_df["importance"]
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
        "forecast_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nForecast feature importance plot saved as "
        "forecast_feature_importance.png"
    )

    return imp_df