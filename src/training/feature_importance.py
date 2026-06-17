import pandas as pd
import matplotlib.pyplot as plt


def feature_importance(model, X):

    importance = model.get_feature_importance()

    feat_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    })

    feat_df = feat_df.sort_values(
        by="importance",
        ascending=False
    )

    print("\n")
    print("=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    print(feat_df)

    plt.figure(figsize=(10, 6))

    plt.barh(
        feat_df["feature"],
        feat_df["importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()

    # save instead of show()
    plt.savefig(
        "feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\nFeature importance plot saved as feature_importance.png"
    )

    return feat_df