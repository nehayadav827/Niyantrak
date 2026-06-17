import shap
import matplotlib.pyplot as plt


def run_shap_analysis(model, X):

    print("\nRunning SHAP Analysis...")

    try:

        sample_size = min(
            1000,
            len(X)
        )

        X_sample = X.sample(
            sample_size,
            random_state=42
        )

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            X_sample
        )

        plt.figure()

        # =====================================================
        # MULTICLASS HANDLING
        # =====================================================

        if isinstance(shap_values, list):

            # Use first class for summary.
            values_to_plot = shap_values[0]

        elif len(getattr(shap_values, "shape", [])) == 3:

            # Shape can be:
            # (samples, features, classes)
            values_to_plot = shap_values[:, :, 0]

        else:

            values_to_plot = shap_values

        shap.summary_plot(
            values_to_plot,
            X_sample,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            "shap_summary.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "SHAP summary saved as shap_summary.png"
        )

    except Exception as e:

        print("\nSHAP analysis failed:")
        print(e)