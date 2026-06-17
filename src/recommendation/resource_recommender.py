def recommend_resources(
    predicted_volume,
    severity
):

    officers = 2
    barricades = 0

    if predicted_volume >= 20:
        officers = 4

    if predicted_volume >= 50:
        officers = 6
        barricades = 2

    if predicted_volume >= 100:
        officers = 8
        barricades = 4

    if severity == "CRITICAL":
        officers += 2
        barricades += 2

    elif severity == "HIGH":
        officers += 1

    return {
        "officers": officers,
        "barricades": barricades
    }