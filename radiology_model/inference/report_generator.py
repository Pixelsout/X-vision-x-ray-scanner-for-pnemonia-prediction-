def generate_report(pred_class):
    mapping = {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}
    disease = mapping.get(pred_class, "Unknown")

    report = f"Diagnosis: {disease}\n"
    if disease == "Normal":
        report += "No abnormalities detected in the X-ray."
    else:
        report += "Signs of infection detected. Recommend further evaluation."
    return report
