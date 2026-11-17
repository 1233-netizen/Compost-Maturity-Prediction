
import gradio as gr
import pandas as pd
import joblib

# Load the trained model and feature names
model = joblib.load("compost_maturity_model.pkl")
feature_names = joblib.load("compost_feature_names.pkl")


def predict_maturity(
    day,
    temperature,
    moisture,
    ph,
    ammonia,
    nitrate,
    total_nitrogen,
    total_organic_carbon,
    electrical_conductivity,
    organic_matter
):
    input_data = pd.DataFrame([{
        "Day": day,
        "Temperature": temperature,
        "MC(%)": moisture,
        "pH": ph,
        "Ammonia(mg/kg)": ammonia,
        "Nitrate(mg/kg)": nitrate,
        "TN(%)": total_nitrogen,
        "TOC(%)": total_organic_carbon,
        "EC(ms/cm)": electrical_conductivity,
        "OM(%)": organic_matter
    }])

    # Match the feature order used during model training
    input_data = input_data[feature_names]

    prediction = model.predict(input_data)[0]

    return round(float(prediction), 2)


with gr.Blocks(title="Compost Maturity Prediction System") as demo:

    gr.Markdown("# 🌱 Compost Maturity Prediction System")

    gr.Markdown(
        "Enter compost measurements to estimate the maturity score "
        "using a trained Random Forest regression model."
    )

    gr.Markdown("### Composting Conditions")

    with gr.Row():
        day = gr.Number(label="Composting Day", value=12)
        temperature = gr.Number(label="Temperature (°C)", value=38)
        moisture = gr.Number(label="Moisture Content (%)", value=45)

    gr.Markdown("### Chemical Properties")

    with gr.Row():
        ph = gr.Number(label="pH", value=8)
        ammonia = gr.Number(label="Ammonia (mg/kg)", value=1500)
        nitrate = gr.Number(label="Nitrate (mg/kg)", value=200)

    with gr.Row():
        total_nitrogen = gr.Number(label="Total Nitrogen (%)", value=1.9)
        total_organic_carbon = gr.Number(
            label="Total Organic Carbon (%)", value=30
        )

    with gr.Row():
        electrical_conductivity = gr.Number(
            label="Electrical Conductivity (ms/cm)", value=3
        )
        organic_matter = gr.Number(label="Organic Matter (%)", value=50)

    predict_button = gr.Button(
        "Predict Maturity Score", variant="primary"
    )

    output = gr.Number(label="Predicted Compost Maturity Score")

    predict_button.click(
        fn=predict_maturity,
        inputs=[
            day, temperature, moisture, ph, ammonia, nitrate,
            total_nitrogen, total_organic_carbon,
            electrical_conductivity, organic_matter
        ],
        outputs=output
    )

    gr.Markdown(
        "Note: This is a model-based estimate, not a laboratory "
        "certification of compost maturity."
    )


import os

port = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port)
