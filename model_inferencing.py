import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load('LastFMmodel.joblib')

# Extract the expected feature names from the model
if hasattr(model, 'feature_names_in_'):
    columns_list = model.feature_names_in_
else:
    # If the model does not have feature_names_in_, fallback to using the columns.csv file
    columns_df = pd.read_csv("columns.csv", header=None)
    columns_list = columns_df[0].tolist()

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json
    Xtest = pd.DataFrame([features])
    Xtest = Xtest.reindex(columns=columns_list, fill_value=0)
    prediction = model.predict(Xtest)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)