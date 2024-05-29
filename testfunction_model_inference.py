import pandas as pd
import joblib

def model_inference(features):
    # Load the model
    model = joblib.load('LastFMmodel.joblib')


    # Extract the expected feature names from the model
    if hasattr(model, 'feature_names_in_'):
        columns_list = model.feature_names_in_
    else:
        # If the model does not have feature_names_in_, fallback to using the columns.csv file
        columns_df = pd.read_csv("columns.csv", header=None)
        columns_list = columns_df[0].tolist()

    # Create a DataFrame with a single row of features and specify the index
    Xtest = pd.DataFrame([features])


    # Align the DataFrame columns with the model's expected columns
    Xtest = Xtest.reindex(columns=columns_list, fill_value=0)

    # Predict using the model
    prediction = model.predict(Xtest)
    return prediction