import pandas as pd
import joblib

# Load the model
model = joblib.load('LastFMmodel.joblib')


# Extract the expected feature names from the model
if hasattr(model, 'feature_names_in_'):
    columns_list = model.feature_names_in_
else:
    # If the model does not have feature_names_in_, fallback to using the columns.csv file
    columns_df = pd.read_csv("columns.csv", header=None)
    columns_list = columns_df[0].tolist()



features = {"degree_centrality": 0, "4111": 1, 1409  : 1, "4648" : 1}
features_df = pd.DataFrame([features])
features_df.index = features_df.index.astype(str)

features_dict = features_df.iloc[0].to_dict() # i want to create features_dict from the features dataframe
print(features_dict)
payload = {
    "features": features_dict
}

print(payload)
# Create a DataFrame with a single row of features and specify the index
Xtest = pd.DataFrame(features_dict)
print(Xtest)



# Align the DataFrame columns with the model's expected columns
Xtest = Xtest.reindex(columns=columns_list, fill_value=0)

# Predict using the model
prediction = model.predict(Xtest)
print(prediction)