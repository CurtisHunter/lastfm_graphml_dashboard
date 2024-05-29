#ytrain.csv - use this. in downloads/lastfmdata
#final_Xtrain_filtered.csv -  use this. in downloads/lastfmdata
import pandas as pd
from lightgbm import LGBMClassifier
import joblib



reg_alpha = 0
feature_fraction = 0.65
num_leaves = 35
learning_rate = 0.07



Xtrain_filtered = pd.read_csv("C:/Users/Curt/Downloads/LastFMmydata/final_Xtrain_filtered.csv")
ytrain = pd.read_csv("C:/Users/Curt/Downloads/LastFMmydata/ytrain.csv", header = None)

ytrain = ytrain.values.ravel()


lgb = LGBMClassifier(reg_alpha=reg_alpha,
                     feature_fraction=feature_fraction,
                     num_leaves=num_leaves,
                     learning_rate=learning_rate,
                     verbose=-1)
lgb.fit(Xtrain_filtered, ytrain)

joblib.dump(lgb, 'LastFMmodel.joblib')

