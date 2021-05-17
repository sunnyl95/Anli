import pandas as pd

def get_stay_home_data():
    return pd.read_csv("./data/generate_data.csv")
