import pandas as pd
import pkg_resources

def get_stay_home_data():
    DATA_FILE = pkg_resources.resource_filename('trackinfomining', 'data/generate_data.csv')
    return pd.read_csv(DATA_FILE)