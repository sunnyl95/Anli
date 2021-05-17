import pandas as pd
import pkg_resources

def get_stay_point_data():
    DATA_FILE = pkg_resources.resource_filename('trackinfomining', 'data/traject.csv')
    return pd.read_csv(DATA_FILE)