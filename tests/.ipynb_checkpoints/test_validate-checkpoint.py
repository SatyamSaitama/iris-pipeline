from src.validate import validate_data
import pandas as pd

def test_validate_data():
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    assert validate_data(df)
