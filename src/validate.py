def validate_data(df):
    required_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    return all(col in df.columns for col in required_columns)
