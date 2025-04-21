import pandas as pd

def preprocess_data(df):

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    marital_map = {
        'single': 0,
        'non-single (divorced / separated / married / widowed)': 1
    }
    if 'Marital status' in df.columns:
        df['Marital status'] = df['Marital status'].map(marital_map)

    df['Education'] = df['Education'].str.replace(' ', '', regex=False)
    df['Occupation'] = df['Occupation'].str.replace(' ', '', regex=False)

    df['Education'] = df['Education'].str.replace('/', '_', regex=False)
    df['Occupation'] = df['Occupation'].str.replace('/', '_', regex=False)

    df['ID'] = df['ID'] - 100000000
    #print(df['Education'].unique())

    categorical_to_encode = []
    if 'Education' in df.columns:
        categorical_to_encode.append('Education')
    if 'Occupation' in df.columns:
        categorical_to_encode.append('Occupation')
    
    df = pd.get_dummies(df, columns=categorical_to_encode, drop_first=True)

    df = df.ffill()
    
    return df

df = pd.read_csv('./sgdata.csv')
df = preprocess_data(df)
df.to_csv('./cleaned_data.csv', index=False)
