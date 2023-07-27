

import pandas as pd
from datetime import datetime, timedelta
import random
df = pd.read_csv('app/data/inference_data.csv')

# Create timestamp column
start = datetime(2022, 1, 1)
end = start + timedelta(minutes=len(df))
df = df.reset_index()
# df[''] = pd.date_range(start, end, periods=len(df))
# del df['Unnamed: 0']

df['label'] = df['label'].astype('float64')

df['id'] = df['index'].astype('float64')
del df['index']
df.to_csv('app/data/inference_data.csv', index=False)


