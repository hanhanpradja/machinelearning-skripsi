import pandas as pd 

file = 'datasetyangdigunakan.xlsx'
df = pd.read_excel(file)
df_output = df['output']

for i in range(len(df_output)):
    if df_output[i] == 2:
        df_output[i] = 0
    elif df_output[i] == 3:
        df_output[i] = 2
    elif df_output[i] == 4:
        df_output[i] = 3

df.to_csv('dataset_balance_2.csv', index=False)


