import pandas as pd

file_b = 'distance_to_babble_f1f2.csv'
df = pd.read_csv(file_b)

#print(df)
df = df.loc[df['Category'] == 'C']
df_m = df.groupby(['Talker']).mean()
df_m['distance_babble'] = df_m['distance_babble'].apply(lambda x: round(x, 2))
#print(df_m)

file_s = 'distance_to_ssn_f1f2.csv'
df_s = pd.read_csv(file_s)

df_s = df_s.loc[df_s['Category'] == 'C']
df_s_mean = df_s.groupby(['Talker']).mean()
df_s_mean['distance_ssn'] = df_s_mean['distance_ssn'].apply(lambda x: round(x, 2))
#print(df_s_mean)

df_s_mean ['distance_babble'] = pd.Series(df_m['distance_babble'])
print(df_s_mean)
#print(ssn)

print(df_s_mean.to_latex(index=False))