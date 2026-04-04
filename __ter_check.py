"""Quick TER join coverage check."""
import re
import pandas as pd

ter = pd.read_csv('Data/ter-of-mf-schemes.csv')
mf  = pd.read_csv('Data/mutual_fund_data.csv')
mf.columns  = [c.strip() for c in mf.columns]
ter.columns = [c.strip() for c in ter.columns]

ter['TER Date'] = pd.to_datetime(ter['TER Date'], dayfirst=True, errors='coerce')
ter_latest = ter.sort_values('TER Date').groupby('Scheme Name').last().reset_index()
ter_latest.rename(columns={'Scheme Name': 'Scheme_Name'}, inplace=True)

print('MF sample:', mf['Scheme_Name'].head(3).tolist())
print('TER sample:', ter_latest['Scheme_Name'].head(3).tolist())

# Direct join
merged0 = mf.merge(ter_latest[['Scheme_Name','Regular Plan - Total TER (%)','Direct Plan - Total TER (%)']], on='Scheme_Name', how='left')
m0 = merged0['Regular Plan - Total TER (%)'].notna().sum()
print(f'\nDirect exact join: {m0}/{len(mf)} ({100*m0/len(mf):.1f}%)')

# Fuzzy: strip plan/option suffixes
def clean_name(s):
    s = str(s)
    s = re.sub(r'\s*[-\u2013]\s*(regular|direct|growth|dividend|idcw|bonus|option|plan)\s*(plan|option)?.*$', '', s, flags=re.I)
    return s.strip().lower()

mf2 = mf.copy(); mf2['_key'] = mf2['Scheme_Name'].apply(clean_name)
ter2 = ter_latest.copy(); ter2['_key'] = ter2['Scheme_Name'].apply(clean_name)
ter2 = ter2.drop_duplicates('_key')

merged2 = mf2.merge(ter2[['_key','Regular Plan - Total TER (%)','Direct Plan - Total TER (%)']], on='_key', how='left')
m2 = merged2['Regular Plan - Total TER (%)'].notna().sum()
print(f'Fuzzy join:  {m2}/{len(mf)} ({100*m2/len(mf):.1f}%)')
print('\nTER Regular stats:')
print(merged2['Regular Plan - Total TER (%)'].describe().round(4))
print('\nTER Direct stats:')
print(merged2['Direct Plan - Total TER (%)'].describe().round(4))
