import pandas as pd


df = pd.concat(
    map(pd.read_csv, ['equal_study_deltabean.csv',
                      'equal_study_deltaiit.csv',
                      'equal_study_deltagauss.csv',
                      'equal_study_deltahc.csv',
                      'equal_study_deltalc.csv']), ignore_index=True)
df.to_csv('equal_study.csv')

