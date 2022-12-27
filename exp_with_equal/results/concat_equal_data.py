import pandas as pd


df = pd.concat(
    map(pd.read_csv, ['equal_study_deltabean.csv',
                      'equal_study_deltaiit.csv',
                      'equal_study_deltagauss.csv',
                      'equal_study_deltahc.csv',
                      'equal_study_deltalc.csv']), ignore_index=True)
df.to_csv('equal_study_delta.csv')


df2 = pd.concat(
    map(pd.read_csv, ['equal_taskbean.csv',
                      'equal_taskiit.csv',
                      'equal_taskgauss.csv',
                      'equal_taskhc.csv',
                      'equal_tasklc.csv']), ignore_index=True)
df2.to_csv('equal_task.csv')