import pandas as pd


df = pd.concat(
    map(pd.read_csv, ['proportional_study_deltabank.csv',
                      'proportional_study_deltacredit.csv',
                      'proportional_study_deltagauss.csv',
                      'proportional_study_deltahc.csv',
                      'proportional_study_deltalc.csv']), ignore_index=True)
df.to_csv('proportional_study_delta.csv')


df2 = pd.concat(
    map(pd.read_csv, ['proportional_taskbank.csv',
                      'proportional_taskcredit.csv',
                      'proportional_taskgauss.csv',
                      'proportional_taskhc.csv',
                      'proportional_tasklc.csv']), ignore_index=True)
df2.to_csv('proportional_task.csv')