import pandas as pd

data = pd.read_csv('xxx.csv')
data.head()

## select the needed colomuns
df = data.iloc[:,3:]
df.head()
df.dtypes

## labeling the number variable by mapping
morning_travel_mapping
## discrete label by mapping

morning_travel_mapping={'None':1,
                       '1 time last week':2,
                       '2-3 times last week':3,
                       '4 times last week':4,
                       '5 times last week':5}

afternoon_travel_mapping={'None':1,
                       '1 time last week':2,
                       '2-3 times last week':3,
                       '4 times last week':4,
                       '5 times last week':5}

vclcnt_mapping={'None':1,
               '1':2,
               '2':3,
               '4':4,
               'More than 3':5}

walk_unwillingness_mapping={'Very happy':1,
                           'Quite happy':2,
                           'Quite unhappy':3,
                           'Very unhappy':4}

df1=df
df1['c19c2_chfl_acttrvl_mor']=df1['c19c2_chfl_acttrvl_mor'].map(morning_travel_mapping)
df1['c19c2_chfl_acttrvl_aft']=df1['c19c2_chfl_acttrvl_aft'].map(afternoon_travel_mapping)
df1['c19c2_chsch_vclcnt']=df1['c19c2_chsch_vclcnt'].map(vclcnt_mapping)
df1['c19c2_chsch_10wlk']=df1['c19c2_chsch_10wlk'].map(walk_unwillingness_mapping)

## before one-hot encoding, for convenience, rename the column names.
df1.columns = ['ethnicity','age','gender','IMD2019_decile','household_size','mor_travel','aft_travel','mode_now','mode_before',
              '#car','unwill_walk']

## now do the one-hot encoding

df2=df1

dummy=pd.get_dummies(df2[['ethnicity','gender','mode_now','mode_before']])
df3=df2.drop(['ethnicity','gender','mode_now','mode_before'], axis=1)
df3=df3.join(dummy)
df3.head(9)

df3.to_csv('processed.csv')
