from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

X = df[['age', 'IMD2019_decile', 'household_size', 'mor_travel', 'aft_travel',
       '#car',  'ethnicity_Pakistani','ethnicity_White British', 'gender_Female']]
Y = df['Y2']
X_train, X_test, y_train, y_test = train_test_split(X,Y,stratify = Y)


# minmax: [0,1]
X_minmax_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X),columns=X.columns)
X_minmax_scaled

## z-score normalization : most of the data will fall in [-3.3]
X_std_scaled = pd.DataFrame(StandardScaler().fit_transform(X),columns=X.columns)
X_std_scaled.head()

#---------------------------------Missing value analysis------------------------------------------------
import missingno as msno
import pandas as pd
data=pd.read_csv('csv_version.csv')
msno.matrix(data)
import matplotlib.pyplot as plt
plt.savefig('C:\\Users\\tssw\\Desktop\\project\\data\\plots\\missing value.png')
msno.bar(data)
plt.savefig('C:\\Users\\tssw\\Desktop\\project\\data\\plots\\missing value distribution.png')
msno.dendrogram(data)
plt.savefig('C:\\Users\\tssw\\Desktop\\project\\data\\plots\\missing value tree.png')
