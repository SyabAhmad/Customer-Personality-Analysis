import pandas as pd
from imblearn.combine import SMOTETomek, SMOTEENN

data = pd.read_csv('data\marketing_campaign.csv', delimiter='\t')




print(data["Education"].unique())

data["Education"] = data['Education'].map(
    {
        "Graduation":0,
        "Master":1,
        "PhD":2,
        "2n Cycle":3
    }
)

print(data["Marital_Status"].unique())

data["Marital_Status"] = data['Marital_Status'].map(
    {
        'Single':0,
        'Together':1,
        'Married':2,
        'Divorced':3,
        'Widow':4,
        'Alone':5,
        'Absurd':6,
        'YOLO':7
    }
)

data.drop('ID', axis=1, inplace=True)

print(data.columns)

print(data.head())
print(data['Response'].unique())

print(data.isna().sum())
data.dropna(subset=['Education','Income'],inplace=True)
print(data.isna().sum())

count = data['Response'].value_counts()
print(count)

# Convert 'Dt_Customer' to datetime format
dates = data["Dt_Customer"].apply(lambda x: x.split("-"))
print(dates)

# Extract year, month, and day
data["Yearly_Customer"] = dates.apply(lambda x: x[2])
data["Monthly_Customer"] = dates.apply(lambda x: x[1])
data["Daily_Customer"] = dates.apply(lambda x: x[0])

# Drop the 'DT_Customer' column
data.drop('Dt_Customer', axis=1, inplace=True)
print(data.columns)

y = data['Response']
x = data.drop('Response', axis=1)
smotetomek = SMOTEENN()
XSampled, YSampled = smotetomek.fit_resample(x, y)

data = pd.concat([XSampled, YSampled], axis=1)

count = data['Response'].value_counts()
print(count)

data.to_csv("data.csv", index=False)
