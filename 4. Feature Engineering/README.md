# Feature Engineering 

## What is Feature Engineering? 

```python

- It is a process to extract features from a raw data with the help of domain knowledge.

- That features can be used in improving the performance of machine learning algorithms.

```

## Types of the Feature Engineering 

```python 

1. Feature transformation 

2. Feature construcion 

3. Feature selection 

4. Feature extraction

```

## Basic Knowledge about all 4 types of Feature Engineering 

# Feature Transformation

## Missing Value Imputation

```python
- Missing value imputation means filling NaN entries with a sensible value instead of leaving them blank, because most ML algorithms need a number in every cell to work.
- The decision to drop vs fill depends on how much of that specific column is missing, not the size of the whole dataset - a few missing rows out of thousands can be dropped, but heavy missingness needs to be filled.
- Numeric columns are usually filled with mean or median, categorical columns with mode (most frequent value).
- Median is safer than mean when the column has outliers, since mean gets pulled toward extreme values.

import pandas as pd
import numpy as np

df = pd.DataFrame({'Name': ['A', 'B', 'C', 'D'], 'Age': [25, 30, np.nan, 35]})
df['Age_filled'] = df['Age'].fillna(df['Age'].median())
print(df)

OUTPUT :
  Name   Age  Age_filled
0    A  25.0        25.0
1    B  30.0        30.0
2    C   NaN        30.0
3    D  35.0        35.0
```

## Handling Categorical Features

```python
- ML algorithms (like scikit-learn's) only work with numbers, so text categories must be converted to numeric form before training.
- One-Hot Encoding: creates a separate 0/1 column for each category - used when categories have no natural order (e.g. City names).
- Label/Ordinal Encoding: assigns a single integer (0, 1, 2...) to each category - used when categories do have a natural order (e.g. Small < Medium < Large).
- The output is not always squeezed into 0-1 - that's specific to one-hot encoding, ordinal encoding can give any integer.

import pandas as pd

df = pd.DataFrame({'City': ['Lisbon', 'Berlin', 'Lisbon']})
df_encoded = pd.get_dummies(df, columns=['City'], dtype=int)
print(df_encoded)

OUTPUT :
   City_Berlin  City_Lisbon
0            0            1
1            1            0
2            0            1
```

## Outlier Detection

```python
- An outlier is a data point far outside the range where most of the data sits (e.g. most values are 20-30, but a few are 80-100).
- They matter because they distort statistics like mean and variance, and mislead distance-based algorithms like KNN.
- A common detection method is IQR: anything below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is flagged as an outlier.
- Outliers aren't always removed - sometimes they're capped or transformed instead, since they may be real, unusual data rather than an error.

import pandas as pd

data = pd.Series([22, 24, 25, 23, 26, 90])
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = data[(data < lower) | (data > upper)]
print(outliers)

OUTPUT :
5    90
dtype: int64
```

## Feature Scaling

```python
- Feature scaling brings all numeric columns onto a similar range, so no column dominates just because its raw numbers happen to be bigger.
- Example: Salary (20,000-100,000) vs Age (20-60) - without scaling, distance-based algorithms treat Salary as far more important simply because its numbers are larger.
- Standardization (StandardScaler) rescales values to have mean = 0 and std = 1.
- Normalization (MinMaxScaler) squeezes values into a fixed range, usually 0 to 1.

import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({'Age': [20, 30, 40], 'Salary': [20000, 50000, 100000]})
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns).round(2)
print(df_scaled)

OUTPUT :
    Age  Salary
0 -1.22   -1.11
1  0.00   -0.20
2  1.22    1.31
```
