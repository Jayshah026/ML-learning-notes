# Feature Scaling - Standardization

## What is Feature Scaling

```python
- Feature Scaling is a technique to bring all numerical features onto a comparable scale.
- Needed because columns like age (20-60) and salary (20000-100000) have a huge scale difference.
- Two main techniques: Normalization and Standardization.
- Here we are covering Standardization.

import pandas as pd

df = pd.DataFrame({"age": [25, 30, 35, 40, 45],
                   "salary": [30000, 45000, 53000, 70000, 90000]})
print(df.describe().loc[["min", "max"]])

OUTPUT :
       age   salary
min   25.0  30000.0
max   45.0  90000.0
```

## Standardization Formula (Z-score)

```python
- Standardization rescales each feature so that mean = 0 and standard deviation = 1.
- Formula: z = (x - mean) / sd
- x = original value, mean = mean of that column, sd = standard deviation of that column.
- The result is NOT limited to a fixed range. Values can be below -3 or above +3.
- It does not change the shape of the distribution, it only shifts and rescales it.
- A bounded range like 0 to 1 is Normalization (Min-Max), not Standardization.

import numpy as np

x = np.array([25, 30, 35, 40, 45])
z = (x - x.mean()) / x.std()
print(z.round(3))

OUTPUT :
[-1.414 -0.707  0.     0.707  1.414]
```

## Example 1 - Manual Standardization of Age and Salary

```python
- Apply the formula column by column using that column's own mean and sd.
- After standardizing, both columns have mean 0 and sd 1, so they are on the same scale.
- ddof=0 is used because sklearn's StandardScaler uses population sd.
- pandas .std() uses ddof=1 by default, which gives slightly different numbers.

import pandas as pd

df = pd.DataFrame({"age": [25, 30, 35, 40, 45],
                   "salary": [30000, 45000, 53000, 70000, 90000]})
z = (df - df.mean()) / df.std(ddof=0)
print(z.round(3))
print("mean:", z.mean().round(3).tolist())
print("std :", z.std(ddof=0).round(3).tolist())

OUTPUT :
     age  salary
0 -1.414  -1.332
1 -0.707  -0.608
2  0.000  -0.222
3  0.707   0.599
4  1.414   1.564
mean: [0.0, -0.0]
std : [1.0, 1.0]
```

## Example 2 - Why KNN Needs It (Distance Before vs After)

```python
- KNN predicts using Euclidean distance between rows.
- Raw salary differences (thousands) completely overwhelm raw age differences (tens).
- Person A (25 yrs, 50000) and B (60 yrs, 51000): 35 years apart but only 1000 salary apart.
- Person A and C (26 yrs, 58000): almost the same age but 8000 salary apart.
- Raw distance says B is much closer to A, only because salary dominates.
- After standardization, age gets a fair say and the ranking flips: C becomes closer to A.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({"age": [25, 60, 26, 35, 45],
                     "salary": [50000, 51000, 58000, 53000, 90000]},
                    index=["A", "B", "C", "D", "E"])

A, B, C = data.loc["A"].values, data.loc["B"].values, data.loc["C"].values
print("raw A-B:", np.linalg.norm(A - B))
print("raw A-C:", np.linalg.norm(A - C))

scaled = pd.DataFrame(StandardScaler().fit_transform(data),
                      index=data.index, columns=data.columns)
print("scaled A-B:", np.linalg.norm(scaled.loc["A"] - scaled.loc["B"]).round(3))
print("scaled A-C:", np.linalg.norm(scaled.loc["A"] - scaled.loc["C"]).round(3))

OUTPUT :
raw A-B: 1000.6123125366787
raw A-C: 8000.000062499999
scaled A-B: 2.678
scaled A-C: 0.537
```

## StandardScaler in sklearn

```python
- StandardScaler automates the formula (x - mean) / sd for every column.
- fit() learns mean_ and scale_ (the sd) from the data.
- transform() applies the formula using the learned values.
- fit_transform() does both in one step.
- The result is a NumPy array, so column names are lost unless you wrap it in a DataFrame.

import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({"age": [25, 30, 35, 40, 45],
                   "salary": [30000, 45000, 53000, 70000, 90000]})
sc = StandardScaler().fit(df)
print(sc.mean_)
print(sc.scale_)
print(sc.transform(df).round(3))

OUTPUT :
[3.50e+01 5.76e+04]
[7.07106781e+00 2.07132808e+04]
[[-1.414 -1.332]
 [-0.707 -0.608]
 [ 0.    -0.222]
 [ 0.707  0.599]
 [ 1.414  1.564]]
```

## Fit on Train, Transform on Test (Avoiding Data Leakage)

```python
- Always learn mean and sd from the training set only.
- Apply those same values to the test set using transform(), never fit on test data.
- Fitting on the full dataset leaks test information into training (data leakage).
- Correct pattern: fit_transform on X_train, only transform on X_test.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = np.arange(20).reshape(10, 2).astype(float)
X_train, X_test = train_test_split(X, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)
print(sc.mean_)
print(X_test_s.round(3))

OUTPUT :
[8.85714286 9.85714286]
[[-0.805 -0.805]
 [ 1.184  1.184]
 [-0.142 -0.142]]
```

## Which Algorithms Need Scaling

```python
- Distance-based: KNN, K-Means, SVM. They compute distances, so large-scale columns dominate.
- Variance-based: PCA. It looks for directions of maximum variance, so a large-scale column hijacks the components.
- Gradient descent based: Linear/Logistic Regression, Artificial Neural Networks.
  Gradient descent is an optimizer, not a model. Unscaled features make the loss surface stretched, so convergence is slow.
- Scaling is decided by the algorithm type, not only by the size of the scale gap.

needs_scaling = ["KNN", "K-Means", "SVM", "PCA",
                 "Linear/Logistic Regression (gradient descent)", "Neural Networks"]
for algo in needs_scaling:
    print(algo)

OUTPUT :
KNN
K-Means
SVM
PCA
Linear/Logistic Regression (gradient descent)
Neural Networks
```

## Why Tree-Based Models Do Not Need Scaling

```python
- Applies to Decision Tree, Random Forest, Gradient Boosting, XGBoost.
- A tree splits on one feature at a time using a threshold, e.g. salary > 61500.
- Standardization is monotonic: it preserves the order of values, so the same row groupings exist.
- Only the threshold number changes (61500 becomes about 0.14), the left/right split stays identical.
- Trees never compute distances or gradients across features, so scale has no effect.

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

X = np.array([[25, 30000], [30, 45000], [35, 53000], [40, 70000],
              [45, 90000], [28, 35000], [50, 95000], [33, 48000]], dtype=float)
y = np.array([0, 0, 0, 1, 1, 0, 1, 0])
Xs = StandardScaler().fit_transform(X)

t_raw = DecisionTreeClassifier(random_state=0).fit(X, y)
t_scaled = DecisionTreeClassifier(random_state=0).fit(Xs, y)
print(t_raw.predict(X))
print(t_scaled.predict(Xs))
print("same predictions:", (t_raw.predict(X) == t_scaled.predict(Xs)).all())
print("root threshold raw:", t_raw.tree_.threshold[0])
print("root threshold scaled:", round(t_scaled.tree_.threshold[0], 3))

OUTPUT :
[0 0 0 1 1 0 1 0]
[0 0 0 1 1 0 1 0]
same predictions: True
root threshold raw: 61500.0
root threshold scaled: 0.143
```