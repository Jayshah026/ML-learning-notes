## Feature Extraction

```python
- Feature extraction means creating a smaller set of brand-new features by mathematically transforming/combining the original ones - unlike selection, the output columns are not any of the original columns.
- Key difference from feature selection: selection keeps a subset of existing features unchanged; extraction produces new features that summarize the originals, usually to reduce dimensions.
- PCA (Principal Component Analysis) is the most common extraction technique - it combines correlated features (like Height & Weight, which move together) into fewer new features that still capture most of the information.

import pandas as pd
from sklearn.decomposition import PCA

df = pd.DataFrame({'Height': [150, 160, 170, 180, 190], 'Weight': [50, 55, 65, 75, 85]})

pca = PCA(n_components=1)
extracted = pca.fit_transform(df)
df_extracted = pd.DataFrame(extracted.round(2), columns=['Extracted_Feature'])
print(df_extracted)

OUTPUT :
   Extracted_Feature
0             -25.57
1             -14.80
2              -0.67
3              13.45
4              27.58
```