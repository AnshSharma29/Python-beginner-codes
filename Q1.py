import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScalar, normalize, Binarizer, scale



df=pd.DataFrame(np.random.randn(4,3),
index=['a','b','c','d'],
columns=['C1','C2','C3'])


print("Original Dataset: ")
print(df)

df1=df.copy()
scalar=MinMxScalar(feature_range=(0,1))
rescaled_data = scalar.fit_transform(df1)
print(rescaled_data)

df2=df.copy()
normalize_data=normalize(df,norm='l1')
print(normalize_data)

binarizer=Binarizer(threshold=0.55555)
binarized_data=binarizer.fit_transform(df)
print(binarized_data)

standardised_data=scale(df,axis=0)
print(standarised_data)



