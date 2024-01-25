# %%
# compine all steps in Piplines and columntransformer

# Import pandas
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import cross_val_score

import pickle

# %%
# Load dataset
df = pd.read_csv('datasets/cc_approvals.data',header=None)

# %%

print(df[3].value_counts())


# %%
X = df.drop([15] , axis=1)

# labels
values = {'+' : 1, '-':0}
y = df[15].map(values)

# %%
# custom transformer class
class ReplaceToNanDrop(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.replace(['?'], np.nan)
        X.columns = X.columns.astype(str)
        X = X.drop(['11', '13'], axis=1)
        X['1'] = X['1'].astype(float)
        return X

# columns to imputers
def get_columns_dtypes(X):
    '''
    the dtypes of the columns were be changed after the fill nulls step
    '''
    X = ReplaceToNanDrop().fit_transform(X)
    cats = X.select_dtypes(include=['object']).columns
    cons = X.select_dtypes(exclude=['object']).columns
    return {'cats': cats, 'cons': cons}


class LabelEncodingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        for col in X.columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        return X


# %%
pipe_numeric = Pipeline([("numeric_null", SimpleImputer(missing_values=np.nan, strategy='mean')),
                  ("scaler", MinMaxScaler(feature_range=(0, 1)))])

pipe_cat = Pipeline([("cats_null", SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
                  ("encode", LabelEncodingTransformer())])

preprocessing = ColumnTransformer(
    [("numeric",pipe_numeric ,get_columns_dtypes(X)['cons']),
    ("cats",pipe_cat ,get_columns_dtypes(X)['cats'])])

pipe = Pipeline([('replace_to_nan_drop', ReplaceToNanDrop()),
                  ('preprocessing', preprocessing),
                 ('classifier',LogisticRegression(max_iter= 150, tol=0.01))])

# %%
# test in cross val score 
print(cross_val_score(pipe, X,y, cv=2))

# %%
# in all data

pipe.fit(X,y)

# %%
with open("model/model.pkl", "wb") as file:
    pickle.dump(pipe, file)

# %%



