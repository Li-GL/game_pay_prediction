import pandas as pd
import numpy as np
df = pd.read_csv('tap_fun_train.csv')

gl_int = df.select_dtypes(include=['int64'])
converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')
gl_float = df.select_dtypes(include=['float'])
converted_float = gl_float.apply(pd.to_numeric,downcast='float')

optimized_df = df.copy()

optimized_df[converted_int.columns] = converted_int
optimized_df[converted_float.columns] = converted_float

dtypes = optimized_df.dtypes
dtypes_col = dtypes.index
dtypes_type = [i.name for i in dtypes.values]
column_types = dict(zip(dtypes_col, dtypes_type))
np.save('data_type.npy',column_types)