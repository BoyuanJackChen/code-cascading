from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
data = load_data()

# answer:
data1 = pd.DataFrame(np.c_[data['data'], data['target']], columns=np.append(data['feature_names'], ['target']))

# test:
assert isinstance(data1, pd.DataFrame)