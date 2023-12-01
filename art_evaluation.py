import pandas as pd
from joblib import load
X_tst = pd.read_csv('test.csv', header=None)

clf = load('model/art_pred.joblib')
print('Prediction result: ', clf.predict(X_tst)[0])
