import pickle
import pandas as pd

dic = pickle.load(open('test/Archives/0.05_0.025_0.1_0.2.pkl', 'rb'))
df = pd.DataFrame(dic)
print(df.columns)
df.to_csv('testArchive.csv')
