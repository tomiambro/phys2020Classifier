import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from aux_funcs import proc_df, numericalize, fix_missing
from sklearn import metrics
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

df = pd.read_feather('datasets/fitted/af-dataset')

y = df['label']
X = df.drop('label', axis=1)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

df_train = X_train.copy()
df_eval = X_eval.copy()
df_train.insert(loc=len(X_train.columns), column='label', value=y_train)
df_eval.insert(loc=len(X_eval.columns), column='label', value=y_eval)

X_train, y_train, nas = proc_df(df_train, 'label')
X_eval, y_eval, nas = proc_df(df_eval, 'label', na_dict=nas)

m_af = RandomForestClassifier(n_estimators=1000, min_samples_leaf=1, max_features='sqrt', n_jobs=7, oob_score=True)
m_af.fit(X_train, y_train)

print('Training Set Metrics:')
print(f'Precision: {precision_score(y_train, m_af.predict(X_train))}')
print(f'Recall: {recall_score(y_train, m_af.predict(X_train))}')
print(f'F1 Score: {f1_score(y_train, m_af.predict(X_train))}')
print('----------------------------------------------------------------------------------------------')
print('Evaluation Set Metrics:')
print(f'Precision: {precision_score(y_eval, m_af.predict(X_eval))}')
print(f'Recall: {recall_score(y_eval, m_af.predict(X_eval))}')
print(f'F1 Score: {f1_score(y_eval, m_af.predict(X_eval))}')
