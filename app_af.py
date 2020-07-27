import sys
from xverse.transformer import WOE
from aux_funcs import *
from pandas_summary import DataFrameSummary
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy as hc
from sklearn import metrics
from sklearn.metrics import f1_score, fbeta_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import plotly.express as px
import streamlit as st

lead = 'lead2'

st.title("Atrial Fibrilation Detector using Single Lead ECG Data")
"""
	Using the dataset provided by the 2020 Physionet Challenge we've developed an Atrial Fibrilation Detector trained to
	identify AF diagnosed patiences from a dataset containing patiances with different pathologies like: PAC, RBBB, I-AVB,
	PVC, LBBB, STD, STE and healthy individuals.

	Although data from 12-lead ECG was provided, for this first analysis we've only used the lead 2 data and we've processed
	the signals in order to create a dataframe consisting of features we believe will help us classify.

"""
@st.cache
def load_data():
	df = pd.read_feather('datasets/corrected/pyhs-raw-lead2-corrected')
	df['PT_duration'] = df['mean_T_Offsets'] - df['mean_P_Onsets']
	df.drop(['mean_T_Offsets', 'mean_P_Onsets'], axis=1, inplace=True)
	return df

df_raw = load_data()

"""
	Raw dataframe with Lead 2 data
"""

df_raw.T

labels = pd.get_dummies(df_raw['label']).describe()

"""
	Let's describe the labels in the data:
"""

labels

y = df_raw['label']
X = df_raw.drop('label', axis=1)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

df_train = X_train.copy()
df_eval = X_eval.copy()
df_train.insert(loc=len(X_train.columns), column='label', value=y_train)
df_eval.insert(loc=len(X_eval.columns), column='label', value=y_eval)

"""
	We use 20% as training data and rename every other category as Non-AF:
"""

df_train.loc[df_train.label != 'AF', 'label'] = 'Non-AF'
df_eval.loc[df_eval.label != 'AF', 'label'] = 'Non-AF'

labels = pd.get_dummies(df_train['label']).describe()
labels

X_train, y_train, nas = proc_df(df_train, 'label')
X_eval, y_eval, nas = proc_df(df_eval, 'label', na_dict=nas)

m_af = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_features='sqrt', n_jobs=7, oob_score=True)
m_af.fit(X_train, y_train)

"""
	Let's build our Random Forest and see the results.

	## Evaluation Metrics
	We are using the sklearn [ f1_score, fbeta_score ] with beta = 2

	This are the evaluation metrics we are actually interested in.
"""

st.markdown(' #### Training Metrics: ')
'F1 and F2 Scores: ', f1_score(y_train, m_af.predict(X_train)), fbeta_score(y_train, m_af.predict(X_train), beta=2)
st.markdown(' #### Validation Metrics: ')
'F1 and F2 Scores: ', f1_score(y_eval, m_af.predict(X_eval)), fbeta_score(y_eval, m_af.predict(X_eval), beta=2)

"""
	## Removing Redundant Features

	By calculating a Dendrogram we look for features that may be providing the same information so we can remove them and end up with a cleaner model.
"""

corr = np.round(scipy.stats.spearmanr(X_train).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,15))
dendrogram = hc.dendrogram(z, labels=X_train.columns, orientation='left', leaf_font_size=16)
st.pyplot(plt, use_container_width=True)

to_drop = ['var_R_Peaks', 'var_RR']
X_train_drop = X_train.drop(to_drop, axis=1)
X_eval_drop = X_eval.drop(to_drop, axis=1)

"""
	## Looking at Feature Importance
	### MDI

	We calculate the feature importance using the MDI (Mean Decrease in Impurity) method, with is the default for sklearn's Random Forests
"""

m_af = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_features='sqrt', n_jobs=7, oob_score=True)
m_af.fit(X_train_drop, y_train)

fi_mdi = rf_feat_importance(m_af, X_train_drop)
fig = plt.figure(figsize=(16,10))
fi_mdi.plot('cols', 'imp', 'barh', figsize=(12,8), legend=False)
st.pyplot(plt, use_container_width=True)

"""
	### MDA

	We now calculate feature importance using the MDA (Mean Decrease in Accuracy) method to compare. We can pass our own score algorithm so we will use the beta score since is the metric we will be optimizing to.

	For more information on how these are calculated please refer to [this link](https://medium.com/the-artificial-impostor/feature-importance-measures-for-tree-models-part-i-47f187c1a2c3).
"""

score_f = make_scorer(f1_score)
res = permutation_importance(m_af, X_train_drop, y_train, scoring=score_f, n_repeats=5, random_state=42, n_jobs=7)

fig = plt.figure(figsize=(16,10))
fi_mda = pd.DataFrame({'cols':X_train_drop.columns, 'imp':res.importances_mean.T}).sort_values('imp', ascending=False)
fi_mda.plot('cols', 'imp', 'barh', figsize=(12,8), legend=False)
plot_fi(fi_mda)
st.pyplot(plt, use_container_width=True)

"""
	Although not the same, the results are similar and make sense from a physiological point of view.

	### Information Value

	Let's take a look at information value using Xverse package
"""

clf = WOE()
clf.fit(X_train_drop, y_train)
iv = clf.iv_df

fig = plt.figure(figsize=(16,10))
iv_xverse = pd.DataFrame({'cols':list(iv['Variable_Name']), 'imp':iv['Information_Value'].T}).sort_values('imp', ascending=False)
iv_xverse.plot('cols', 'imp', 'barh', figsize=(12,8), legend=False)
st.pyplot(plt)

"""
	Again, we see that the most meaningful features appear to be the same.
	
	### Let's plot a few interesting variables

	As we will see through this notebook, HRV and mean_P_Peaks are some of the most interesting features in terms of importance and predictive power. Also, age plays a critical role in AF so we'll examine it as well.

	#### HRV per label
"""

df = df_raw.loc[df_raw['age'] > 0]
fig = px.scatter(df, x="label", y="HRV", color='age', opacity=0.5)
fig.update_layout(yaxis = dict(
      range=[200,1100]))
# fig.show()
st.plotly_chart(fig)

"""
	#### Mean P peak value per label
"""

df = df_raw.loc[df_raw['age'] > 0]
fig = px.scatter(df, x="label", y="mean_P_Peaks", color='age', opacity=0.5)
fig.update_layout(yaxis = dict(
      range=[-200,500]))
# fig.show()
st.plotly_chart(fig)

"""
	### Age per label
"""

df = df_raw.loc[df_raw['age'] > 0]
fig = px.scatter(df, x="label", y="age", color='label', opacity=0.4)
fig.update_layout(yaxis = dict(
      range=[0,120]))
# fig.show()
st.plotly_chart(fig)

"""
	### HRV against age
"""

df = df_raw.loc[df_raw['age'] > 0]
fig = px.scatter(df, x="age", y="HRV", color='label', opacity=0.7)
fig.update_layout(yaxis = dict(
      range=[200,1100]))
# fig.show()
st.plotly_chart(fig)

"""
	### Histogram of Age for each label
"""

fig = px.histogram(df_raw, orientation='v', x="age", color='label')
st.plotly_chart(fig)

"""
	## Lets optimize our model based on MDI results
"""

thresh = 0.034
to_keep = list(fi_mdi[fi_mdi['imp'] > thresh].cols)
X_train_keep = X_train_drop[to_keep]
X_eval_keep = X_eval_drop[to_keep]
m_af = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_features='sqrt', n_jobs=7, oob_score=True)
m_af.fit(X_train_keep, y_train)
# print_fscores(m_af, X_eval_keep, y_eval)


st.markdown(' #### Training Metrics: ')
'F1 and F2 Scores: ', f1_score(y_train, m_af.predict(X_train_keep)), fbeta_score(y_train, m_af.predict(X_train_keep), beta=2)
st.markdown(' #### Validation Metrics: ')
'F1 and F2 Scores: ', f1_score(y_eval, m_af.predict(X_eval_keep)), fbeta_score(y_eval, m_af.predict(X_eval_keep), beta=2)