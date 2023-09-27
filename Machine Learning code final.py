#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import feature_selection
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier as RF
from xgboost import XGBClassifier as XG

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as MSE

import gc


# In[3]:


train = pd.read_csv(r'D:\UT\MMF\Machine Learning/application_train.csv')
test = pd.read_csv(r'D:\UT\MMF\Machine Learning/application_test.csv')


# In[4]:


#QA
train


# In[5]:


# checking missing data
total_null = train.isnull().count()
percent = (train.isnull().sum()/train.isnull().count()*100)
missing_check  = pd.concat([total_null, percent], axis=1, keys=['Total Null', 'Percent']).sort_values(ascending = False, by = ['Percent'])
missing_check.head(10)


# In[6]:


train.loc[:,['COMMONAREA_MEDI','COMMONAREA_AVG','COMMONAREA_MODE','LIVINGAPARTMENTS_MEDI','NONLIVINGAPARTMENTS_MODE',
             'NONLIVINGAPARTMENTS_AVG','NONLIVINGAPARTMENTS_MEDI','FONDKAPREMONT_MODE',
             'LIVINGAPARTMENTS_MODE','LIVINGAPARTMENTS_AVG','LIVINGAPARTMENTS_MEDI']]


# In[7]:


train['COMMONAREA_MEDI'].fillna(value=train['COMMONAREA_MEDI'].mean(), inplace=True)
train['COMMONAREA_AVG'].fillna(value=train['COMMONAREA_AVG'].mean(), inplace=True)
train['COMMONAREA_MODE'].fillna(value=train['COMMONAREA_MODE'].mean(), inplace=True)
train['LIVINGAPARTMENTS_MEDI'].fillna(value=train['LIVINGAPARTMENTS_MEDI'].mean(), inplace=True)
train['NONLIVINGAPARTMENTS_AVG'].fillna(value=train['NONLIVINGAPARTMENTS_AVG'].mean(), inplace=True)
train['NONLIVINGAPARTMENTS_MEDI'].fillna(value=train['NONLIVINGAPARTMENTS_MEDI'].mean(), inplace=True)
train['LIVINGAPARTMENTS_MODE'].fillna(value=train['LIVINGAPARTMENTS_MODE'].mean(), inplace=True)
train['LIVINGAPARTMENTS_AVG'].fillna(value=train['LIVINGAPARTMENTS_AVG'].mean(), inplace=True)
train['LIVINGAPARTMENTS_MEDI'].fillna(value=train['LIVINGAPARTMENTS_MEDI'].mean(), inplace=True)


# In[8]:


df = train.loc[:,train.notna().all(axis=0)]


# In[9]:


#remove high correlation ： correlation >0.5 or <-0.5 --remove multicolinearity
cor_matrix = train.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.5)]
df = train.drop(columns = to_drop, axis=1)


# In[10]:


# QA
df


# In[11]:


#QA
column_names = list(df.columns.values)
column_names


# In[12]:


len(column_names)


# In[13]:


def kde_plot (df, var_name):
    plt.figure(figsize = (12, 6))

    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.loc[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.loc[df['TARGET'] == 1, var_name], label = 'TARGET == 1')

    # label the plot
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend()


# In[14]:


#QA check
plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_CREDIT")
ax = sns.distplot(df["AMT_CREDIT"])


# In[15]:


kde_plot(df, 'AMT_CREDIT')


# In[16]:


df=pd.get_dummies(df)


# In[17]:


df


# In[18]:


#remove infinit effect
df.replace([np.inf, -np.inf], np.nan)

for col in list(df.columns):
    df[col].fillna(0, inplace = True)


# In[19]:


#QA
len(column_names)


# In[20]:


# random forest model
fig1 = plt.figure(figsize=[12,12])
ax1 = fig1.add_subplot(111,aspect = 'equal')

# Extract the ids
ids = df['SK_ID_CURR']

# Extract the labels for training
labels = df['TARGET']

features = df.drop(columns = ['SK_ID_CURR', 'TARGET'])

print('Training Data Shape: ', features.shape)

# Extract feature names
feature_names = list(features.columns)

# Convert to np arrays
training_matrix = np.array(features)

# Create the kfold object
k_fold = KFold(n_splits = 5, shuffle = True, random_state = 50)

#make our model object
model = RandomForestClassifier(n_estimators=100, 
                               max_depth = 3, min_samples_split = 200, min_samples_leaf = 50)

# Empty array for feature importances
feature_importance_values = np.zeros(len(feature_names))

tprs = []
aucs = []


#fold counter
i=1

for train,test in k_fold.split(features,labels):
    prob = model.fit(features.iloc[train],labels.iloc[train]).predict_proba(features.iloc[test])[:,1]
    fpr, tpr, t = roc_curve(labels.iloc[test], prob)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC AUC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i= i+1
    
plt.legend(loc="lower right")
plt.show()


# In[21]:


# xgboost model
fig1 = plt.figure(figsize=[12,12])
ax1 = fig1.add_subplot(111,aspect = 'equal')

# Extract the ids
ids = df['SK_ID_CURR']

# Extract the labels for training
labels = df['TARGET']

features = df.drop(columns = ['SK_ID_CURR', 'TARGET'])

print('Training Data Shape: ', features.shape)

# Extract feature names
feature_names = list(features.columns)

# Convert to np arrays
training_matrix = np.array(features)

# Create the kfold object
k_fold = KFold(n_splits = 5, shuffle = True, random_state = 50)

#make our model object
xgboost=xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic', booster='gbtree')
 
# Empty array for feature importances
feature_importance_values = np.zeros(len(feature_names))

tprs = []
aucs = []


#fold counter
i=1

for train,test in k_fold.split(features,labels):
    prob = xgboost.fit(features.iloc[train],labels.iloc[train]).predict(features.iloc[test])
    fpr, tpr, t = roc_curve(labels.iloc[test], prob)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC AUC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i= i+1
    
plt.legend(loc="lower right")
plt.show()


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20)


# In[23]:


from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def roc_curve(probabilities,classification):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list
    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    Class_df = pd.DataFrame([probabilities,classification])
    Class_df = Class_df.T
    Class_df.columns = ['Prob','class']

    Class_df = Class_df.sort_values('Prob')

    ThreshHolds = []
    FPR_list = []
    TPR_list = []
    for p in Class_df['Prob']:
        thresh = p

        Class_df['Model_label'] = np.where(Class_df['Prob'] >= thresh,1,0)#Create model label

        TPR_numerator = len(Class_df[(Class_df['class'] == 1) & (Class_df['Model_label'] == 1)])
        TPR_denom = Class_df['class'].sum()
        TPR  = float(TPR_numerator)/TPR_denom

        FPR_numerator = len(Class_df[(Class_df['class'] == 0) & (Class_df['Model_label'] == 1)])
        FPR_denom = len(Class_df[(Class_df['class'] == 0)])
        FPR = (float(FPR_numerator)/FPR_denom)

        #append to the lists
        ThreshHolds.append(thresh)
        FPR_list.append(FPR)
        TPR_list.append(TPR)
    return TPR_list,FPR_list,ThreshHolds

from sklearn.metrics import roc_curve
def plot_roc(X,y,models,ax):
    X_train, X_test, y_train, y_test= train_test_split(X,y,shuffle = True)


    for i,model in enumerate(models):
        model = model
        model.fit(X_train,y_train)
        test_probs = model.predict_proba(X_test)
        FPR_list,TPR_list,ThreshHolds = roc_curve(y_test,test_probs[:,0])
        ax.plot(TPR_list,FPR_list,
                 linestyle = '--',
                 linewidth = 2,
                 label = model.__class__.__name__)
    ax.plot([0,1],[0,1],ls = '--',c = 'navy',lw = 3)
    ax.grid(alpha = .3,ls = '-',c= 'g')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    return ax


# In[24]:


_,ax = plt.subplots(1,1,figsize = (10,10))
models = [RF(n_estimators=50), XG()]                                             
plot_roc(features,labels,models,ax)


# In[55]:


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def standard_confusion_matrix(y_true,y_predict):
    tp = np.sum((y_predict == 1) & (y_predict == y_true))
    fp = np.sum((y_predict == 1) & (y_true == 0))
    fn = np.sum((y_predict == 0) & (y_true == 1))
    tn = np.sum((y_predict == 0) & (y_true == y_predict))
    confusion_matrix = np.array([[tp,fn],[fp,tn]])
    return confusion_matrix,fp,tp
    # """Make confusion matrix with format:
    #               -----------
    #               | TP | FP |
    #               -----------
    #               | FN | TN |
    #               -----------
    # Parameters
    # ----------
    # y_true : ndarray - 1D
    # y_pred : ndarray - 1D
    # Returns
    # -------
    # ndarray - 2D
    # """
    # [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    # return np.array([[tp, fp], [fn, tn]])

def profit_curve(cost_benefit, predicted_probs, labels):
    '''
    INPUTS:
    cost_benefit: your cost-benefit matrix
    predicted_probs: predicted probability for each datapoint (between 0 and 1)
    labels: true labels for each data point (either 0 or 1)
    OUTPUTS:
    array of profits and their associated thresholds
    '''
    idx = np.argsort(predicted_probs)
    predicted_probs= predicted_probs[idx]
    #predicted_probs = np.insert(predicted_probs,-1,1)

    labels = labels[idx]
    pred_temp = np.zeros(len(labels))
    thresholds = predicted_probs
    thresholds = np.insert(predicted_probs,0,0)

    cost = []
    for thresh in thresholds:

        pred_temp = np.zeros(len(labels))
        pred_temp[predicted_probs > thresh] = 1
        pred_temp[predicted_probs <= thresh] = 0
        conf, fpr,tpr,= standard_confusion_matrix(np.array(labels),np.array(pred_temp))

        cost.append(np.sum((conf*cost_benefit))/len(labels))


    return (np.array([cost,thresholds]))

def plot_profit_curve(model, cost_benefit, X_train, X_test, y_train, y_test,ax):
    model = model
    model.fit(X_train,y_train)
    test_probs = model.predict_proba(X_test)
    profits = profit_curve(cost_benefit, test_probs[:,1], y_test.values)
    profits = list(reversed(profits[0,:]))
    p = np.linspace(0,len(profits)/8,len(profits))

    ax.plot(p,profits,label=model.__class__.__name__)
    ax.grid(alpha = .4,color = 'r',linestyle = ':')
    ax.set_xlabel('Number of Test instances (decreasing by score)')
    ax.set_ylabel('Profit')
    ax.set_title('Profit Curves')
    return model.predict(X_test),profits,p


# In[80]:


# cost matrix calculation for scenarios
#1 true positive
predic1_actual1= 0
#2 true nagative (earn interest)
predic0_actual0= (interest_rate)*amount_balance_avg
#3 false positive: loss = opportunity cost
predic1_actual0= -(interest_rate)*amount_balance_avg
#4 false negative: loss = partial loan
predic0_actual1= -(interest_rate + 1)*current_credit


# In[81]:


amount_balance_avg


# In[75]:


current_credit


# In[82]:


#Define the Cost Matrix
cost_matrix = np.array([[predic1_actual1, predic1_actual0], [predic0_actual1, predic0_actual0]]).T


# In[83]:


cost_matrix


# In[84]:


#Make the plot
_,ax = plt.subplots(1,1,figsize = (10,5))
models = [RF(n_estimators=100),XG()]

#plot out the profit curve for each model
for model in models:
    m,profits,p = plot_profit_curve(model, cost_matrix, X_train, X_test,                      y_train, y_test,ax)
    
    print('The percentage of correct classification for \nmodel:',
          model.__class__.__name__,
          'is: ',
          np.round(np.sum(m == y_test)/len(y_test)*100,3),
         '%')
    print('-'*50)

ind = np.argmax(profits)
ax.axvline(p[ind],0,np.max(profits),           linestyle = '--', color = 'k',label = 'Max Profit')

ax.legend();


# In[85]:


print(np.sum(labels == 1)/len(labels)*100,'% of the users target in this data set')


# # Make Profit Curve & Choose optimal cut-off (threshold value): RF model
# The optimal cut off would be where tpr is high and fpr is low
# tpr - (1-fpr) is zero or near to zero is the optimal cut off point

# In[86]:


optimal_proba_cutoff = sorted(list(zip(np.abs(tpr - fpr), t)), key=lambda i: i[0], reverse=True)[0][1]


# In[87]:


optimal_proba_cutoff


# In[68]:


optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = t[optimal_idx]
print("Threshold value is:", optimal_threshold)


# The optimal cut off point is 0.07871018, so anything above this can be labeled as 1 else 0. 

# # Get cost and benefit assumptions: cost matrix

# In[28]:


cash_balance = pd.read_csv (r'D:\UT\MMF\Machine Learning/credit_card_balance.csv')


# In[30]:


cash_balance = pd.read_csv (r'D:\UT\MMF\Machine Learning/credit_card_balance.csv')
bureau = pd.read_csv(r'D:\UT\MMF\Machine Learning/bureau.csv')
pre_appl=pd.read_csv(r'D:\UT\MMF\Machine Learning/previous_application.csv')


# In[31]:


bureau.loc[:,['AMT_CREDIT_SUM']]


# In[32]:


bureau_agg = bureau.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg([np.mean, np.median,
                     np.min, np.max]).reset_index()
bureau_agg.head()


# In[33]:


bureau_agg['AMT_CREDIT_SUM'].mean()


# In[34]:


#variable
current_credit=380739.782262


# In[35]:


cash_balance.loc[:,['AMT_BALANCE','AMT_PAYMENT_TOTAL_CURRENT','AMT_CREDIT_LIMIT_ACTUAL']]


# In[36]:


cash_balance.sort_values(by=['MONTHS_BALANCE'],ascending=False)[cash_balance['SK_ID_CURR']==100028]


# In[37]:


cash_balance_var = cash_balance.groupby('SK_ID_CURR')   .apply(lambda x: pd.Series({
      'amt_balance': x['AMT_BALANCE'].sum(),
      'amt_payment_total_current': x['AMT_PAYMENT_TOTAL_CURRENT'].sum(),
      'amt_credit_limit_actual': x['AMT_CREDIT_LIMIT_ACTUAL'].sum()
  })
)


# In[38]:


# Join to the df dataframe
df = df.merge(cash_balance_var, on = 'SK_ID_CURR', how = 'left')

for col_name in ['amt_balance', 'amt_payment_total_current', 'amt_credit_limit_actual']:
    df[col_name] = df[col_name].fillna(0)

# Fill the missing values with 0 
df.head()


# In[39]:


df.loc[:,['amt_balance', 'amt_payment_total_current', 'amt_credit_limit_actual']].mean()*12/49


# In[40]:


#variables
amount_balance_avg=151860.459012
amt_credit_limit_avg=393199.502072


# In[41]:


pre_appl.loc[:,['RATE_INTEREST_PRIMARY','RATE_INTEREST_PRIVILEGED']].mean()


# In[42]:


#to calculate average interest profit
interest_rate= 0.188357*0.7+0.773503*0.3


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




