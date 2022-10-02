# import liberary
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
# preprocessing data
from sklearn.preprocessing import OrdinalEncoder , StandardScaler  
# algorithms 
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split , StratifiedKFold
## score 
from sklearn.metrics import roc_auc_score


train_data =pd.read_csv('../input/tabular-playground-series-may-2022/train.csv').drop(['id'] ,axis=1) 
test_data = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv').drop(['id'] , axis=1) 
sample =pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv')


# check nulls values 
train_data.isna().sum().sort_values(ascending=False)


#spliting features as it is type
numerical_features=[] 
catigorical_features=[] 


for col in train_data.columns:
    if(np.dtype(train_data[col])== 'object'):
        catigorical_features.append(col) 
    else : 
        numerical_features.append(col)
numerical_features = [col for col in numerical_features if col not in ['target']]


print(len(numerical_features))

print(len(catigorical_features))


## feature Engineering  
OE_model = OrdinalEncoder()
def feature_engineering(df):
    df['char_unique']=df['f_27'].apply(lambda x: len(set(x)))
    for i in range(df.f_27.str.len().max()):
        df['f_27_char{}'.format(i+1)]=OE_model.fit_transform(df['f_27'].str.get(i).values.reshape(-1,1))
    return df.drop(['f_27'],axis=1)


train_data = feature_engineering(train_data) 
test_data = feature_engineering(test_data)


## spliting data
y=train_data['target']
train_data = train_data.drop(['target']  ,axis=1) 
X = train_data 
X_test =test_data

##  scalling 
SC_model = StandardScaler()
X=SC_model.fit_transform(X)
X_test = SC_model.fit_transform(X_test)



### models
params = {'boosting_type': 'gbdt',
              'n_estimators': 250,
              'num_leaves': 50,
              'learning_rate': 0.1,
              'colsample_bytree': 0.9,
              'subsample': 0.8,
              'reg_alpha': 0.1,
              'objective': 'binary',
              'metric': 'auc',
              'random_state': 21}


SKF_model = StratifiedKFold(n_splits=5)
LGBM_model = LGBMClassifier(**params)
LGBM_score=[] 
for count , (train_idx , test_idx) in enumerate(SKF_model.split(X,y)):
    X_train = X[train_idx] 
    X_valid = X[test_idx] 
    y_train = y[train_idx]
    y_valid = y[test_idx]
    LGBM_model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)],
                                       verbose=100,
                                       eval_metric=['binary_logloss','auc'])
    y_predict = LGBM_model.predict(X_valid) 
    test_predict = LGBM_model.predict_proba(X_test)[: , 1]
    LGBM_score.append(test_predict)
    print("************ fold(",count+1 , ")**************")
    score = roc_auc_score(y_valid , y_predict)
    print("score : ",score)
    
    
    























