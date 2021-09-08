#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix ,accuracy_score,mean_squared_error,classification_report,r2_score,roc_curve,roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVR,SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# In[33]:


df=pd.read_csv("Heart.csv")
df.head()


# In[34]:


y=df["target"]
x=df.drop(["target"],axis=1)


# In[46]:


x


# In[47]:


y


# In[48]:


#modeltuning
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.20,
                                               random_state=50)


# In[49]:


log_model=LogisticRegression(solver="liblinear").fit(x,y)


# In[50]:


svm=SVC()


# In[51]:


svm_params={"C":np.arange(1,10),#ceza parametresi
           "kernel":["linear","rbf"]}
svm_cv_model=GridSearchCV(svm,svm_params,cv=5,n_jobs=-1,verbose=2).fit(x_train,y_train)


# In[52]:


svm_cv_model.best_params_


# In[64]:


svm_tuned_model=SVC(C=1,kernel="linear").fit(x_train,y_train)


# In[54]:


xgb=XGBClassifier()


# In[55]:


xgb_params={"n_estimators":[100,500,1000,2000],
           "subsample":[0.6,0.8,1],
           "max_depth":[3,5,7],
           "learning_rate":[0.1,0.001,0.01]}


# In[56]:


xgb_cv_model=GridSearchCV(xgb,xgb_params,cv=10,n_jobs=-1,verbose=2
                         ).fit(x_train,y_train)


# In[70]:


xgb_cv_model.best_params_


# In[71]:


xgb_tuned_model=XGBClassifier(learning_rate=0.01,
                       max_depth=3,
                       n_estimators=500,
                       subsample=0.6).fit(x_train,y_train)


# In[72]:


rf_model=RandomForestClassifier().fit(x_train,y_train)


# In[73]:


rf=RandomForestClassifier()
rf_params={"n_estimators":[100,200,500,1000],#kullanılacak ağaç sayısı
          "max_features":[3,5,7,8],#değişken sayısı
          "min_samples_split":[2,5,10,20]}#dallanmayı kontrol etme


# In[74]:


rf_cv_model=GridSearchCV(rf,rf_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)


# In[103]:


rf_cv_model.best_params_


# In[104]:


rf_tuned_model=RandomForestClassifier(max_features=3,min_samples_split=10,n_estimators=1000).fit(x_train,y_train)


# In[105]:


knn_model=KNeighborsClassifier().fit(x_train,y_train)


# In[106]:


knn_params={"n_neighbors":np.arange(1,50)}


# In[108]:


knn_cv_model=GridSearchCV(knn_model,knn_params,cv=10).fit(x_train,y_train)


# In[109]:


knn_cv_model.best_params_


# In[110]:


knn_tuned_model=KNeighborsClassifier(n_neighbors=24).fit(x_train,y_train)


# In[111]:


gbm=GradientBoostingClassifier()


# In[112]:


gbm_params={"learning_rate":[0.1,0.01,0.001,0.05],
           "n_estimators":[100,500,300,200,1000],
           "max_depth":[2,3,5,8]}


# In[113]:


gbm_cv_model=GridSearchCV(gbm,gbm_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)


# In[114]:


gbm_cv_model.best_params_


# In[115]:


gbm_tuned_model=GradientBoostingClassifier(learning_rate=0.05,max_depth=2,n_estimators=500).fit(x_train,y_train)


# In[116]:


cart=DecisionTreeClassifier()


# In[117]:


cart_params={"max_depth":[1,3,5,8,10],
            "min_samples_split":[2,3,5,10,20,50]}


# In[118]:


cart_cv_model=GridSearchCV(cart,cart_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)


# In[119]:


cart_cv_model.best_params_


# In[139]:


cart_tuned_model=DecisionTreeClassifier(max_depth=3,
                                  min_samples_split=2).fit(x_train,y_train)


# In[147]:


scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
scaler.fit(x_test)
x_test=scaler.transform(x_test)


# In[148]:


mlpc_params={"alpha":[1,2,3,5,0.1,0.01,0.03,0.005,0.001],
             "hidden_layer_sizes":[(10,10),(100,100,100),(100,100),(3,5)]}


# In[149]:


mlpc=MLPClassifier(solver="lbfgs",activation="logistic")


# In[150]:


mlpc_cv_model=GridSearchCV(mlpc,mlpc_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)


# In[151]:


mlpc_cv_model.best_params_


# In[152]:


mlpc_tuned_model=MLPClassifier(solver="lbfgs",activation="logistic",alpha=2,hidden_layer_sizes=(100,100)).fit(x_train,y_train)


# In[126]:


lgb=LGBMClassifier()


# In[127]:


lgb_params={"learning_rate":[0.001,0.01,0.1],
           "n_estimators":[200,500,100],
           "max_depth":[1,3,5,8]}


# In[128]:


lgb_cv_model=GridSearchCV(lgb,lgb_params,cv=10,n_jobs=-1,verbose=2
                         ).fit(x_train,y_train)


# In[129]:


lgb_cv_model.best_params_


# In[130]:


lgb_tuned_model=LGBMClassifier(learning_rate=0.01,
                        max_depth=3,
                         n_estimators=500
                        ).fit(x_train,y_train)


# In[131]:


catb=CatBoostClassifier()


# In[132]:


catb_params={"iterations":[200,500,1000],
            "learning_rate":[0.01,0.03,0.1],
            "depth":[4,5,8]}


# In[133]:


catb_cv_model=GridSearchCV(catb,catb_params,cv=10,n_jobs=-1,verbose=2
                          ).fit(x_train,y_train)


# In[134]:


catb_cv_model.best_params_


# In[135]:


catb_tuned_model=CatBoostClassifier(depth=4,iterations=1000,
                              learning_rate=0.01).fit(x_train,y_train)


# In[153]:


modeller=[log_model,
          svm_tuned_model,
          xgb_tuned_model,
          rf_tuned_model,
          knn_tuned_model,
          gbm_tuned_model,
          cart_tuned_model,
          mlpc_tuned_model,
          catb_tuned_model,
          lgb_tuned_model]


# In[154]:


import pandas as pd
sonuc=[]
sonuclar=pd.DataFrame(columns=["Modeller","Accuracy"])


# In[155]:


for i in modeller:
    isimler=i.__class__.__name__
    y_pred=i.predict(x_test)
    dogruluk=accuracy_score(y_test,y_pred)
    sonuc=pd.DataFrame([[isimler,dogruluk*100]],columns=["Modeller","Accuracy"])
    sonuclar=sonuclar.append(sonuc)


# In[156]:


sns.barplot(x="Accuracy",y="Modeller",data=sonuclar,color="y")


# In[157]:


sonuclar


# In[ ]:




