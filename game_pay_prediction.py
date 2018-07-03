# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA

#定义诊断方式RMSE
def rmse(true_data, prediction):
	diff = np.sqrt(mean_squared_error(true_data, prediction))
	return diff
RMSE  = make_scorer(rmse, greater_is_better=False)
####################数据前处理##################################################################
#train数据
data_type = np.load('data_type.npy').item()
train = pd.read_csv('tap_fun_train.csv',dtype=data_type,parse_dates=['register_time'])
test = pd.read_csv('tap_fun_test.csv',dtype=data_type,parse_dates=['register_time'])
#所有数据
all = pd.concat([train,test],ignore_index=True,sort=False)
#加两个feature
all['dayofweek'] = all['register_time'].dt.weekday
all['hourofday'] = all['register_time'].dt.hour
all.drop(columns = ['user_id','prediction_pay_price','register_time'],axis = 1,inplace = True)

#PCA 数据降维
pca = PCA(n_components=8)
pca_fit = pca.fit_transform(all.drop(columns='pay_price'))
column_name = ['P' + str(i) for i in np.arange(1, 9)]
all_PCA = pd.DataFrame(pca_fit,columns=column_name)
all_PCA=pd.concat([all_PCA,all['pay_price']],axis=1)

# 训练自变量和因变量
n_train=train.shape[0]
X = all_PCA[:n_train]
y = train['prediction_pay_price']-train['pay_price']
# 最终测试PCA自变量
test_X = all_PCA[n_train:].reset_index(drop=True)

#分类数据
y_category = y.copy().round(2)
y_category[y_category==0]=0
y_category[(y_category>0)&(y_category<=300)]=1
y_category[(y_category>300)]=2

# 非0数据
catg_1 = y_category[y_category==1].index
X_1 = X.loc[catg_1]
y_1 = y.loc[catg_1]
catg_2 = y_category[y_category==2].index
X_2 = X.loc[catg_2]
y_2 = y.loc[catg_2]
#####################################模型测试调参#####################################################

####先分类用xgb
xgb_classfier= XGBClassifier()
para_grid = {'learning_rate': [0.1],'n_estimators':[10],'max_depth':[5],'min_child_weight':[2],
			 'subsample': [1], 'colsample_bytree': [1]}
grid = GridSearchCV(xgb_classfier, param_grid = para_grid,cv=5, verbose=10)
grid.fit(X, y_category)
print (grid.best_score_)
print(grid.best_params_)

####1类数据回归选用xgb模型
#1类数据xgb调参,51.58,选用
xgb= XGBRegressor()
para_grid = {'n_estimators':[40],'max_depth':[3],'min_child_weight':[5],
			 'subsample': [1], 'colsample_bytree': [0.9]}
grid = GridSearchCV(xgb, param_grid = para_grid,cv=5, scoring=RMSE,verbose=10)
grid.fit(X_1, y_1)
print (grid.best_params_)
print (grid.best_score_)

#####2类数据选用xgb
outlier = y_2[(y_2>25000)&(X_2['pay_price']<1000)].index
X_2_pay = X_2['pay_price'].drop(outlier).values.reshape(-1,1)

#2类数据xgb调参,1863,137.9
xgb= XGBRegressor()
para_grid = {'n_estimators':[43],'max_depth':[3],'min_child_weight':[5],
			 'subsample': [0.7], 'colsample_bytree': [0.9]}
grid = GridSearchCV(xgb, param_grid = para_grid,cv=5, scoring=RMSE,verbose=10)
grid.fit(X_2_pay, y_2.drop(outlier))
print (grid.best_params_)
print (grid.best_score_)
print (grid.cv_results_['std_test_score'])

#################################模型kf验证#####################################################
y_true = train['prediction_pay_price']
y_pay =  train['pay_price']
kf = KFold(n_splits=5)
mse = []
for train_index, test_index in kf.split(X):
	X_train, X_test = X.loc[train_index], X.loc[test_index]
	y_train, y_test = y_category.loc[train_index], y_category.loc[test_index]
	xgb_classfier = XGBClassifier(n_estimators=10, max_depth=5, min_child_weight=2)
	xgb_classfier.fit(X_train, y_train)
	y_pred= xgb_classfier.predict(X_test)

	se_y_pred = pd.Series(y_pred, index=test_index)
	se_y_pred[se_y_pred == 0] = 0
	idx_test_0 = se_y_pred[se_y_pred == 0].index
	se_pred_0 = se_y_pred[idx_test_0] + y_pay.loc[idx_test_0]

	idx_train_1 = y_train[y_train==1].index
	idx_test_1 = se_y_pred[se_y_pred == 1].index
	idx_train_2 = y_train[y_train==2].index
	idx_test_2 = se_y_pred[se_y_pred == 2].index

	# 1类数据处理
	xgb = XGBRegressor(n_estimators=40, max_depth=3, min_child_weight=5, colsample_bytree=0.9)
	xgb.fit(X_1.loc[idx_train_1], y_1.loc[idx_train_1])
	pred_1 = xgb.predict(X_test.loc[idx_test_1])
	se_pred_1 = pd.Series(pred_1,index=idx_test_1)
	se_pred_1=se_pred_1+y_pay.loc[idx_test_1]

	#2类数据处理
	outlier = y_2[(y_2 > 25000) & (X_2['pay_price'] < 1000)].index
	xgb_2 = XGBRegressor(n_estimators=43, max_depth=3, min_child_weight=5, subsample=0.7,colsample_bytree=0.9)
	if outlier.isin(X_2.loc[idx_train_2].index):
		xgb_2.fit(X_2.loc[idx_train_2].drop(outlier)['pay_price'].values.reshape(-1, 1), y_2.loc[idx_train_2].drop(outlier))
		print 'outlier'
	else:
		xgb_2.fit(X_2.loc[idx_train_2]['pay_price'].values.reshape(-1,1), y_2.loc[idx_train_2])
	pred_2 = xgb_2.predict(X_test.loc[idx_test_2]['pay_price'].values.reshape(-1,1))
	se_pred_2 = pd.Series(pred_2,index=idx_test_2)
	se_pred_2=se_pred_2+y_pay.loc[idx_test_2]

	pred_combine = pd.concat([se_pred_0,se_pred_1,se_pred_2]).sort_index()
	rmse_kf = np.sqrt(mean_squared_error(pred_combine, y_true.loc[y_test.index]))
	print rmse_kf
	mse.append(rmse_kf)
print 'Mean RMSE:',np.mean(mse),'std:',np.std(mse), mse

############################################预测及提交##########################################
#Test数据分类
xgb_classfier = XGBClassifier(n_estimators=10, max_depth=5, min_child_weight=2)
xgb_classfier.fit(X, y_category)
test_pred= xgb_classfier.predict(test_X)
se_test_pred = pd.Series(test_pred, index=test_X.index)
se_test_pred[se_test_pred == 0] = 0
idx_test_0 = se_test_pred[se_test_pred == 0].index
se_test_pred_0 = se_test_pred[idx_test_0] + test_X['pay_price'].loc[idx_test_0]

#1、2类的idx
idx_test_1 = se_test_pred[se_test_pred == 1].index
test_X_1 = test_X.loc[idx_test_1]
idx_test_2 = se_test_pred[se_test_pred == 2].index
test_X_2 = test_X.loc[idx_test_2]

# 1类数据处理
xgb = XGBRegressor(n_estimators=40, max_depth=3, min_child_weight=5, colsample_bytree=0.9)
xgb.fit(X_1, y_1)
test_pred_1 =xgb.predict(test_X_1)
se_test_pred_1 = pd.Series(test_pred_1,index=idx_test_1)
se_test_pred_1 = se_test_pred_1 + test_X['pay_price'].loc[idx_test_1]

# 2类数据处理
outlier = y_2[(y_2 > 25000) & (X_2['pay_price'] < 1000)].index
X_2_pay = X_2['pay_price'].drop(outlier).values.reshape(-1,1)
xgb_2 = XGBRegressor(n_estimators=43, max_depth=3, min_child_weight=5, subsample=0.7,colsample_bytree=0.9)
xgb_2.fit(X_2_pay, y_2.drop(outlier))
test_pred_2 =xgb_2.predict(test_X_2['pay_price'].values.reshape(-1,1))
se_test_pred_2 = pd.Series(test_pred_2,index=idx_test_2)
se_test_pred_2 = se_test_pred_2 + test_X['pay_price'].loc[idx_test_2]
test_pred_combine = pd.concat([se_test_pred_0,se_test_pred_1,se_test_pred_2]).sort_index()

my_submission = pd.DataFrame({'user_id': test.user_id, 'prediction_pay_price': test_pred_combine})
my_submission = my_submission[['user_id','prediction_pay_price']]
my_submission=my_submission.round(2)
my_submission.to_csv('prediction.csv', index=False,encoding='utf-8')
