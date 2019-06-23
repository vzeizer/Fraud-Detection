#!/usr/bin/python

import sys
import pickle
import numpy 

import pandas as pd
import numpy as np
# path to import functions
sys.path.append("../tools/")

# appending the outlier path!
#sys.path.append("../outliers/")
# Importing the outlier Cleaner!
#from outlier_cleaner import outlierCleaner

from sklearn.model_selection import train_test_split


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# importing important metrics
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score # the f1_score
from sklearn.metrics import recall_score # the recall 
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

# importing the classifiers!
from sklearn import linear_model # linear classifier!
from sklearn.naive_bayes import GaussianNB # gaussian Naive-Bayes classifier!
from sklearn.svm import SVC # Support Vector Classifier!
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier!
from sklearn.neighbors import KNeighborsClassifier # KNN Classifier
from sklearn.ensemble import RandomForestClassifier # Random Forest Classifier!
from sklearn.ensemble import AdaBoostClassifier # Ada Boost Classifier!


# MinMax Scaler -> Rescaling!

from sklearn.preprocessing import MinMaxScaler

# PCA! Principal Component Analysis!

from sklearn.decomposition import PCA


# importing GridSearchCV
from sklearn.model_selection import GridSearchCV

# importing StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

# Pipeline -> can be used to chain multiple estimators into one!
# -> for convenience and encapsulation
# -> you can grid search over parameters of all estimators in the pipeline at once!


from sklearn.pipeline import make_pipeline
# make_pipeline is a shorthand for constructing pipelines; it takes a variable number
# of estimators and returns a pipeline, filling in the names automatically
from sklearn.pipeline import Pipeline
# the Pipeline is built using a list of ('key','value') pairs, where the 'key' is a string 
# containing the name you want to give this step and 'value' is an estimator object:


# feature selection
from sklearn.feature_selection import SelectKBest # select Kbest features! 
from sklearn.feature_selection import chi2 # the Chi2 used to evaluate data fitting // metric to be used
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile # the percents best features to be used!


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# including the features: 'bonus' and 'exercised_stock_options'
# and many others!
#features_list = ['poi','salary'] # You will need to use more features
#features_list = ['poi','salary','bonus','exercised_stock_options','shared_receipt_with_poi','from_this_person_to_poi','director_fees','deferred_income','long_term_incentive','from_poi_to_this_person']
#features_list = ['poi','salary','bonus','exercised_stock_options','shared_receipt_with_poi','director_fees','deferred_income','long_term_incentive']
features_list = ['poi','salary','bonus','exercised_stock_options','shared_receipt_with_poi','long_term_incentive']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#print data_dict['POWERS WILLIAM']

n_attr=3

#data_dict_df= pd.DataFrame(data_dict.items()) 

df_1= pd.DataFrame.from_dict(data_dict,orient='index')

# taking the features_list columns out of df

df_1= df_1[features_list]
#df_1 = df_1.drop(['poi'], axis=1)


#print df_1['poi'].tolist()

for feat in features_list:
	df_test= df_1[feat].tolist()
	somanull=0
	for i in range(len(df_test)):
		if(df_test[i]=='NaN'):
			somanull+=1
	if(somanull>=0.6*len(df_test)):
		print feat,'esse feature tem mtos NaNs'

#		if(df[df[feat] == 'NaN'].mean() >= 0.4):
#			print feat
"""
'director_fees' tem muitos NaNs
'deferred_income' tem muitos NaNs
"""

def feature_selection(features_list,data_dict):
	"""
		This function selects the best features according to the 
		SelectKbest Method!
		Inputs: 
				-features_list: list of all the (relevant) features to be investigated
				- data_dict: dictionary containing the data!
		Outputs:
				-colnames_selected: the best features, selected by SelectKBest
	"""
	# converting data_dict to a dataframe
	df= pd.DataFrame.from_dict(data_dict,orient='index')

	# taking the features_list columns out of df
	df = df[features_list]

	# taking the 'poi' feature out of df
	df_poi = df['poi']

	# dropping 'poi' column
	df_clean = df.drop(['poi'], axis=1)

	# replacing 'NaN' with zeros
	df_clean = df_clean.replace('NaN', 0)
#	df_clean.apply(lambda x: x.fillna(x.mean()),axis=0)
		
#	now using SelectKbest to select the best features 
#	by using the metric f_classif
#	selection = SelectKBest(f_classif, k=5)
	selection = SelectKBest(f_classif, k=n_attr)
	selected_features = selection.fit(df_clean, df_poi)
	# the selected indeces
	indices_selected = selected_features.get_support(indices=True)
	# the columns with the selected features
	colnames_selected = [df_clean.columns[i] for i in indices_selected]

	return colnames_selected

print len(data_dict),'data_dict len'

features_list_new = feature_selection(features_list,data_dict)
#features_list_new = features_list

# the first should be 'poi'
feat_new=[]
for i in range(len(features_list_new)+1):
    if(i!=0):
        feat_new.append(features_list_new[i-1])
    else:
        feat_new.append('poi')

features_list_new=feat_new	

print features_list_new,'features_new'

#print features_list_new,'colnames selected'


### Task 2: Remove outliers



#data = featureFormat(data_dict, features_list)
data = featureFormat(data_dict, features_list_new)
labels, features = targetFeatureSplit(data)


print len(features),'bbbbbbbbbbbbbbb','linha 189'


#print 'llllllll', len(labels)

'''
features = numpy.reshape( numpy.array(features), (len(features), 1))
labels = numpy.reshape( numpy.array(labels), (len(labels), 1))
'''

# first let us define the function OutlierCleaner!


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    error=[]
#    print(predictions)
    for i in range(len(predictions)):
			error.append([i,(predictions[i]-net_worths[i])**2])
	
#    print(error)
#    print(error[1])
    ### your code goes here
    error.sort(key=lambda x: x[1],reverse=True)
#    print(error)
	
	
    order=[] 
	
    for i in range(len(error)):
		order.append(error[i][0])
		
#    ages=ages.tolist()

    listtotal=[]
#    for i in range(9):
    for i in range(16):
        listtotal.append(order[i])
#    print(len(order),'elnnnn')
#    print(listtotal,'aaa')
#    print ages,'ages'
    finalresult=[]    
#    for i in range(90):
    for i in range(len(error)):
        if(i not in listtotal):
#            print('aaquiii',ages.tolist()[i][0],error[i][0],float(net_worths[i][0]))
#            try:
#				finalresult.append([ages.tolist()[i][0],float(net_worths[i][0]),error[i][0]])
             finalresult.append([ages[i][0],float(net_worths[i][0]),error[i][0]])
#            except:
#                continue
#            try:
#                finalresult.append([ages[i][0],float(net_worths[i][0]),error[i][0]])
#            except:
#                nada =0
				
#    print(order)

    cleaned_data=finalresult
#    del ages[order]

#    order[88]
    
    return cleaned_data,listtotal




def removing_outliers(features,labels,features_list,data_dict3):
	"""
	This function remove outlier from the features!
	Inputs: -features : features of the data
			-labels : labels of the data
			-features_list: list of features to be investigated
	Outputs: 
			 -features_new = features of the data with the outliers removed
			 -labels_new = labels of the data with the outliers removed
	"""
	# reshaping features and labels
	features = numpy.reshape( numpy.array(features), (len(features),  len(features_list)-1))
	labels = numpy.reshape( numpy.array(labels), (len(labels), 1))
	# splliting into train and test
#	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=42)
	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.0, random_state=42)
	# performing the linear regression
	reglinear = linear_model.LinearRegression()
	reglinear.fit(features_train,labels_train)
	#	print('coef angular',reglinear.coef_)
	# the predicted labels
	predictions = reglinear.predict(features_train)
	#	print 'aqui'
	# calling the OutlierCleaner function to remove outliers!
	cleaned_data,listtotal = outlierCleaner( predictions, features_train, labels_train )
	#	print cleaned_data
	print 'Outlier Removal working fine!'


	# retrieving features and labels from the cleaned_data
	selected = []
	for i in range(len(cleaned_data)):
		selected.append(cleaned_data[i][2])
	features_new = []	
	labels_new = []
	for i in selected:
		features_new.append(features[i])
		labels_new.append(labels[i])

	data_dict_novo = remove_outlier_datadict(data_dict3,listtotal)
#	data_dict_novo = remove_outlier_datadict(data_dict3,selected)
		

	return features_new,labels_new, data_dict_novo


def remove_outlier_datadict(data_dict3,selected):
    """
    df_2= pd.DataFrame.from_dict(data_dict3,orient='index')
#    print df_2.index[0]
    listtotalnew=[]
    for i in listtotal:
		listtotalnew.append(df_2.index[i])
    df_2.drop(listtotalnew,axis=0,inplace=True)
    data_dict_novo = df_2.to_dict('index')
    """
    df_2= pd.DataFrame.from_dict(data_dict3,orient='index')
#    print df_2.index[0]
    listtotalnew=[]
    for i in range(len(df_2)):
		if(i not in selected):
			listtotalnew.append(df_2.index[i])
    df_2.drop(listtotalnew,axis=0,inplace=True)
    data_dict_novo = df_2.to_dict('index')
	
    return data_dict_novo	


#features,labels = removing_outliers(features,labels,features_list_new)


#print len(features_train),len(features)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
#my_dataset = cleaned_data

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_new, sort_keys = True)
labels, features = targetFeatureSplit(data)

print len(features),'bbbbbbbbbbbbbbb','linha 344'


###################################################################################################
###################################################################################################
#features,labels = removing_outliers(features,labels,features_list_new)



#print features,len(features)
#print len(data_dict['POWERS WILLIAM']),'n_o de atributos!'



# selecting features

def relative_to_bonus(features,data_dict,features_list_new):
	"""
		This function gives the quocient of the best features
		relatively to the "bonus" feature.
		Input(s): 
				 -features: features of the data
		Output(s):
				  -new_features: features of the data divided by bonus
		Remarks: We do not normalize the first column, which is "bonus".
		To avoid DivisionByZero error, we replace zeros by the mean!		
	"""
	
	
	new_features = []
	new_data_dict={}


	# converting data_dict to a dataframe
	df= pd.DataFrame.from_dict(data_dict,orient='index')

	# taking the features_list columns out of df
	df = df[features_list_new]

	# taking the 'poi' feature out of df
#	df_poi = df['poi']

	# dropping 'poi' column
	df_clean=df
#	df_clean.drop(['poi'],axis=1,inplace=True)

	# replacing 'NaN' with zeros
	df_clean = df_clean.replace('NaN', 0)
#	df_clean.apply(lambda x: x.fillna(x.mean()),axis=0)
	chosen=2
	mean_bonus = df_clean[features_list_new[chosen]].mean(skipna=True)
#	df_clean[features_list_new[1]] = df_clean.replace(0,mean_bonus)
	df_clean[features_list_new[chosen]] = df_clean[features_list_new[chosen]].replace(0,mean_bonus)
#
#	for i in range(2,5):
	for i in range(0,n_attr+1):
#		df_clean[features_list_new[i]] = df_clean[features_list_new[i]]/mean_bonus
		if(i!=0 and i!=chosen):
			df_clean[features_list_new[i]] = df_clean[features_list_new[i]].div(df_clean[features_list_new[chosen]].values,axis=0)
#			df_clean[features_list_new[i]]=df_clean[features_list_new[i]].div(df_clean[features_list_new[chosen]], axis=0)
	new_data_dict2 = df_clean.to_dict('index')


#	features
	# calculating the average of the first feature!
	avg_0=0
	for i in range(len(features)):
		avg_0+=features[i][1]
	
	# iterating through features
	for i in range(len(features)):
		feat_j=[]
#		for j in range(5):
		for j in range(n_attr):
#			if the the list element is not zero!
			if(features[i][1]!=0):
				if(j!=1 and j!=0):
						feat_j.append(features[i][j]/features[i][1])
				elif(j==0):
#					feat_j.append(features[i][j])
					pass
				else:
					feat_j.append(features[i][1])
#					pass
		

#			if there is a zero, replace it by the average!
			if(features[i][1]==0):
				if(j!=1 and j!=0):
					feat_j.append(features[i][j]/avg_0)
				elif(j==0):
#					feat_j.append(features[i][j])
					pass
				else:
					feat_j.append(avg_0)
					
		new_features.append(np.array(feat_j))
		
#	return new_features
	return new_data_dict2,new_features
#print data_dict 	


#new_data_dict=data_dict


#print new_data_dict
#print data_dict
df_v= pd.DataFrame.from_dict(data_dict,orient='index')

# taking the features_list columns out of df
df_v = df_v[features_list_new]

print df_v

"""
df_b= pd.DataFrame.from_dict(new_data_dict,orient='index')

# taking the features_list columns out of df
df_b = df_b[features_list_new]

print df_b

print len(new_data_dict),'aaaaaaaaaa'
"""



"""
data = featureFormat(new_data_dict, features_list_new, sort_keys = True)
labels, features = targetFeatureSplit(data)
"""

data = featureFormat(data_dict, features_list_new, sort_keys = True)
labels, features = targetFeatureSplit(data)

###################################################################################################
###################################################################################################

#features,labels,new_data_dict = removing_outliers(features,labels,features_list_new,new_data_dict)
"""
data = featureFormat(data_dict, features_list_new, sort_keys = True)
labels, features = targetFeatureSplit(data)
"""


print len(features),'bbbbbbbbbbbbbbb','linha 488'
print len(data),'bbbbbbbbbbbbbbb','linha 489'


features,labels,new_data_dict_o = removing_outliers(features,labels,features_list_new,data_dict)

#new_data_dict_o = data_dict

new_data_dict,features=relative_to_bonus(features,new_data_dict_o,features_list_new)


data = featureFormat(new_data_dict, features_list_new, sort_keys = True)
labels, features = targetFeatureSplit(data)


df_c= pd.DataFrame.from_dict(new_data_dict,orient='index')

# taking the features_list columns out of df
df_c = df_c[features_list_new]

print df_c


print len(new_data_dict),'bbbbbbbbbbbbbbb','linha 500'

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

###################################################################################################
###################################################################################################

# Pipeline Usage
# https://stackoverflow.com/questions/33091376/python-what-is-exactly-sklearn-pipeline-pipeline
# Good Usage: "https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html"

# Conforme Anderson Uyekita no slack!


# PCA

# what is the suitable number of components?
#pca=PCA(svd_solver='full')
pca=PCA()

###################################################################################################
###################################################################################################


# Provided to give you a starting point. Try a variety of classifiers.

# Gaussian Classifier!

gnb = GaussianNB()
#clf.fit()

###################################################################################################
###################################################################################################


# Support Vector Classifier!

svc = SVC()
#clf.fit()

# Decision Tree!

dtc = DecisionTreeClassifier(min_samples_split=5, random_state=42)


# K Nearest Neighbors!

knn = KNeighborsClassifier()

# Random Forest Classifier!

rfc = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=42)


ada=AdaBoostClassifier()

scaler = MinMaxScaler()

pipeline = Pipeline(
[('scaler', scaler),
#    ('pca', pca),
#    ('gnb', gnb),
#    ('svc', SVC()),
    ('dtc', dtc)
#    ('rfc',rfc),
#    ('knn', knn),
#     ('ada', ada),
])

#param_range=list(2*i+1 for i in range(5))
param_range=list(2*i+1 for i in range(3))

print param_range

#comp_pca = [i for i in range(1,n_attr+1)]
comp_pca = [i for i in range(1,n_attr)]

parameters = {
#	'pca__n_components': [1, 2, 3, 5],
#	'pca__n_components': comp_pca,
#    'svc__gamma' : [1,5,10],
#    'svc__kernel' : ['linear','rbf'],
#	'svc__C' : [10000]
#    ''
#	'knn__n_neighbors':[1,2,5,10,20],
#	'ada__n_estimators':[69,68,67,66,65],
#    'ada__learning_rate':[0.8,0.9, 1.0, 1.1,1.2]

	'dtc__criterion': ['gini', 'entropy'],
	'dtc__min_samples_split':[2],
	'dtc__min_samples_leaf': [1],
	'dtc__max_depth': [7,8],
#	'dtc__max_features':[3,4],
	'dtc__max_features':[3],
	'dtc__splitter': ['best','random'],
	'dtc__max_leaf_nodes':[10,11,12,13,15],

#	'rfc__criterion': ['gini', 'entropy'],
#	'rfc__min_samples_leaf': [1],
#	'rfc__max_depth': [6],
#	'rfc__min_samples_split': [2],
#	'rfc__min_samples_leaf': [1],
#	'rfc__max_features':['auto', 'sqrt', 'log2'],
#	'rfc__n_estimators':[1000],

}


parameters_rfc = {'rfc':[RandomForestClassifier()],
                #'rfc_criterion':['gini', 'entropy'],
                'rfc__min_samples_leaf': param_range,
                'rfc__max_depth': param_range,
                'rfc__min_samples_split': param_range[1:]},



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
from  sklearn.model_selection import train_test_split

#features_list_new=features_list_new[1:]

#data = featureFormat(new_data_dict, features_list_new, sort_keys = True)
#	labels, features = targetFeatureSplit(data)

#features=features[1:]

#features_old=features


#features_neu=
	
#features = scaler.fit_transform(features)

print features, 'featttt'

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.5, random_state=42)

###################################################################################################
###################################################################################################

"""
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)
"""

print features,'features',len(features)
print labels,'labels',len(labels)

sss = StratifiedShuffleSplit(n_splits=8*50,test_size = 0.5, random_state=42)

scoring = {'accuracy':make_scorer(accuracy_score),
         'precision':make_scorer(precision_score),
         'recall':make_scorer(recall_score)}


#grid_search = GridSearchCV(pipeline, parameters,n_jobs=-1, verbose=1,cv=sss,scoring=scoring,refit='recall', return_train_score=True)
grid_search = GridSearchCV(pipeline, parameters,n_jobs=-1, verbose=1,cv=sss,scoring=scoring,refit='precision', return_train_score=True)

#grid_search.fit(features_train, labels_train)
grid_search.fit(features, labels)

#predict = grid_search.predict(features_test)
predict = grid_search.predict(features)

print 'prediction', predict

#best_parameters = grid_search.best_estimator_.get_params()
best_parameters = grid_search.cv_results_

rec = recall_score(predict,labels)
prec = precision_score(predict,labels)
#rec = recall_score(predict,labels, average='micro')
#prec = precision_score(predict,labels, average='micro')

print 'recall', rec,
print 'precision', prec
#print 'best parameters',best_parameters
#print best_parameters

#clf=best_parameters
clf= grid_search.best_estimator_


#print clf,'clf'

print labels, 'labels',len(labels)

listone = []

for i in range(len(labels)):
	if (labels[i]==1):
		listone.append(i)
		
print listone,'listone'

#print features, 'featurrrrrrrrrrrrrrrrrrrre'

best_parameters = grid_search.cv_results_

df_best= pd.DataFrame.from_dict(best_parameters,orient='columns')

#print df_best.columns
#print df_best['mean_test_recall']
#print df_best['params']

print pd.DataFrame(grid_search.cv_results_)[['mean_test_recall', 'params']]

#print clf.cv_results_
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, data_dict, features_list)
dump_classifier_and_data(clf, new_data_dict, features_list_new[:])
#dump_classifier_and_data(clf, my_dataset, features_list)

###################################################################################################
###################################################################################################

