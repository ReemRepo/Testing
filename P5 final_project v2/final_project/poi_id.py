
# coding: utf-8

# In[1]:

# %load poi_id.py
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
features_list1 = ['poi',
'salary',
 'exercised_stock_options',
 'bonus',
 'restricted_stock_deferred',
 'to_poi_ratio' ,
 'from_poi_ratio' , 
 'bonus_to_salary_ratio']  

features_list =[ 'poi',
 'salary',
 'to_messages',
 'deferral_payments',
 'total_payments',
 'exercised_stock_options',
 'bonus',
 'restricted_stock',
 'shared_receipt_with_poi',
 'restricted_stock_deferred',
 'total_stock_value',
 'expenses',
 'loan_advances',
 'from_messages',
 'director_fees',
 'deferred_income',
 'long_term_incentive',
 'to_poi_ratio' ,
 'from_poi_ratio' , 
 'bonus_to_salary_ratio']  

selected_features_list = ['poi',
 'salary',
 'total_payments',
 'bonus',
 'bonus_to_salary_ratio',
 'total_stock_value',
 'exercised_stock_options',
 'to_poi_ratio',
 'deferred_income',
 'restricted_stock',
 'long_term_incentive']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



# In[2]:

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0 ) #spreadsheet total raw
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0) #not a person
data_dict.pop('LOCKHART EUGENE E', 0) #contains NaNs for all features

len(data_dict) #143 persons




# In[3]:

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#1st and 2nd features inspired from lesson's Quiz: Feature Selection
#3rd feature from searching about the Enron Fraud case

# 1st new feature: to_poi_ratio
for key, feature in my_dataset.iteritems():
    if feature['from_this_person_to_poi'] == "NaN" or feature['from_messages'] == "NaN" or feature['from_this_person_to_poi']== 0:
        feature['to_poi_ratio'] = 0
    else:
        feature['to_poi_ratio'] = float(feature['from_this_person_to_poi']) / float(feature['from_messages'])

# 2nd new feature: from_poi_ratio
for key, feature in my_dataset.iteritems():
    if feature['from_poi_to_this_person'] == "NaN" or feature['to_messages'] == "NaN" or feature['from_poi_to_this_person'] == 0:
        feature['from_poi_ratio'] = 0
    else:
        feature['from_poi_ratio'] = float(feature['from_poi_to_this_person']) / float(feature['to_messages'])

# 3rd new feature: bonus_to_salary_ratio
for key, feature in my_dataset.iteritems():
    if feature['bonus'] == "NaN" or feature['salary'] == "NaN":
        feature['bonus_to_salary_ratio'] = 0
    else:
        feature['bonus_to_salary_ratio'] = float(feature['bonus']) / float(feature['salary'])

### Store to my_dataset for easy export below.
my_dataset = data_dict


print '# of features (with new ones):' , len(my_dataset['METTS MARK'])

print 'POI:'
# decide to keep or remove new feature by check their value with POI
for key, feature in my_dataset.iteritems():
    if feature['poi']:
        print key, feature['to_poi_ratio'] , feature['from_poi_ratio'] , feature['bonus_to_salary_ratio']



print 'Non-POI'
# decide to keep or remove new feature by check their value with POI
for key, feature in my_dataset.iteritems():
    if not(feature['poi']):
        print key, feature['to_poi_ratio'] , feature['from_poi_ratio'] , feature['bonus_to_salary_ratio']
    
        
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list,sort_keys=True)
labels, features = targetFeatureSplit(data)


# In[4]:

from sklearn.feature_selection import SelectKBest


#Select K-Best features
k = 5
k_best = SelectKBest(k=k)
k_best.fit(features, labels)
scores = k_best.scores_
unsorted_features = zip(features_list[1:], scores)
sorted_features = list(reversed(sorted(unsorted_features, key=lambda x: x[1])))
k_best_features = dict(sorted_features[:k])



selected_features_list = ['poi'] + k_best_features.keys()


for key, value in sorted_features:
    print key , '=' , round(value,2)

print selected_features_list
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, selected_features_list)
labels, features = targetFeatureSplit(data)


# In[5]:

#Scale features using MinMaxScaler
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)


# In[10]:

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)
    
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, grid_search
from sklearn.neighbors import KNeighborsClassifier



def test_clf(c):
    clf = c
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    accuracy = accuracy_score(labels_test, predictions)
    precision = precision_score(labels_test, predictions)
    recall = recall_score(labels_test, predictions)
    F1 = f1_score(labels_test, predictions)
    
    #precision =  precision_score(features_test,labels_test,average='weighted')
    #recall =recall_score(features_test,labels_test,average='weighted')
    #F1 = f1_score(features_test,labels_test,average='weighted')

    print c
    print 'accuracy =' , round(accuracy ,2)
    print 'precision =', round(precision ,2)
    print 'recall =', round(recall ,2)
    print 'F1 =', round(F1 ,2)
    
clf = GaussianNB()
test_clf(GaussianNB())
print''
test_clf(tree.DecisionTreeClassifier(min_samples_split=40))
print''
test_clf(tree.DecisionTreeClassifier(min_samples_split=60))
print''
test_clf(tree.DecisionTreeClassifier(min_samples_split=100))
print''
test_clf(KMeans(n_clusters=2, random_state=0))
print''
test_clf(KMeans(n_clusters=2, tol=0.001))


# In[ ]:

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

# using Pipline and GridSearch for tuning classifiers

print''
test_clf(SVC(kernel="rbf", C=10000.0))
lclf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', random_state = 42))])


lclf2 = Pipeline(steps=[
        ('scaler', preprocessing.MinMaxScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', random_state = 42))])
print''
test_clf(lclf2)

print''
test_clf(LogisticRegression(C=1e5))

print''
test_clf(lclf)
clf = rfclf = RandomForestClassifier(max_depth = 5,max_features = 'sqrt',n_estimators = 10, random_state = 42)
print''
test_clf(rfclf)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10,10000.0]}
svr = SVC()
svrclf = grid_search.GridSearchCV(svr, parameters) #too long in tester

#print''
#test_clf(svrclf)

DT = tree.DecisionTreeClassifier()
parameters = {'min_samples_split':[40,60, 100]}
DTclf = grid_search.GridSearchCV(DT, parameters)
print''
test_clf(DTclf)

print''
test_clf(KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=5, p=2, weights='distance'))

KNeighbors = KNeighborsClassifier()
parameters = {'leaf_size':[20,30,50], 'n_neighbors':[3,5, 10], 'weights':['uniform', 'distance']}
KNclf = grid_search.GridSearchCV(KNeighbors, parameters)

print''
test_clf(KNclf)



# In[23]:


clf1 = GaussianNB()
clf2= tree.DecisionTreeClassifier(min_samples_split=40)
clf3 = tree.DecisionTreeClassifier(min_samples_split=60)
clf4 = tree.DecisionTreeClassifier(min_samples_split=100)
clf5 = KMeans(n_clusters=2, random_state=0)
clf6 = KMeans(n_clusters=2, tol=0.001)
clf7 = SVC(kernel="rbf", C=10000.0)
clf8 = lclf
clf9 = lclf2
clf10 = LogisticRegression(C=1e5)
clf11 = rfclf
#clf12 = svrclf
clf13 = DTclf
clf14 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=5, p=2, weights='distance')
clf15 = KNclf


#select best classifier (KNeighborsClassifier)

clf=clf14


# In[24]:

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, selected_features_list)


# In[13]:

# %load tester.py
#!/usr/bin/pickle

""" a basic script for importing student's POI identifier,
    and checking the results that they get from it 
 
    requires that the algorithm, dataset, and features list
    be written to my_classifier.pkl, my_dataset.pkl, and
    my_feature_list.pkl, respectively

    that process should happen at the end of poi_id.py
"""

import pickle
import sys
from sklearn.cross_validation import StratifiedShuffleSplit
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

PERF_FORMAT_STRING = "\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\tRecall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

def dump_classifier_and_data(clf, dataset, feature_list):
    with open(CLF_PICKLE_FILENAME, "w") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "w") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "w") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)

def load_classifier_and_data():
    with open(CLF_PICKLE_FILENAME, "r") as clf_infile:
        clf = pickle.load(clf_infile)
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    with open(FEATURE_LIST_FILENAME, "r") as featurelist_infile:
        feature_list = pickle.load(featurelist_infile)
    return clf, dataset, feature_list

def main():
    ### load up student's classifier, dataset, and feature_list
    clf, dataset, feature_list = load_classifier_and_data()
    ### Run testing script
  
    test_classifier(clf1, dataset, feature_list)
    #test_classifier(clf2, dataset, feature_list)
    #test_classifier(clf3, dataset, feature_list)
    #test_classifier(clf4, dataset, feature_list)
    test_classifier(clf5, dataset, feature_list)
    test_classifier(clf6, dataset, feature_list)
    test_classifier(clf7, dataset, feature_list)
    test_classifier(clf8, dataset, feature_list)
    test_classifier(clf9, dataset, feature_list)
    test_classifier(clf10, dataset, feature_list)
    test_classifier(clf11, dataset, feature_list)
    #test_classifier(clf12, dataset, feature_list)
    test_classifier(clf13, dataset, feature_list)
    test_classifier(clf14, dataset, feature_list)
    #test_classifier(clf15, dataset, feature_list)




if __name__ == '__main__':
    main()


# In[27]:


KNclf.best_params_
DTclf.best_params_ 

