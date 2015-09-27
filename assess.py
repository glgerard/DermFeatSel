# -*- coding: utf-8 -*-
"""assess.py: Evaluate the performance with and without feature selection of
various classifiers.

Created on Sun Sep 20 14:40:34 2015
"""
__author__ = "Gianluca Gerard"
__copyright__ = "Copyright 2015, Gianluca Gerard"
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Gianluca Gerard"
__email__ = "gianluca.gerard01@universitadipavia.it"
__status__ = "Production"

import scipy as sp
import Orange
import cfs
        
rg = Orange.misc.Random(42)

# t coefficient to compute CI at 95% of a 10 fold cross validation
alpha = 0.05
cv_t = sp.stats.t.ppf(1-alpha/2, 9)

# Function used for univariate ranking of the features
score = Orange.feature.scoring.InfoGain()

# Load the dataset
derm = Orange.data.Table("dermatology.csv")

# Remove the "Age" feature as it contains missing values
domain_noAge = Orange.data.Domain(derm.domain.features[:-1] + 
                                [derm.domain.class_var])
derm_noAge = Orange.data.Table(domain_noAge, derm)

subset_idx = [31, 7, 26, 24, 14, 5, 15, 28, 9, 6, 29, 27, 33, 22, 20, 21]
subset = [domain_noAge.features[i-1] for i in subset_idx ]
ig_subset = [f for _, f in
            sorted((score(x, derm_noAge), x) for x in
            domain_noAge.features)][-int(len(domain_noAge.features)*2/3)-1:]

# Information Gain filtering of attributes

# Setup all classifiers

nbc = Orange.classification.bayes.NaiveLearner(name="nbc")
knn = Orange.classification.knn.kNNLearner(name="knn")
svm = Orange.classification.svm.SVMLearnerEasy(name="svm")                    

type = 'SFFS'

# Compare the performances of various classifiers with and without
# feature selection    

nbc_learner = cfs.SubsetLearner(features=subset,
                        name="nbc_"+type,
                        base_learner=Orange.classification.bayes.NaiveLearner)
                            
knn_learner = cfs.SubsetLearner(features=subset,
                        name="knn_"+type,
                        base_learner=Orange.classification.knn.kNNLearner)
lsvm_learner = cfs.SubsetLearner(features=subset,
                        name="lsvm_"+type,
                        base_learner=Orange.classification.svm.LinearSVMLearner)                        
svm_learner = cfs.SubsetLearner(features=subset,
                        name="svm_"+type,
                        base_learner=Orange.classification.svm.SVMLearnerEasy)
ig_svm_learner = cfs.SubsetLearner(features=ig_subset,
                        name="ig_svm_"+type,
                        base_learner=Orange.classification.svm.LinearSVMLearner)

learners = [  svm, lsvm_learner, ig_svm_learner ]
res = Orange.evaluation.testing.cross_validation(learners, derm_noAge,
                                                 folds=10, random_generator=rg)
print "Accuracy as evaluated in 10-fold cross validation of the training set."
print ", ".join("%s: (%.3f, %.3f)" % (l.name, s[0], s[1]) for l, s in zip(learners,
                    Orange.evaluation.scoring.CA(res, report_se=True)))
  
# For a t-Student comparison of the classifiers output the accuracy of
# each iteration.
for k,res_k in enumerate(Orange.evaluation.scoring.split_by_classifiers(res)):
    print
    print "Classifier:", learners[k].name
    print "Iteration, Accuracy"
    for i,res_i in enumerate(Orange.evaluation.scoring.split_by_iterations(res_k)):
        print i, Orange.evaluation.scoring.CA(res_i)[0]