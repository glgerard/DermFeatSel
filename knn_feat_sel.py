# -*- coding: utf-8 -*-
"""knn_feat_sel.py: Searches the best subset of features in the dermatology DB.

It uses a Sequential Forward Floating Selection algorithm.
The evaluation function is the multi-class Brier score obtained by a
k-Nearest Neighbor classifier with K=10 and a leave-one out validation.

The process iterates over different training/testing split of the data:
50-50%, 60-40%, 70-30%, 80-20% and 90-10%.

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

# Basic classifiers used as comparison
nbc = Orange.classification.bayes.NaiveLearner(name="nbc")
knn = Orange.classification.knn.kNNLearner(k=10,name="knn")
svm = Orange.classification.svm.SVMLearnerEasy(name="svm")

# Load the dataset
derm = Orange.data.Table("dermatology.csv")

# Associate a progressive identifier to each feature
features_dict = {f.name: str(i+1) for i,f in
                    enumerate(derm.domain.features)}
                    
# Remove the "Age" feature as it contains missing values
new_domain = Orange.data.Domain(derm.domain.features[:-1] + 
                                [derm.domain.class_var])
new_derm = Orange.data.Table(new_domain, derm)

type = 'SFFS'

print "==== %s / kNN / Brier =====" % type
print

# Iterate over the various training-testing splits
for p in [0.5, 0.6, 0.7, 0.8, 0.9]:
    
    print "Training proportion of the dataset %d %%" % (p*100)
    
# Split the dataset in training and testing
    indices2 = Orange.data.sample.SubsetIndices2(p0=p)
    indices2.random_generator = rg
    ri2 = indices2(new_derm)
    train_all = new_derm.select(ri2, 0)
    test_all = new_derm.select(ri2, 1)

# Rank the features with the method defined by `score()`.
# Only the top 2 thirds are retained.
    top_features = [f for _, f in
            sorted((score(x, train_all), x) for x in
            new_domain.features)][-int(len(new_domain.features)*2/3)-1:]
            
# Restrict the features of both the training and testing set
# to the top 2 third of the features as ranked before
    domain = Orange.data.Domain(top_features + [new_domain.class_var])
    train = Orange.data.Table(domain, train_all)
    test = Orange.data.Table(domain, test_all)

    print "Features:", len(train.domain.features)
    print "Best ones:", ", ".join([features_dict[x.name] for x in top_features])

# Start the features selection
    seq_learner = cfs.SequentialSelectionLearner(
                                type=cfs.SequentialSelectionLearner.SFFS,
                                features=top_features,
                                base_learner=Orange.classification.knn.kNNLearner(k=10),
                                eval=Orange.evaluation.scoring.Brier_score,
                                random_generator=rg)
        
    seq_learner(train)

# Output the selected features
    best = [features_dict[f.name] for f in seq_learner.features_subset]
    
    print type, "best (", len(best), ")", ", ".join(best)
    print
    
# Compare the performances of various classifiers with and without
# feature selection    
    bayes_learner = cfs.SubsetLearner(features=seq_learner.features_subset,
                            name="bayes_"+type,
                            base_learner=Orange.classification.bayes.NaiveLearner)
    knn_learner = cfs.SubsetLearner(features=seq_learner.features_subset,
                            name="knn_"+type,
                            base_learner=Orange.classification.knn.kNNLearner(k=10))
    lsvm_learner = cfs.SubsetLearner(features=seq_learner.features_subset,
                            name="lsvm_"+type,
                            base_learner=Orange.classification.svm.LinearSVMLearner)                        
    svm_learner = cfs.SubsetLearner(features=seq_learner.features_subset,
                            name="svm_"+type,
                            base_learner=Orange.classification.svm.SVMLearnerEasy)

    learners = [ nbc, bayes_learner, knn, knn_learner,
                svm, lsvm_learner, svm_learner ]
    res = Orange.evaluation.testing.cross_validation(learners, train, folds=10,
                                                     random_generator=rg)
    print "Accuracy as evaluated in 10-fold cross validation of the training set."
    print ", ".join("%s: (%.3f, %.3f)" % (l.name, s[0], cv_t*s[1]) for l, s in zip(learners,
                    Orange.evaluation.scoring.CA(res, report_se=True)))
    print
    
    res = Orange.evaluation.testing.learn_and_test_on_test_data(learners, train, test)
    print "Accuracy as evaluated on the testing set."
    print ", ".join("%s: (%.3f)" % (l.name, s) for l, s in zip(learners,
                    Orange.evaluation.scoring.CA(res))) 
    print
    print
