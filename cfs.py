# -*- coding: utf-8 -*-
"""cfs.py: An implementation of SFS/SBS/SFFS/SBBS algorithms. 
Created on Sun Sep 20 14:40:34 2015
"""
__author__ = "Gianluca Gerard"
__copyright__ = "Copyright 2015, Gianluca Gerard"
__credits__ = "Juha Reunanen"
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Gianluca Gerard"
__email__ = "gianluca.gerard01@universitadipavia.it"
__status__ = "Production"

import Orange

def argmin(R):
    """Return the index of the smallest number in a list R of numbers."""
    return sorted([(r,i) for i,r in enumerate(R) if r ])[0][1]

class SubsetLearner(Orange.classification.PyLearner):
    """A learning algorithm which operates on a subset of the
    domain's features. Constructor parameters set the
    corresponding attributes.
        
    .. attribute:: features
    
        The list with the subset of features.
        
    .. attribute:: base_learner
    
        The actual learner of :class:`~Orange.classification.Learner`.
        Defaults to :class:`~Orange.classification.bayes.NaiveLearner`.
        
    .. attribute:: name
    
        The name of the learner instance. Defaults to 'small'.

    """
    def __init__(self,
                 features,
                 base_learner=Orange.classification.bayes.NaiveLearner,
                 name='small'):                     
        self.name = name
        self.base_learner = base_learner
        self.features = features
        
    def __call__(self, data, weight=None):
        """Learn from the given table of data instances.
        
        :param data: Data instances to learn from.
        :type data: :class:`~Orange.data.Table`
        :param weight: Id of meta attribute with weights of instances
        :type weight: int
        :rtype: :class:`~Orange.classification.PyClassifier`
        """
            
        domain = Orange.data.Domain(self.features + [data.domain.class_var])
        
        model = self.base_learner(Orange.data.Table(domain, data), weight)
        return Orange.classification.PyClassifier(classifier=model,
                                                  name=self.name)

class SequentialSelectionLearner(Orange.classification.Learner):
    """A learning algorithm which select an optimal subset of features
    by relying on a sequential selection. The selection methods implemented
    are SFS/SBS/SFFS and SBFS.
            
    .. attribute:: type
    
        The sequential learning method to be used. One of
            :class:`~cfs.SequentialSelectionLearner.SFS`
            :class:`~cfs.SequentialSelectionLearner.SBS`
            :class:`~cfs.SequentialSelectionLearner.SFFS`
            :class:`~cfs.SequentialSelectionLearner.SBFS`

    .. attribute:: eval
    
        The evaluation function. Lower values denote better subsets.
        
    .. attribute:: features
    
        The list with all the domain's features.
        
    .. attribute:: base_learner
    
        The actual learner of :class:`~Orange.classification.Learner`.
        Defaults to :class:`~Orange.classification.bayes.NaiveLearner`.
        
    .. attribute:: name
    
        The name of the learner instance. Defaults to 'seq'.
        
    """
    SFS, SBS, SFFS, SBFS = range(4)
    
    def __init__(self, name='seq', type=SFS, features=None,
                 base_learner=Orange.classification.bayes.NaiveLearner,
                 eval=None, random_generator = Orange.misc.Random(0)):
        algorithms = ['Forward',
                      'Backward',
                      'ForwardFloating',
                      'BackwardFloating']

        self.name = name
        if type < len(algorithms):
            self.type = algorithms[type]
        else:
            raise AttributeError('type: undefined method.')
            
        self.features = features
        self.base_learner = base_learner
        if not eval:
            self.eval = lambda x: [1-a for a in
                                    Orange.evaluation.scoring.CA(x)]
        else:
            self.eval = eval
        self.random_generator = random_generator

    def __call__(self, data, weight=None):
        """Learn from the given table of data instances.
        
        :param data: Data instances to learn from.
        :type data: :class:`~Orange.data.Table`
        :param weight: Id of meta attribute with weights of instances
        :type weight: int
        :rtype: :class:`~Orange.classification.PyClassifier`
        """
        self.data = data
                 
        if not self.features:
            self.features = [f for _, f in
                            sorted((Orange.feature.scoring.InfoGain(x, data), x)
                            for x in data.domain.features)]
                        
        self.n = len(self.features)
        
        method_name = self.type
        method = getattr(self, method_name, lambda: "nothing")
        
        self.B = method()

        scores = sorted([(S[1],k) for k,S in enumerate(self.B)])
        k = scores[0][1]
                
        subset = self.B[k][0]
        
        self.features_subset = [f for i,f in enumerate(self.features)
                        if subset[i]]
                            
        return SubsetLearner(data, features=self.features_subset)         

    def J(self, S):  
        """Runs the evaluation function `eval` on the results of
        the leave-one out testing of the chosen learner, `base_learner`,
        with the features restricted to the `features_subset`.
        
        :param S: list of :obj:`~Orange.feature.Descriptor`.
        """

        if S == None:
            return None
        else:
            features_subset = [f for i,f in enumerate(self.features) if S[i]]
            res = Orange.evaluation.testing.leave_one_out(
                            [SubsetLearner(
                            base_learner=self.base_learner,
                            features=features_subset)], self.data)            
            return self.eval(res)[0]
        
    def Forward(self):
        """Implementation of the Sequential Forward Selection algorithm."""

        S = [False]*self.n
        B = [(None,0)]*self.n
        for k in range(self.n):
            R = [None]*self.n
            for j,f in enumerate(S):
                if not f:
                    S[j] = True
                    R[j] = self.J(S)
                    S[j] = False
            j = argmin(R)
            S[j] = True
            B[k] = (list(S),R[j])
#            print "SFS",k+1,j,B[k]
        return B
        
    def Backward(self):
        """Implementation of the Sequential Backward Selection algorithm."""

        S = [True]*self.n
        B = [(None,0)]*self.n
        B[self.n-1] = (list(S),self.J(S))
        for k in range(self.n-1,0,-1):
            R = [None]*self.n
            for j,f in enumerate(S):
                if f:
                    S[j] = False
                    R[j] = self.J(S)
                    S[j] = True
            j = argmin(R)
            S[j] = False
            B[k-1] = (list(S), R[j])
#            print "SBS",k,j,B[k-1]
        return B
        
    def ForwardFloating(self):
        """Implementation of the Sequential Forward Floating Selection
        algorithm.
        """

        S = [False]*self.n
        B = [(None,0)]*self.n
        # SFS
        k = 0
        while k < self.n:
            R = [None]*self.n
            for j,f in enumerate(S):   # repeat for each possible branch
                if not f:
                    S[j] = True
                    R[j] = self.J(S)
                    S[j] = False
            j = argmin(R)
            if (False if not B[k][0] else R[j] >= B[k][1]):
                # avoid following a less promising path
                # if a better one of size k was already
                # found
                S = list(B[k][0])
            else:
                S[j] = True
                B[k] = (list(S),R[j])
                
                # SBS
                while k > 1:
                    R = [None]*self.n
                    for j,f in enumerate(S):
                        if f:
                            S[j] = False
                            R[j] = self.J(S)
                            S[j] = True
                    j = argmin(R)
                    if R[j] < B[k-1][1]:
                        k = k - 1
                        S[j] = False
                        B[k] = (list(S),R[j])
                    else:
                        break
            k = k + 1
        return B

    def BackwardFloating(self):
        """Implementation of the Sequential Backward Floating Selection
        algorithm.
        """

        S = [True]*self.n
        B = [(None,0)]*self.n
        B[self.n-1] = (list(S),self.J(S))
        # SBS
        k = self.n-2
        while k >= 0:
            R=[None]*self.n
            for j,f in enumerate(S):
                if f:
                    S[j] = False
                    R[j] = self.J(S)
                    S[j] = True
            j = argmin(R)
            if (False if not B[k][0] else R[j] >= B[k][1]):
                # avoid following a less promising path
                # if a better one of size k was already
                # found
                S = list(B[k][0])
            else:
                S[j] = False
                B[k]= (list(S), R[j])
                # SFS
                while k < self.n-2:
                    R = [None]*self.n
                    for j,f in enumerate(S):
                        if not f:
                            S[j] = True
                            R[j] = self.J(S)
                            S[j] = False
                    j = argmin(R)
                    if R[j] < B[k+1][1]:
                        k = k + 1
                        S[j] = True
                        B[k] = (list(S), R[j])
                    else:
                        break
            k = k - 1
        return B
