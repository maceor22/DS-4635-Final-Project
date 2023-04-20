import numpy as np
from numpy import random
from sklearn.metrics import accuracy_score, roc_auc_score

# import model to be evaluated
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# seed to random generator; yield consistent results
random.seed(0)


# simple function for loading train/test datasets
def load_data(root):
    
    train = np.load(root+'/train.npy')
    train_input, train_label = train[:,:-1], train[:,-1]
    
    test = np.load(root+'/test.npy')
    test_input, test_label = test[:,:-1], test[:,-1]
    
    return train_input, train_label, test_input, test_label


# Notes:
#   Minimal code adaptations are required due to how similarly sklearn implements its models
#   I left notes below on what needs to be changed to evaluate each separate model
#   You may need to manually create folders to store results
#   This whole program took less than 2 minutes to run locally on my machine
#   However, I suspect logistic regression will take longer as it uses gradient descent
#   If this program takes ~15 minutes to run don't freak out
#   Especially if you're using cloud resources (jupyter notebooks) then it may take longer, not sure
#   Text/call me with any questions 


if __name__ == '__main__':
    
    # number of components to be evaluated for datasets 2 & 3, don't change this
    n_comps = np.arange(10, 31)
    
    # hyperparameters to be evaluated.
    # change this to sweep the values of hyperparameters you want to evaluate
    hparams = np.arange(0.1, 1.1, 0.1)
    
    # load dataset1 (train and test)
    train_input1, train_label1, test_input1, test_label1 = load_data(root = 'Datasets/raw data')
    
    # arrays to store performance metrics for later plotting 
    train_metrics1 = np.zeros((2,len(hparams)))
    test_metrics1 = np.zeros((2,len(hparams)))
    
    # iterating over hyperparameters
    for j in range(len(hparams)):
        # only need to change this line, everything else should still work
        # ex:
        #   classifier = LogisticRegression(...)
        classifier = LDA(solver = 'lsqr', shrinkage = hparams[j])
        classifier.fit(train_input1, train_label1)
        
        train_pred = classifier.predict(train_input1)
        test_pred = classifier.predict(test_input1)
        
        train_metrics1[0,j] = accuracy_score(train_label1, train_pred)
        test_metrics1[0,j] = accuracy_score(test_label1, test_pred)
        
        train_pred_proba = classifier.predict_proba(train_input1)[:,1]
        test_pred_proba = classifier.predict_proba(test_input1)[:,1]

        train_metrics1[1,j] = roc_auc_score(train_label1, train_pred_proba)
        test_metrics1[1,j] = roc_auc_score(test_label1, test_pred_proba)
        
    # change directories below to correct path
    np.save('LDA results/raw data/train_metrics.npy', train_metrics1)
    np.save('LDA results/raw data/test_metrics.npy', test_metrics1)
    
    
    
    # load dataset2 (train and test)
    train_input2, train_label2, test_input2, test_label2 = load_data(root = 'Datasets/entire pca')
    
    # performance metric storage arrays for dataset2
    train_metrics2 = np.zeros((2,len(n_comps),len(hparams)))
    test_metrics2 = np.zeros((2,len(n_comps),len(hparams)))
    
    # iterating over number of principal components to be included
    for i in range(len(n_comps)):
        tr_input = train_input2[:,:n_comps[i]]
        te_input = test_input2[:,:n_comps[i]]
        
        # iterating over hyperparameters to be evaluated
        for j in range(len(hparams)):
            # same as above, only need to edit this line
            classifier = LDA(solver = 'lsqr', shrinkage = hparams[j])
            classifier.fit(tr_input, train_label2)
            
            train_pred = classifier.predict(tr_input)
            test_pred = classifier.predict(te_input)
            
            train_metrics2[0,i,j] = accuracy_score(train_label2, train_pred)
            test_metrics2[0,i,j] = accuracy_score(test_label2, test_pred)
            
            train_pred_proba = classifier.predict_proba(tr_input)[:,1]
            test_pred_proba = classifier.predict_proba(te_input)[:,1]
            
            train_metrics2[1,i,j] = roc_auc_score(train_label2, train_pred_proba)
            test_metrics2[1,i,j] = roc_auc_score(test_label2, test_pred_proba)
    
    # change directories below to correct path
    np.save('LDA results/entire pca/train_metrics.npy', train_metrics2)
    np.save('LDA results/entire pca/test_metrics.npy', test_metrics2)



    # load dataset3 (train and test)
    train_input3, train_label3, test_input3, test_label3 = load_data(root = 'Datasets/numeric and bool pca')
    
    # performance metric storage arrays for dataset3
    train_metrics3 = np.zeros((2,len(n_comps),len(hparams)))
    test_metrics3 = np.zeros((2,len(n_comps),len(hparams)))
    
    # iterating over the number of principal components to be included
    for i in range(len(n_comps)):
        # include the first 8 columns (numeric data) in addition to specified number of PCs
        tr_input = train_input3[:,:n_comps[i]+8]
        te_input = test_input3[:,:n_comps[i]+8]
        
        # iterating over hyperparameter values to be evaluated
        for j in range(len(hparams)):
            # same as above, only need to edit this line
            classifier = LDA(solver = 'lsqr', shrinkage = hparams[j])
            classifier.fit(tr_input, train_label3)
            
            train_pred = classifier.predict(tr_input)
            test_pred = classifier.predict(te_input)
            
            train_metrics3[0,i,j] = accuracy_score(train_label3, train_pred)
            test_metrics3[0,i,j] = accuracy_score(test_label3, test_pred)
            
            train_pred_proba = classifier.predict_proba(tr_input)[:,1]
            test_pred_proba = classifier.predict_proba(te_input)[:,1]
            
            train_metrics3[1,i,j] = roc_auc_score(train_label3, train_pred_proba)
            test_metrics3[1,i,j] = roc_auc_score(test_label3, test_pred_proba)
    
    # change directories below to correct path
    np.save('LDA results/numeric and bool pca/train_metrics.npy', train_metrics3)
    np.save('LDA results/numeric and bool pca/test_metrics.npy', test_metrics3)

























