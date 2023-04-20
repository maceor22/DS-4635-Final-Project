import pandas as pd
import numpy as np
from numpy import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

random.seed(0)


def generate_PC_plots(pca, input_data, label_data, title1, title2):
    
    num_comps = np.arange(1, pca.n_components_+1)
    
    plt.figure()
    plt.plot(num_comps, np.cumsum(pca.explained_variance_ratio_), 'k')
    plt.xlabel('Number of Principle Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title(title1)
    
    neg_idxs = label_data.nonzero()[0]
    neg_input = input_data[neg_idxs,:]
    pos_input = np.delete(input_data, neg_idxs, axis = 0)
    
    plt.figure()
    plt.plot(neg_input[:,0], neg_input[:,1], 'o', color = 'blue', label = 'readmitted')
    plt.plot(pos_input[:,0], pos_input[:,1], 'o', color = 'orange', label = 'not readmitted')
    plt.legend()
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title2)


def normalize_numeric(raw_train, raw_test):
    
    train_numeric = raw_train[:,:8]
    test_numeric = raw_test[:,:8]
    
    norm = StandardScaler()
    
    scaled_train_numeric = norm.fit_transform(train_numeric)
    scaled_test_numeric = norm.transform(test_numeric)
    
    raw_train[:,:8] = scaled_train_numeric
    raw_test[:,:8] = scaled_test_numeric
    
    return raw_train, raw_test
    


def create_dataset1(raw_train, raw_test):
    
    train_data, test_data = normalize_numeric(raw_train, raw_test)    
    
    return train_data, test_data



def create_dataset2(raw_train, raw_test, n_components = None, plot_principal_components = False):
    
    norm_train, norm_test = normalize_numeric(raw_train, raw_test)
    
    norm_train_input = norm_train[:,:-1]
    train_label = norm_train[:,-1]

    norm_test_input = norm_test[:,:-1]
    test_label = norm_test[:,-1]
    
    pca = PCA(n_components = n_components)
    
    train_input = pca.fit_transform(norm_train_input)
    test_input = pca.transform(norm_test_input)
    
    if plot_principal_components:
        generate_PC_plots(
            pca, train_input, train_label, 
            title1 = 'Principal Components of All Data Channels', 
            title2 = 'Linear Separation Exploration | All Data Channels',
            )
        
    train_data = np.concatenate([train_input, train_label.reshape(-1,1)], axis = 1)
    test_data = np.concatenate([test_input, test_label.reshape(-1,1)], axis = 1)
    
    return train_data, test_data
    


def create_dataset3(raw_train, raw_test, n_components = None, plot_principal_components = False):
    
    norm_train, norm_test = normalize_numeric(raw_train, raw_test)
    
    bool_train_input = norm_train[:,8:-1]
    train_label = norm_train[:,-1]

    bool_test_input = norm_test[:,8:-1]
    test_label = norm_test[:,-1]
    
    pca = PCA(n_components = n_components)
    
    bool_train_input = pca.fit_transform(bool_train_input)
    bool_test_input = pca.transform(bool_test_input)
    
    if plot_principal_components:
        generate_PC_plots(
            pca, bool_train_input, train_label, 
            title1 = 'Principal Components of Boolean Data Channels', 
            title2 = 'Linear Separation Exploration | Boolean Data Channels',
            )
        
    train_data = np.concatenate(
        [norm_train[:,:8], bool_train_input, train_label.reshape(-1,1)], axis = 1)
    test_data = np.concatenate(
        [norm_test[:,:8], bool_test_input, test_label.reshape(-1,1)], axis = 1)
    
    return train_data, test_data




if __name__ == '__main__':
    
    raw = pd.read_csv('raw_data.csv')
    
    idxs = np.arange(len(raw))
    random.shuffle(idxs)
    bound = int(.85*len(raw))
    train_idxs = idxs[:bound]
    test_idxs = idxs[bound:]
    
    raw_train = raw.values[train_idxs,:].astype(np.float64)
    raw_test = raw.values[test_idxs,:].astype(np.float64)
    
    
    train1, test1 = create_dataset1(raw_train, raw_test)
    
    train2, test2 = create_dataset2(raw_train, raw_test, n_components = 30, plot_principal_components = True)
    
    train3, test3 = create_dataset3(raw_train, raw_test, n_components = 30, plot_principal_components = True)
    
    
# =============================================================================
#     np.save('Datasets/raw data/train.npy', train1)
#     np.save('Datasets/raw data/test.npy', test1)
#     
#     np.save('Datasets/entire pca/train.npy', train2)
#     np.save('Datasets/entire pca/test.npy', test2)
# 
#     np.save('Datasets/numeric and bool pca/train.npy', train3)
#     np.save('Datasets/numeric and bool pca/test.npy', test3)
# =============================================================================

















