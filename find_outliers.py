
import numpy as np
import scipy
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

from features import transform_data

def find_outliers(list_data):
    data = np.array(list_data)
    trans = transform_data(data)
    
    # Remove features with zero variance
    sds = np.std(trans, axis=0)
    feature_indices = []
    for i, sd in enumerate(sds):
        if sd != 0:
            feature_indices.append(i)
    trans = trans[:,feature_indices]
    cov = np.cov(trans.T)
    # PLU matrix factorization
    P, L, U = scipy.linalg.lu(cov)

    # Boolean array for multicolinear features
    redundant_features = np.zeros(len(feature_indices))
    red_feats = False
    for i in range(np.shape(U)[0]):
        if U[i,i] == 0 or L[i,i] == 0:
            redundant_features[i] = 1
            red_feats = True
    if red_feats:
        # We have to do this to allow for the pivot in PLU
        redundant_features = np.matmul(P,redundant_features.T)
        feature_indices = [ i for i, redundant_bool in enumerate(redundant_features) if not redundant_bool ]
        trans = trans[:, feature_indices]
        cov = np.cov(trans.T)
        P, L, U = scipy.linalg.lu(cov)
        
    avs = np.average(trans, axis=0)

    # This hard coded value seems to work better than a percentage point of Chi2
    # Presumably because features are not normally distributed
    # Be cafeful - this might depend on the number of features*
    crit = 250

    outliers = []

    for text, datum in zip(list_data, trans):
        # Repeatedly solve systems of equations rather than inverting P, L, U
        a = np.linalg.solve(P, datum-avs)
        b = np.linalg.solve(L, a)
        y = np.linalg.solve(U, b)
        maha = np.dot(datum-avs, y)
        if maha > crit:
            outliers += [text]
            print("Datum:", text, ", MD:", maha)
    return outliers


