import numpy as np
from sklearn import metrics

def compute_purity(label_pred,label_true):
    """ This function is used to compute the purity of a given clustering
    solution w.r.t a second clustering solution, which could be but does not
    necessarily have to be the ground truth

    Parameters
    ----------
    label_pred: array, len(n_instances)
        label of the first clustering solution

    label_true: array, len(n_instances)
        label of the second clustering solution

    Returns
    -------
    score_purity: float
        purity score, between 0 and 1
    """

    # get number of samples
    n_instances = len(label_pred)

    # unique labels
    label_pred_unique = np.unique(label_pred)
    label_true_unique = np.unique(label_true)

    # store sample indices correponding to each unique label
    clust_pred = []
    for i in label_pred_unique:
        clust_pred.append(np.where(label_pred == i)[0])
    clust_true = []
    for i in label_true_unique:
        clust_true.append(np.where(label_true == i)[0])
    
    # compute purity score
    score_purity = 0.
    max_purity = 0.
    tmp_2 = []
    for k in range(len(label_pred_unique)):
        tmp_1 = []
        for j in range(len(label_true_unique)):
            tmp_and = set(clust_pred[k]).intersection(set(clust_true[j]))
            tmp_or = set(clust_pred[k]).union(set(clust_true[j]))
            tmp_1.append(len(tmp_and))
            tmp_2.append(len(tmp_and)*1./len(tmp_or))
        score_purity += max(tmp_1)*1./n_instances
    max_purity = max(tmp_2)
    return (score_purity,max_purity)


if __name__ == "__main__":
    label_pred = np.array([1,1,1,1,1,1,\
            2,2,2,2,2,2,\
            3,3,3,3,3])
    label_true = np.array([1,1,1,1,1,2,\
            1,2,2,2,2,3,\
            1,1,3,3,3])

    print "purity: ",compute_purity(label_pred,label_true)
    print "NMI: ",metrics.normalized_mutual_info_score(label_pred,label_true)
    print "RandIndex: ",metrics.adjusted_rand_score(label_pred,label_true)
    print "ConfusionMatrix: ",metrics.confusion_matrix(label_pred,label_true)

