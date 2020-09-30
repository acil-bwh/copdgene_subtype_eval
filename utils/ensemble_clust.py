import numpy as np

def construct_hypergraph(solution,flag_union):
    """ Given a clustering solution in the form of case_id indexed dictionary,
    construct h matrix in the form of 0-1 coding
    Parameters
    ----------
    solution: dict
        each solution is indexed by solution name, the corresponding value is
        one clustering solution, where case ids are used as keys and cluster
        assignments are used as values
    
    flag_union: boolean
        whether to take the union of case ids in all input clustering solutions
        flag_union = True means taking the union of case ids
        flag_union = False means taking the intersection of case ids

    Returns
    -------
    H: array, shape(n_instance,n_clusters)
        hypergraph representation

    case_id_unique: list, len(n_instances_union)
        unique case ids after combining all solutions
    """

    # number of solutions
    n_solution = len(solution)
    
    # union
    if flag_union == True:
        # number of samples
        tmp = []
        for k in solution.keys():
            tmp = tmp + solution[k].keys()
        # find the unique set of ids as the first dimenion of hypergraph
        case_id_unique = list(np.unique(tmp))
        n_instance = len(case_id_unique)
    
    # intersection
    else:
        assert n_solution > 1, "there's only one input solution"
        solution_id = solution.keys()
        tmp = solution[solution_id[0]].keys()
        for k in range(1,n_solution):
            tmp = list(set(tmp).intersection(set(solution[solution_id[k]].keys())))
        case_id_unique = tmp
        n_instance = len(case_id_unique)

    # construct h for each solution
    h_list = []
    for k in solution.keys():
        # find all possible cluster_ids among case_id_unique in solution[k]
        tmp_clust_id = []
        for key in solution[k].keys():
            if key in case_id_unique:
                tmp_clust_id.append(solution[k][key])
        # find cluster ids
        clust_id = list(np.unique(tmp_clust_id))

        if 'NA' in clust_id:
            clust_id.remove('NA')
        h = np.zeros((n_instance,len(clust_id)))
        for i in range(len(case_id_unique)):
            if case_id_unique[i] in solution[k].keys():
                for j in range(len(clust_id)):
                    if solution[k][case_id_unique[i]] == clust_id[j]:
                        h[i,j] = 1.
                    else:
                        h[i,j] = 0.
            else:
                h[i,:] = 0.
        h_list.append(h)

    H = np.hstack(tuple(h_list))

    return (H,case_id_unique)

if __name__ == "__main__":
    #lambda_1 = {'x1':1,'x2':1,'x3':1,'x4':2,'x5':2,'x6':3,'x7':3}
    # create missing case_ids
    lambda_1 = {'x4':2,'x5':2,'x6':3}
    lambda_2 = {'x1':2,'x2':2,'x3':2,'x4':3,'x5':3,'x6':1,'x7':1}
    lambda_3 = {'x1':1,'x2':1,'x3':2,'x4':2,'x5':3,'x6':3,'x7':3}
    lambda_4 = {'x1':'1','x2':'2','x3':'NA','x4':'1','x5':'2','x6':'NA','x7':'NA'}
    solution = {'sol_1':lambda_1,'sol_2':lambda_2,'sol_3':lambda_3,\
            'sol_4':lambda_4}
    H,case_id_unique = construct_hypergraph(solution,False)
    print H
    print case_id_unique
