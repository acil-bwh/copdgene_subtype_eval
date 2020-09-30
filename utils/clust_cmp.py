import numpy as np
from sklearn import metrics
import scipy.cluster.hierarchy as sch
#import pylab
from matplotlib import pylab
import csv
from compute_purity import compute_purity
import pdb

def compute_sim_solution(solution,flag_nmi):
    """ This function is used to compute the similarity between clustering
    solutions

    Parameters
    ----------
    solutions: dictionary of dicts, len(n_solution)
        each solution is indexed by solution_id, in each solution case_ids are
        used as keys and cluster assignments are used as labels. Note that the
        samples that are not assigned to any cluster have label -9

    flag_nmi: int
        1: only consider the overlapping of samples between two solutions and
        also remove samples that are not assigned to any cluster, i.e. with
        label -9
        2: consider the union of samples in two solutions, samples that are not
        assigned (-9) or that only appear in one solution are given label -9
        3: consider overlapping samples and compute purity score
        4: consider overlapping samples and compute rand index

    Returns
    -------
    sim_soltuion: array, shape(n_solution,n_solution)
        (i,j)-th element represents the similarity between solution i and
        solution j
    """

    # compute similarity between pariwise solutions
    solution_id = solution.keys()
    n_solution = len(solution_id)

    sim_solution = np.eye(n_solution)    
    if n_solution <= 1:
        print "There's no more than 1 solution in the given directory"
    else:
        for i in range(n_solution):
            for j in range(n_solution):
                # i-th solution
                tmp_i = solution[solution_id[i]]
                # j-th solution
                tmp_j = solution[solution_id[j]]
                # find the intersection of case ids between two solutions
                tmp_ij = list(set(tmp_i.keys()).intersection(set(tmp_j.keys())))
                # find the union of case ids between two solutions
                tmp_union = list(set(tmp_i.keys()).union(set(tmp_j.keys())))

                # find labels for two solutions
                label_i = []
                label_j = []

                # NMI_1: consider only the overlapping case-ids between two
                # solutions, leave out samples that are not assigned to any cluster
                if flag_nmi != 2:
                    for k in range(len(tmp_ij)):
                        if tmp_i[tmp_ij[k]] != -9 and tmp_j[tmp_ij[k]] != -9:
                            label_i.append(tmp_i[tmp_ij[k]])
                            label_j.append(tmp_j[tmp_ij[k]])
                
                # NMI_2: consider the union of case-ids between two solutions,
                # case-ids only appear in one solution are assigned label -9
                if flag_nmi == 2:
                    # add intersection first 
                    for k in range(len(tmp_ij)):
                        if tmp_i[tmp_ij[k]] != -9 and tmp_j[tmp_ij[k]] != -9:
                            label_i.append(tmp_i[tmp_ij[k]])
                            label_j.append(tmp_j[tmp_ij[k]])
                    
                    # consider unique elements in solution i
                    for k in list(set(tmp_union).difference(set(tmp_j))):
                        if tmp_i[k] != -9:
                            # add the correct label to i
                            label_i.append(tmp_i[k])
                            # use a unique label and add to j
                            label_j.append(-10)

                    # consider unique elements in solution j
                    for k in list(set(tmp_union).difference(set(tmp_i))):
                        if tmp_j[k] != -9:
                            label_j.append(tmp_j[k])
                            label_i.append(-10)

                if len(label_i) >= 5:
                    # compute NMI
                    if flag_nmi == 1 or flag_nmi == 2:
                        sim_solution[i,j] = \
                                metrics.normalized_mutual_info_score(label_i,\
                                label_j)

                    # compute purity score
                    if flag_nmi == 3 or flag_nmi == 4:
                        (purity_avg,purity_max) = \
                                compute_purity(np.array(label_i),\
                                np.array(label_j))
                        if flag_nmi == 3:
                            sim_solution[i,j] = purity_max
                        if flag_nmi == 4:
                            sim_solution[i,j] == purity_avg
                #sim_solution[j,i] = sim_solution[i,j]

    return sim_solution

def plot_sim_solution(sim_solution,solution_id,mtr_p,feat_name_reorder,\
        feat_use_name,feat_use_class,\
        solution_id_gwas,feat_gwas,data_gwas,\
        filename_dend,filename_sim_mtr,\
        filename_sol_feat_p,filename_sol_feat_p_csv,\
        fig_format,flag_space):
    """ This function is used to plot the similarity matrix between solutions
    
    Parameters
    ----------
    sim_solution: array, shape(n_solution,n_solution)
        (i,j)-th element represents the similarity between solution i and
        solution j
    
    solution_id: list,len(n_solution)
        solution id for each solution, used as labels for the dendrogram
    
    mtr_p: array, shape(n_solution,n_feat_use)
        p_value matrix, which should be used to do visualization

    feat_name_reorder: len(n_feat_use)
        feature names that are used to characterize clusters
    
    feat_use_name: len(n_feat_use)
        feature names before reordering

    feat_use_class: len(n_feat_use)
        feature class of the selected features

    solution_id_gwas: list
        solution id in GWAS analysis results

    feat_gwas: list
        features in GWAS results

    data_gwas: array, shape(n_solution_id_gwas,n_feat_gwas)
        GWAS analysis results

    filename_dend: string
        filename for the visualization of dendrogram and similarity matrix

    filename_sim_mtr: string
        filename for storing the similarity matrix values

    filename_sol_feat_p: string
        filename for visualizing solution x feature matrix

    filename_sol_feat_p_csv: string
        csv file storing the actual p-values
    
    fig_format: string, 'png' or 'eps'
        output figure format
    
    flag_space: boolean
        whether to add space between different feature groups in visualization

    Returns
    -------
    solution_id_reorder: list
        reorder solution_id after applying hierarchical clustering
    """
    
    n_solution = sim_solution.shape[0]
    # apply hierarchical clustering on the similarity matrix between solutions
    # return linkage matrix
    # get distance matrix D from similarity matrix
    D = 1 - sim_solution

    # plot the similarity matrix with dendrogram
    if n_solution <=32:
        t_size = 8
    else:
        t_size = int(n_solution*0.25)
    fig = pylab.figure(figsize=(t_size,t_size))
    
    # second dendrogram
    ax2 = fig.add_axes([0.2,0.8,0.6,0.19])
    Y_2 = sch.linkage(D,method='average')
    Z2 = sch.dendrogram(Y_2,orientation='top')
    ax2.set_xticks([])
    ax2.set_yticks([])
    idx2 = Z2['leaves']
    
    # reorder solution ids according to dendrogram
    # re-arrange according to rows
    solution_id_reorder = []
    for k in idx2:
        solution_id_reorder.append(solution_id[k])

    # plot similarity matrix after re-arranging solutions 
    axmatrix = fig.add_axes([0.2,0.2,0.6,0.6])
    D = D[idx2,:]
    D = D[:,idx2]
    sim_solution_reorder = 1-D
    im = axmatrix.matshow(sim_solution_reorder, aspect='auto', origin='upper',\
            cmap=pylab.cm.Spectral_r,interpolation='none',vmin=0,vmax=1)
    #im.set_xticks([])
    pylab.yticks(range(n_solution),tuple(solution_id_reorder))
    
    # show x ticks at bottom
    axmatrix.xaxis.set_ticks_position('bottom')
    pylab.xticks(range(n_solution),solution_id_reorder,rotation='vertical')

    # Plot colorbar.
    axcolor = fig.add_axes([0.81,0.2,0.02,0.6])
    pylab.colorbar(im, cax=axcolor)
    #fig.show()
    fig.savefig(filename_dend,format=fig_format)
  
    # save the similarity matrix to csv file
    csvfile = open(filename_sim_mtr,"wb")
    csvwriter = csv.writer(csvfile)
    # header
    csvwriter.writerow([' '] + solution_id_reorder)
    for k in range(len(solution_id_reorder)):
        # format the output
        tmp = []
        for l in list(sim_solution_reorder[k]):
            tmp.append(float("{0:.2f}".format(l)))
        csvwriter.writerow([solution_id_reorder[k]] + tmp)
    csvfile.close()
    
    # phenotype features for evaluating cluster characteristics
    n_feat = len(feat_name_reorder)
    
    # GWAS results features
    n_feat_gwas = len(feat_gwas)
    
    # find the number of feature classes in order to add space between
    # different groups in visualization
    feat_class_id = []
    feat_class_hist = []

    for k in feat_use_class:
        if k not in feat_class_id:
            feat_class_id.append(k)
            feat_class_hist.append(1)
        else:
            feat_class_hist[feat_class_id.index(k)] += 1
    n_feat_class = len(feat_class_id)
    #print feat_class_id,feat_class_hist

    # rearranging the order of solution ids
    mtr_p_reorder = mtr_p[idx2,:]
    mtr_p_thd = np.zeros((n_solution,n_feat + n_feat_gwas))

    # iterate each solution according to the order after reshuffling
    for i in range(n_solution):
        # take logarithm of the p-values and thresholding
        for j in range(n_feat):
            if mtr_p_reorder[i,j] <= 1e-5:
                mtr_p_thd[i,j] = 5
            else:
                mtr_p_thd[i,j] = -np.log10(mtr_p_reorder[i,j])
        
        # for GWAS result features, apply the same procedure
        for j in range(n_feat,n_feat+n_feat_gwas):
            if solution_id_reorder[i] in solution_id_gwas:
                idx_tmp = solution_id_gwas.index(solution_id_reorder[i])
                val_tmp = data_gwas[idx_tmp,j-n_feat]
                # p_values
                if val_tmp <= 1e-5:
                    mtr_p_thd[i,j] = 5
                else:
                    mtr_p_thd[i,j] = -np.log10(val_tmp)

    # plot the solution x feature matrix, add space between different feature
    # groups
    feat_dict = {}
    for k in range(len(feat_use_name)):
        feat_dict[feat_use_name[k]] = feat_use_class[k]

    mtr_p_thd_space = np.zeros((n_solution,n_feat+n_feat_gwas+n_feat_class))
    feat_name_reorder_space = []
    for k in feat_name_reorder:
        if len(feat_name_reorder_space) == 0:
            feat_name_reorder_space.append(k)
        else:
            if feat_dict[feat_name_reorder_space[-1]] == feat_dict[k]:
                feat_name_reorder_space.append(k)
            else:
                feat_name_reorder_space.append(' ')
                feat_name_reorder_space.append(k)
    feat_name_reorder_space.append(' ')
    assert len(feat_name_reorder_space) == n_feat+n_feat_class

    idx_l = 0
    offset = 0
    for k in feat_class_hist:
        idx_r = idx_l + k
        mtr_p_thd_space[:,(idx_l+offset):(idx_r+offset)] = \
                mtr_p_thd[:,idx_l:idx_r]
        for i in range(n_solution):
            mtr_p_thd_space[i,idx_r + offset] = float('nan')
        idx_l = idx_r
        offset += 1
    
    mtr_p_thd_space[:,-n_feat_gwas:] = mtr_p_thd[:,-n_feat_gwas:]

    # determine figure size
    if n_feat <= 48:
        t_size_feat = 12
    else:
        t_size_feat = int((n_feat+n_feat_gwas+n_feat_class)*0.25)
    fig2 = pylab.figure(figsize=(t_size_feat,t_size))
    axmatrix2 = fig2.add_axes([0.15,0.1,0.75,0.6])
    if flag_space == True:
        im2 = axmatrix2.matshow(mtr_p_thd_space,aspect='auto',origin='upper',\
                cmap=pylab.cm.Spectral_r,interpolation='none',vmin=0,vmax=5)
        pylab.xticks(range(n_feat+n_feat_gwas+n_feat_class),\
                feat_name_reorder_space+feat_gwas,rotation='vertical')
    else:
        im2 = axmatrix2.matshow(mtr_p_thd,aspect='auto',origin='upper',\
                cmap=pylab.cm.Spectral_r,interpolation='none',vmin=0,vmax=5)
        pylab.xticks(range(n_feat+n_feat_gwas),\
                feat_name_reorder+feat_gwas,rotation='vertical')
    
    pylab.yticks(range(n_solution),solution_id_reorder)
    axcolor = fig2.add_axes([0.91,0.1,0.02,0.6])
    pylab.colorbar(im2,cax=axcolor)
    fig2.savefig(filename_sol_feat_p,format=fig_format)
    
    # save p-value matrix after thresholding to a csv file
    csvfile = open(filename_sol_feat_p_csv,"wb")
    csvwriter = csv.writer(csvfile)
    # header
    csvwriter.writerow([' '] + feat_name_reorder + feat_gwas)
    for k in range(n_solution):
        tmp = []
        for l in list(mtr_p_thd[k,:]):
            tmp.append(float("{0:.2f}".format(l)))
        csvwriter.writerow([solution_id_reorder[k]] + tmp)

    return solution_id_reorder

def compute_sim_clust(solution,flag_overlap,figure_size,\
        filename_sim_clust_fig,filename_sim_clust_csv,fig_format):
    """ This script is used to compute the similarity between clusters in
    different clustering solution
    
    Parameters
    ----------
    solution: dictionary, len(n_solution)
        clustering solutions
    
    flag_overlap: boolean
        whether to find the overlap between case_ids in two solutions as base
        set for computing concordance score between pairwise clusters

    fig_size: int
        size of the output figure

    filename_sim_clust_fig: string
        output figure name

    filename_sim_clust_csv: string
        output csv file name
    
    fig_format: string: 'eps' or 'png'
        output figure format

    Returns
    -------
    sim_clust_reorder: array, shape(n_cluster,n_clusters)

    cluster_id_reorder: list, len(n_cluster)
    """

    # compute the total number of cluster and index them
    solution_id = solution.keys()
    n_solution = len(solution_id)

    cluster_id_pre = []
    cluster_id_suf = []
    for item in solution_id:
        tmp = list(set(solution[item].values()).difference({-9}))
        for i in tmp:
            cluster_id_pre.append(item)
            cluster_id_suf.append(i)
    
    # total number of clusters
    n_clusters = len(cluster_id_pre)

    # compute similarity between clusters
    sim_clust = np.zeros((n_clusters,n_clusters))
    
    # find the overlap between samples of two solutions as base set for
    # computing concordance between two clusters
    for i in range(n_clusters):
        # find case_ids for samples in cluster i
        clust_i = []
        for case_id,label in solution[cluster_id_pre[i]].iteritems():
            if label == cluster_id_suf[i]:
                clust_i.append(case_id)
        for j in range(i,n_clusters):
            clust_j = []
            for case_id,label in solution[cluster_id_pre[j]].iteritems():
                if label == cluster_id_suf[j]:
                    clust_j.append(case_id)
            
            # find the overlap between samples of two solutions as base set for
            # computing concordance between two clusters
            if flag_overlap == True:
                tmp = set(solution[cluster_id_pre[i]].keys()).intersection(\
                        set(solution[cluster_id_pre[j]].keys()))
                tmp_deno = len(set(clust_i).union(set(clust_j)).intersection(tmp))

                if tmp_deno > 0:
                    sim_clust[i,j] = len(set(clust_i).intersection(\
                            set(clust_j)).intersection(tmp))*1./tmp_deno
                else:
                    sim_clust[i,j] = 0
            else:
                sim_clust[i,j] = len(set(clust_i).intersection(set(clust_j)))*\
                        1./len(set(clust_i).union(set(clust_j)))
            
            sim_clust[j,i] = sim_clust[i,j]
    
    D = 1-sim_clust
    if n_clusters <= 64:
        figure_size = 16
    else:
        figure_size = int(n_clusters*0.25)
    fig = pylab.figure(figsize=(figure_size,figure_size))
    ax = fig.add_axes([0.2,0.8,0.6,0.19])
    Y = sch.linkage(D,method='average')
    Z = sch.dendrogram(Y,orientation='top')
    ax.set_xticks([])
    ax.set_yticks([])
    idx = Z['leaves']

    # reorder cluster_id according to dendrogram
    cluster_id_pre_reorder = []
    cluster_id_suf_reorder = []
    cluster_id_reorder = []
    for k in idx:
        cluster_id_pre_reorder.append(cluster_id_pre[k])
        cluster_id_suf_reorder.append(cluster_id_suf[k])
        cluster_id_reorder.append(cluster_id_pre[k] + '_' + \
                str(cluster_id_suf[k]))

    # plot clust x clust similarity matrix after re-arranging clusters
    axmatrix = fig.add_axes([0.2,0.2,0.6,0.6])
    D = D[idx,:]
    D = D[:,idx]
    sim_clust_reorder = 1-D
    im = axmatrix.matshow(sim_clust_reorder,aspect='auto',origin='upper',\
            cmap=pylab.cm.Spectral_r,interpolation='none',vmin=0,vmax=1)
    axmatrix.set_xticks([])
    pylab.yticks(range(n_clusters),cluster_id_reorder)
    
    # show x ticks at bottom
    axmatrix.xaxis.set_ticks_position('bottom')
    pylab.xticks(range(n_clusters),cluster_id_reorder,rotation='vertical')

    # plot colorbar
    axcolor = fig.add_axes([0.81,0.2,0.02,0.6])
    pylab.colorbar(im,cax=axcolor)
    fig.savefig(filename_sim_clust_fig,format=fig_format)
    
    # save the similarity matrix to a csv file
    csvfile = open(filename_sim_clust_csv,"wb")
    csvwriter = csv.writer(csvfile)
    # header
    csvwriter.writerow([' '] + cluster_id_reorder)
    for k in range(len(cluster_id_reorder)):
        tmp = []
        for l in list(sim_clust_reorder[k]):
            tmp.append(float("{0:.4f}".format(l)))
        csvwriter.writerow([cluster_id_reorder[k]] + tmp)

    return sim_clust_reorder,cluster_id_reorder

