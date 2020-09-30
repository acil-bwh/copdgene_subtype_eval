import numpy as np
from scipy import stats
import csv
import copy
import os
import pdb
from check_features_type import check_features_type

def eval_phenotype(feature_name,case_id_pheno,data_pheno,\
        solution,feature_of_interest,subtype_name,description,\
        feat_use,feat_use_class,feat_use_type,dir_output,symbol_m='NA'):
    """ This function evaluate clustering solutions through phenotype
    characteristics
    Parameters
    ----------
    feature_name: list, len(n_features)
        list containing all the feature names in the given file

    case_id_pheno: list, len(n_instances)
        list containing all the case ids in the given file

    data_pheno: array, shape(n_instances,n_features)
        phenotype data in the given file
    
    solution: dictionary
        save each solution in a dictionary, where case_id as key and cluster
        asignment as value
    
    feature_of_interest: dictionary
        specified variables for each solution

    subtype_name: dictionary
        meaning of each subtype in the given solution

    description: dictionary
        description of the algorithm or the clinical rules
    
    feat_use: list
        the common set of variables for all solutions
        feature names are used as keys, feature types and classes are used as
        values

    feat_use_class: list
        feature classes for feat_use

    feat_use_type: list
        feature types for feat_use

    dir_output: string
        directory that contains the phenotype analysis results
    
    symbol_m: string, default to be 'NA'
        symbol for missing values

    Returns
    -------
    res_pheno: dictionary
        for each clustering solution, we generate the following variables
            a) feat_type: list, len
                
            b) res_feat_clust: array, shape(n_features_use,n_clusters)
                res_feat_clust[j,i] represents the phenotype characteristic 
                summary in i-th cluster for j-th feature. The format differs 
                according to feature types
             
            c) anova_p: vector, len(n_features_use) 
                given a clustering solution, compute the ANOVA p-value for each
                feature
            
            d) kw_p: vector, len(n_features_use)
                given a clustering solution, compute the Kruskal-Wallis
                p-values for each feature

    mtr_p: array, shape(n_solution,n_feature)
        p_value matrix for futher visualization

    feat_name_reorder: len(n_features)
        features reordered according to feature class, which is manually
        specified by Mike
    """
    res_pheno = {}
    counter = 1
    mtr_p = []
    feat_name_reorder = []

    # analyze each solution one by one sequentially
    for item in solution.keys():
        mtr_p.append([])
        # get feature list for this clustering solution
        # if feature list is specified in the input, then use the specification,
        # otherwise use a common default feature list
        if item+'_variable' in feature_of_interest.keys():
            feature_use = feature_of_interest[item+'_variable']
        else:
            feature_use = feat_use
        
        # indices of feature_use in the original phenotype dataset
        idx_feature_use = []
        for k in feature_use:
            idx_feature_use.append(feature_name.index(k))
        # determine feature type for this set of features
        # determine through data values in the original file 
        #feature_use_type = check_features_type(np.vstack((np.array(feature_use),\
        #        data_pheno[:,idx_feature_use])))
        # determine through manual specification
        feature_use_type = feat_use_type
        feature_use_class = feat_use_class

        case_id_use = []
        for i in solution[item].keys():
            if i in case_id_pheno:
                case_id_use.append(i)
        
        # label corresponding to case_id_use
        label_use = []
        for i in case_id_use:
            label_use.append(solution[item][i])
        label_use = np.array(label_use)

        # find the number of clusters
        label_unique = np.unique(label_use)
        n_clusters = len(label_unique)
        
        res_feat_clust = np.zeros((len(feature_use),n_clusters),dtype=list)
        
        # indices in original phenotype dataset of case ids in all clusters, put
        # the indices of the same cluster in one list
        idx_case_id_groups = []
        # traverse each cluster
        for i in range(n_clusters):
            # find the indices in original phenotype dataset of case ids in one cluster
            idx_case_id_clust = []
            for k in range(len(label_use)):
                if label_use[k] == label_unique[i]:
                    idx_case_id_clust.append(case_id_pheno.index(case_id_use[k]))
            idx_case_id_groups.append(idx_case_id_clust)

            # traverse each feature
            for j in range(len(feature_use)):
                tmp = data_pheno[idx_case_id_clust,idx_feature_use[j]]
                # indices with no missing values
                idx_sel = (tmp!=symbol_m)
                # make sure the sliced vector is not empty
                if sum(idx_sel) > 0:
                    # for continuous features, compute median and IQR in each
                    # cluster
                    if feature_use_type[j] in ['continuous','interval','ordinal']:
                        tmp_1 = np.double(tmp[idx_sel])
                        # compute median and IQR range
                        res_feat_clust[j,i] = [np.median(tmp_1),\
                                (np.percentile(tmp_1,25),np.percentile(tmp_1,75)),\
                                (np.mean(tmp_1),np.std(tmp_1))]
                    if feature_use_type[j] in ['categorical','binary']:
                        tmp_1 = tmp[idx_sel]
                        tmp_1_unique = np.unique(tmp_1)
                        tmp_1_dist = {}
                        for k in tmp_1_unique:
                            tmp_1_dist[k] = 0
                        for k in range(len(tmp_1)):
                            for l in range(len(tmp_1_unique)):
                                if tmp_1[k] == tmp_1_unique[l]:
                                    tmp_1_dist[tmp_1_unique[l]] += 1
                        res_feat_clust[j,i] = tmp_1_dist
                else:
                    res_feat_clust[j,i] = 'na'
        # compute the percent of missing values for each feature
        perc_missing_feat = []
        idx_case_id_all = []
        for k in idx_case_id_groups:
            idx_case_id_all += k
        for j in range(len(feature_use)):
            tmp = data_pheno[idx_case_id_all,idx_feature_use[j]]
            # compute missing value percentage
            perc_missing_feat.append(sum(tmp == symbol_m) *\
                    100./len(idx_case_id_all))

        # statistical significance test
        anova_f = np.zeros(len(feature_use))
        anova_p = np.zeros(len(feature_use))
        kw_h = np.zeros(len(feature_use))
        kw_p = np.zeros(len(feature_use))
        for j in range(len(feature_use)):
            if feature_use_type[j] in \
                    ['continuous','interval','ordinal','binary']:
                groups = []
                for i in range(n_clusters):
                    if label_unique[i] == -9:
                        pass
                    else:
                        tmp = data_pheno[idx_case_id_groups[i],\
                                idx_feature_use[j]]
                        idx_sel = (tmp!=symbol_m)
                        if sum(idx_sel) >= 2:
                            groups.append(np.double(tmp[idx_sel]))

                #print "group number = ", len(groups)
                # output 'nan' if the groups is less than 2
                if len(groups) < 2:
                    anova_f[j] = float('nan')
                    anova_p[j] = float('nan')
                    kw_h[j] = float('nan')
                    kw_p[j] = float('nan')
                # Kruskal-Wallis test for continuous variables
                elif feature_use_type[j] != 'binary':
                    anova_f[j],anova_p[j] = stats.f_oneway(*groups)
                    kw_h[j],kw_p[j] = stats.mstats.kruskalwallis(*groups)
                # chi2 test for binary variables
                elif feature_use_type[j] == 'binary':
                    # construct contingence table
                    tmp_tab = np.zeros((2,len(groups)))
                    for k in range(len(groups)):
                        tmp_tab[0,k] = np.sum(groups[k]==0)
                        tmp_tab[1,k] = np.sum(groups[k]==1)
                    # there should not exist any zero elements in contingence
                    # table
                    expected = stats.contingency.expected_freq(tmp_tab)
                    if np.any(expected == 0):
                        kw_p[j] = float('nan')
                        #print "there are zeros elements in contingence table"
                    else:
                        chi2_res = stats.chi2_contingency(tmp_tab)
                        kw_p[j] = chi2_res[1]
                    
        res_pheno[item] = [copy.deepcopy(res_feat_clust),copy.copy(anova_p),\
                copy.copy(kw_p)]
        
        # write the analysis result into csv files
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)
        csvfile = open(dir_output+"/"+item+'_clust_analysis.csv',"wb")
        csvwriter = csv.writer(csvfile)
        # write headers
        head_id = []
        item.split('.')[0]
        for k in list(label_unique):
            head_id.append(item.split('.')[0] + ': ' + str(k))
        csvwriter.writerow(['Cluster ID']+ head_id +\
                ['Pvalue','%Missing'])
        tmp = ['#Samples']
        for i in range(n_clusters):
            tmp.append(sum(label_use == label_unique[i]))
        csvwriter.writerow(tmp)

        # write according to feature class
        # unique class
        tmp = list(set(feature_use_class))
        feature_use_class_unique = []
        if 'Demographic' in tmp:
            feature_use_class_unique.append('Demographic')
        if 'Spirometry' in tmp:
            feature_use_class_unique.append('Spirometry')
        if 'Function' in tmp:
            feature_use_class_unique.append('Function')
        if 'Imaging' in tmp:
            feature_use_class_unique.append('Imaging')
        if 'Comorbidity' in tmp:
            feature_use_class_unique.append('Comorbidity')
        if 'Longitudinal' in tmp:
            feature_use_class_unique.append('Longitudinal')
        if 'Biomarker' in tmp:
            feature_use_class_unique.append('Biomarker')
        if 'SNPs' in tmp:
            feature_use_class_unique.append('SNPs')
        tmp_1 = list(set(tmp)-set(feature_use_class_unique))
        if len(tmp_1) > 0:
            feature_use_class_unique += tmp_1

        # group feature indices by class
        feature_use_class_group = []
        for k in range(len(feature_use_class_unique)):
            feature_use_class_group.append([])

        for j in range(len(feature_use)):
            tmp = feature_use_class_unique.index(feature_use_class[j])
            feature_use_class_group[tmp].append(j)
        
        for k in range(len(feature_use_class_unique)):
            # write feature class
            csvwriter.writerow([])
            csvwriter.writerow([feature_use_class_unique[k]])
            for j in feature_use_class_group[k]:
                # store the feature names after reordering
                if counter == 1:
                    feat_name_reorder.append(feature_use[j])
                # use percentage for binary variables
                if feature_use_type[j] != 'binary':
                    tmp = [feature_use[j]]
                else:
                    tmp = [feature_use[j]+" (%'1')"]

                for i in range(n_clusters):
                    if res_feat_clust[j,i] != 'na':
                        if feature_use_type[j] in ['continuous','interval','ordinal']:
                            tmp_mean = \
                                    float("{0:.2f}".format(res_feat_clust[j,i][2][0]))
                            tmp_std = \
                                    float("{0:.2f}".format(res_feat_clust[j,i][2][1]))
                            tmp_median = \
                                    float("{0:.2f}".format(res_feat_clust[j,i][0]))
                            tmp_iqr_l = \
                                    float("{0:.2f}".format(res_feat_clust[j,i][1][0]))
                            tmp_iqr_r = \
                                    float("{0:.2f}".format(res_feat_clust[j,i][1][1]))
                            if feature_use_class[j] != 'SNPs':
                                tmp.append(str(tmp_median) + ' (' + str(tmp_iqr_l) + \
                                        ',' + str(tmp_iqr_r) + ')')
                            else:
                                tmp.append(str(tmp_mean) + ' (' + str(tmp_std)\
                                        + ')')
                        if feature_use_type[j] == 'binary':
                            assert '1' in res_feat_clust[j,i].keys() \
                                    or '0' in res_feat_clust[j,i].keys(), \
                            " '1','0' should be coded values for binary" +\
                            "variables"
                            if '1' in res_feat_clust[j,i].keys():
                                tmp_cnt_1 = res_feat_clust[j,i]['1']
                            else:
                                tmp_cnt_1 = 0

                            if '0' in res_feat_clust[j,i].keys():
                                tmp_cnt_0 = res_feat_clust[j,i]['0']
                            else:
                                tmp_cnt_0 = 0
                            if tmp_cnt_1 + tmp_cnt_0 > 0:
                                tmp.append(float("{0:.2f}".\
                                        format(tmp_cnt_1*1./(tmp_cnt_1+tmp_cnt_0))))
                            else:
                                tmp.append(0)

                        if feature_use_type[j] == 'categorical':
                            tmp.append(res_feat_clust[j,i])
                    else:
                        tmp.append('na')
                
                # write statistical test
                if feature_use_type[j] in \
                        ['continuous','interval','ordinal','binary']:
                    tmp = tmp + ['%.4e' % kw_p[j]]
                    mtr_p[-1].append(kw_p[j])
                else:
                    tmp = tmp + ['na']
                    mtr_p[-1].append('na')
                tmp.append(float("{0:.2f}".format(perc_missing_feat[j])))
                csvwriter.writerow(tmp)
        csvfile.close()

        print 'analyzing solution '+str(counter)+':\t'+item
        counter += 1
    
    return (res_pheno,np.array(mtr_p),feat_name_reorder)

