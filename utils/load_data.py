import numpy as np
import csv
import string
from read_txt_data import read_txt_data
import os
from check_features_type import check_features_type
import pandas as pd
import pdb

def load_data_pheno(filename):
    """ This function loads phenotype data from the original 10,000 dataset
    Parameters
    ----------
    filename: string
        string containing the filename

    Returns
    -------
    feature_name: list, len(n_features)
        list containing all the feature names in the given file

    case_id_pheno: list, len(n_instances)
        list containing all the case ids in the given file

    data_pheno: array, shape(n_instances,n_features)
        phenotype data in the given file
    """
    tmp = read_txt_data(filename)
    feature_name = list(tmp[0,:])[1:]
    case_id_pheno = list(tmp[1:,0])
    data_pheno = tmp[1:,1:]

    return (feature_name,case_id_pheno,data_pheno)

def load_cg_merged(filename):
    """ This script loads the merged dataset stored in "cgSubtypeMerged.csv",
    which will be used to evaluate cluster characteristics of various
    clustering solutions.
    Parameters
    ----------
    filename: string
        file containing the merged dataset, including phenotypes, biomarkers,
        imaging features
    
    Returns
    -------
    feature_name: list, len(n_features)
        all the features names in the given dataset
    
    case_id: list, len(n_instances)
        all the IDs of patients in the given dataset

    data: array, shape(n_instances,n_features)
        data in the given dataset
    """
    csvfile = open(filename,"rb")
    csvreader = csv.reader(csvfile)
    lines = [line for line in csvreader]
    csvfile.close()
    
    # number of samples
    n_instances = len(lines) - 1

    # number of features
    n_features = len(lines[0]) - 1

    # extract feature names
    feature_name = lines[0][1:]

    # extract case ids
    case_id = []

    # extract dataset
    data = np.empty((n_instances,n_features),dtype=list)
    for i in range(n_instances):
        case_id.append(lines[i+1][0])
        data[i,:] = lines[i+1][1:]
    
    return (feature_name,case_id,data)

def load_feat_pheno_dict(filename):
    """ This function loads the dictionary information of the phenotype
    features in 10,000 dataset
    Parameters
    ----------
    filename: string
        file containing the feature information

    Returns
    -------
    feat_pheno_dict: dictionary
        feature names are used as key and feature info are used as values

    feat_var_names: list
        variable names for feature information
    """
    csvfile = open(filename,"rb")
    csvreader = csv.reader(csvfile,delimiter=',')
    lines = [line for line in csvreader]
    csvfile.close()
    feat_pheno_dict = {}
    for i in range(1,len(lines)):
        feat_pheno_dict[lines[i][1]] = lines[i][2:]
    feat_var_names = lines[0][2:]
    return (feat_pheno_dict,feat_var_names)

# Note this function determine feature types from values in the original
# dataset, which might cause misleading results for some features. Therefore,
# we use the information providied by Mike instead 
def load_feat_pheno_use(filename_feat_pheno_sel,feature_name,data_pheno,\
        filename_feat_pheno_dict,filename_output):
    """ This function loads features used for phenotype evaluation
    Parameters
    ----------
    filename_feat_pheno_sel: string
        file contain a list of feature names of interest, note each feature 
        should be specified in a row
    
    feature_name: list, len(n_features)
        all feature names in the complete phenotype dataset

    data_pheno: array, shape(n_instances,n_features)
        phenotype data in the complete complete phenotype dataset

    filename_feat_pheno_dict: string
        file containing information of phenotype features

    filename_output: string
        write outputs of feature information into a .csv file

    Returns
    -------
    feat_pheno_use: list
        list containing the phenotype feature names

    feat_pheno_missing: list
        features that do not appear in the complete phenotype dataset

    feat_pheno_use_type: list
        feature types of features in feat_pheno_use

    feat_pheno_use_source: list
        feature sources of feature in feat_pheno_use
    """

    textfile = open(filename_feat_pheno_sel,"r")
    lines = textfile.readlines()
    feat_pheno_sel = []
    for item in lines:
        feat_pheno_sel.append(item.rstrip('\n'))
    textfile.close()
    
    # check whether selected features appears in the original dataset
    feat_pheno_use = []
    # indices of feat_pheno_use in the original phenotype dataset
    idx_feat_pheno_use = []
    for k in feat_pheno_sel:
        if k in feature_name:
            feat_pheno_use.append(k)
            idx_feat_pheno_use.append(feature_name.index(k))

    # output features that does not appear in the 10,000 dataset
    feat_pheno_missing = list(set(feat_pheno_sel)-set(feat_pheno_use))
    
    # determine feature types
    feat_pheno_use_type = \
            check_features_type(np.vstack((np.array(feat_pheno_use),\
            data_pheno[:,idx_feat_pheno_use])))
    
    
    # load dictionary for phenotype features
    feat_pheno_dict,feat_var_names = \
            load_feat_pheno_dict(filename_feat_pheno_dict)
    
    # extract feature information
    feat_pheno_use_info = []
    for k in feat_pheno_use:
        if k in feat_pheno_dict.keys():
            feat_pheno_use_info.append(feat_pheno_dict[k])
        else:
            feat_pheno_use_info.append(['Unknown']*len(feat_var_names))
            
    csvfile = open(filename_output,"wb")
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['FeatureName','FeatureType']+feat_var_names)
    for i in range(len(feat_pheno_use)):
        csvwriter.writerow([feat_pheno_use[i],feat_pheno_use_type[i]]+\
                feat_pheno_use_info[i])
    csvfile.close()

    return (feat_pheno_use,feat_pheno_missing,feat_pheno_use_type,\
            feat_pheno_use_info)


def load_feat_use(filename,feature_name):
    """ This function load feature information from the input file provided by
    user.
    Parameters
    ----------
    filename: string
        file containing information of the selected features, an example input
        file is "feat_pheno_use_infoV2.csv"
    
    feature_name: list, len(n_features)
        all the features appearing in the original dataset

    Returns
    -------
    feat_use: list
        features appearing in both selected feature list and the original
        dataset

    feat_use_class: list
        classes of feat_use

    feat_use_type: list
        types of feat_use

    feat_missing: list
        features appearing in selected feature list but missing in the original
        dataset
    """
    csvfile = open(filename,"rb")
    csvreader = csv.reader(csvfile,delimiter=',')
    lines = [line for line in csvreader]
    csvfile.close()
    
    # use list instead dictionary to keep feature order specified in the input
    # file manually
    feat_use = []
    feat_use_class = []
    feat_use_type = []
    # missing features not found in the data file
    feat_missing = []
    for i in range(len(lines)-1):
        # check if this feature appears in the original dataset
        if lines[i+1][0] in feature_name:
            feat_use.append(lines[i+1][0])
            feat_use_class.append(lines[i+1][1])
            feat_use_type.append(lines[i+1][2])
        else:
            feat_missing.append(lines[i+1][0])

    
    return (feat_use,feat_use_class,feat_use_type,feat_missing)


def load_data_geno(filename):
    """ This function loads genotype data
    Parameters
    ----------
    filename: string
        string containing the filename
    
    Returns
    -------
    geno_feat_name: list, len(n_features_geno)
        all the genotype features

    case_id_geno: list, len(n_instances_geno)
        all the case ids in the given file

    data_geno: array, shape(n_instances_geno,n_features_geno)
        geno type data in the given file
    """
    txtfile = open(filename,"rb")
    csvreader = csv.reader(txtfile,delimiter='\t')
    lines = [line for line in csvreader]
    txtfile.close()

    # genotype feature name, case id, genotype data
    geno_feat_name = lines[0][2:]
    case_id_geno = []
    data_geno = []
    for i in range(len(lines)-1):
        case_id_geno.append(lines[i+1][0])
        data_geno.append(lines[i+1][2:])
    data_geno = np.array(data_geno)
    
    return (geno_feat_name,case_id_geno,data_geno)

def load_solutions(dir_name):
    """ This function loads all the clustering solutions under the given
    directory
    Parameters
    ----------
    dir_name: string
        directory name

    Returns
    -------
    solution: dictionary
        save each solution in a dictionary, where case_id as key and cluster
        asignment as value
    
    feature_of_interest: dictionary
        specified variables for each solution

    subtype_name: dictionary
        meaning of each subtype in the given solution

    description: dictionary
        description of the algorithm or the clinical rules
    """
    # each solution is identified by its file name, e.g. "kmeans_pjc"
    solution_id = []
    # optional 3: each variable list is identified by its file name, e.g. 
    # "kmeans_pjc_variable"
    feature_of_interest_id = []
    # optional 1: translate subtype numbers into subtype names, e.g.
    # "kmeans_pjc_names"
    subtype_name_id = []
    # optional 2: description of the algorithm or the clinical rules, e.g.
    # "kmean_pjc_description"
    description_id = []
    
    # parse file names under the given directory dir_name
    assert os.path.exists(dir_name),"input directory name is incorrect"
    for cur,_dirs,files in os.walk(dir_name):
        pass
    for item_suf in files:
        # file name should start with a letter instead of special characters 
        if item_suf[0] in list(string.ascii_letters):
            """
            # keep only file name without suffix
            item = item_suf.split('.')[0]
            if item[-8:] == 'variable':
                feature_of_interest_id.append(item)
            elif item[-5:] == 'names':
                subtype_name_id.append(item)
            elif item[-10:] == 'description':
                description_id.append(item)
            else:
                solution_id.append(item)
            """
            solution_id.append(item_suf)
    # save each solution in a dictionary, where case_id as key and cluster
    # assignment as value
    solution = {}
    for item in solution_id:
        #txtfile = open(dir_name+'/'+item,"rb")
        ## detect delimiter
        #header = txtfile.readline()
        #if header.find(",") != -1:
        #    tmp = ","
        #elif header.find("\t") != -1:
        #    tmp = "\t"
        #elif header.find(' ') != -1:
        #    tmp = ' '
        #else:
        #    tmp = "\t"
        #txtfile.seek(0)
        #csvreader = csv.reader(txtfile,delimiter=tmp)
        #pdb.set_trace()        
        #lines = [line for line in csvreader]
        #txtfile.close()
        ## store one clustering solution
        #tmp = header.split(tmp)
        #tmp_dict = {tmp[0]:tmp[1]}
        #for i in lines:
        #    tmp_dict[i[0]] = int(i[1])

        df = pd.read_csv(dir_name + item, header=None)
        tmp_dict = {}
        for i in xrange(0, df.shape[0]):
            tmp_dict[df.ix[i][0]] = df.ix[i][1]

        solution[item] = tmp_dict

    # optional 3: save variable list for each solution in a list
    feature_of_interest = {}
    for item in feature_of_interest_id:
        txtfile = open(dir_name+'/'+item,"rb")
        csvreader = csv.reader(txtfile)
        lines = [line for line in csvreader]
        txtfile.close()
        # store the variable list for one clustering solution
        tmp_list = []
        for i in lines:
            tmp_list.append(i[0])
        feature_of_interest[item] = tmp_list

    # optional 1: save each line in a list
    subtype_name = {}
    for item in subtype_name_id:
        txtfile = open(dir_name+'/'+item,"rb")
        csvreader = csv.reader(txtfile)
        lines = [line for line in csvreader]
        txtfile.close()
        subtype_name[item] = lines

    # optional 2: description information can be stored in a list
    description = {}
    for item in description_id:
        txtfile = open(dir_name+'/'+item,"rb")
        csvreader = csv.reader(txtfile)
        lines = [line for line in csvreader]
        txtfile.close()
        description[item] = lines
    
    return (solution,feature_of_interest,subtype_name,description)


def load_gwas_res(filename,symbol_m='NA'):
    """ This script loads GWAS analysis results provided by MIke
    Parameters
    ----------
    filename: string
        filename containing the GWAS solutions
    
    symbol_m: string
        symbol for missing values, default to be 'NA'

    Returns
    -------
    solution_id: list
        all the solutions

    feat_gwas: list
        feature names of GWAS analysis results

    data_gwas: array, shape(n_solutions,n_feat_gwas)
        array containing the data
    """
    csvfile = open(filename,"rb")
    csvreader = csv.reader(csvfile)
    lines = [line for line in csvreader]
    feat_gwas = lines[0][1:]
    solution_id = []
    data_gwas = []
    for i in range(1,len(lines)):
        solution_id.append(lines[i][0])
        tmp = []
        for j in range(1,len(lines[i])):
            if lines[i][j] == symbol_m:
                tmp.append('nan')
            else:
                tmp.append(float(lines[i][j]))
        data_gwas.append(tmp)
    data_gwas = np.array(data_gwas).astype('float')

    return (solution_id,feat_gwas,data_gwas)



