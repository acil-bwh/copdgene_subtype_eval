""" This script is built for the evaluation of COPD subtype solutions. 
The specification of input/output format  is specified in file 
"Subtype characterization pipeline.html"
"""

import numpy as np
import csv
from utils import load_data
from utils.eval_phenotype import eval_phenotype
from utils import clust_cmp
from argparse import ArgumentParser
import os

desc = """ This script permits you select the directory containing multiple
clustering solutions and directory storing the clustering analysis outputs.
"""
parser = ArgumentParser(description=desc)
parser.add_argument('--di', help='input directory name', \
    dest='dir_input', metavar='<dir_input>', default=None)
parser.add_argument('--do', help='output directory name', \
    dest='dir_output', metavar='<dir_output>', default=None)
parser.add_argument('--resources', help='folder containing COPDGene resource \
    data files needed for analysis', dest='resources', metavar='<dir_input>',
    default=None)

options = parser.parse_args()

############## Step 1: Load Phenotype and Genotype Data #################

# load the original 10,000 COPD phenotype data without imputation
filename_cg_merged = os.path.join(options.resources, "cgSubtypeMerged.csv")
feature_name,case_id_pheno,data_pheno = \
        load_data.load_cg_merged(filename_cg_merged)

# load the original COPD genotype data
#filename_geno = "copd_data/COPDGene_Genotypes.txt"
#geno_feat_name,case_id_geno,data_geno = \
#        load_data.load_data_geno(filename_geno)

############## Step 2: Load Clustering Solutions ###########################

# directory containing multiple clustering solutions
# dir_name = "Results_RF_PJC"
dir_name = options.dir_input
solution,feature_of_interest,subtype_name,description = \
        load_data.load_solutions(dir_name)

################## Step 3: Genetic Association ##########################
# load GWAS analysis results provided by Mike
solution_id_gwas,feat_gwas,data_gwas = load_data.load_gwas_res(\
        os.path.join(options.resources, "heatmapAddon.csv"),'NA')
 
################## Step 4: Evaluation by Phenotypes ##########################

# Variable input table: for each variable desired in output, will need 
# a) variable name; b) variable type; c) variable class; d) transformation
# As Pete suggested: Each cluster solution should be related to the same common
# set of variables

# load information of selected features specified by user (Mike)
filename_feat_use = os.path.join(options.resources, "feat_pheno_use_infoV2.csv")
(feat_use,feat_use_class,feat_use_type,feat_missing) = \
        load_data.load_feat_use(filename_feat_use,feature_name)

# directory that stores phenotype evaluation results
dir_output = options.dir_output
res_pheno,mtr_p,feat_name_reorder = eval_phenotype(feature_name,\
        case_id_pheno,data_pheno,solution,feature_of_interest,subtype_name,\
        description,feat_use,feat_use_class,feat_use_type,dir_output,'NA')

################# Step 5: Overall Clustering Comparison ######################

dir_fig = dir_output + "/fig"
if not os.path.exists(dir_fig):
    os.makedirs(dir_fig)

# specify figure format: 'png' or 'eps'
fig_format = 'png'

# compute NMI_1 and make plots
sim_solution_1 = clust_cmp.compute_sim_solution(solution,1)
clust_cmp.plot_sim_solution(sim_solution_1,solution.keys(),\
        mtr_p,feat_name_reorder,feat_use,feat_use_class,\
        solution_id_gwas,feat_gwas,data_gwas,\
        dir_fig + '/dend_nmi_1.'+fig_format,\
        dir_fig + '/sim_solution_nmi_1.csv',\
        dir_fig + '/solution_feat_p_nmi_1.'+fig_format,\
        dir_fig + '/solution_feat_p_nmi_1.csv',fig_format,False)

# compute NMI_2 and make plots
sim_solution_2 = clust_cmp.compute_sim_solution(solution,2)
clust_cmp.plot_sim_solution(sim_solution_2,solution.keys(),\
        mtr_p,feat_name_reorder,feat_use,feat_use_class,\
        solution_id_gwas,feat_gwas,data_gwas,\
        dir_fig + '/dend_nmi_2.'+fig_format,\
        dir_fig + '/sim_solution_nmi_2.csv',\
        dir_fig + '/solution_feat_p_nmi_2.'+fig_format,\
        dir_fig + '/solution_feat_p_nmi_2.csv',fig_format,True)

# compute maximal concordance the clusters in two clustering solution and
# make plots
sim_solution_3 = clust_cmp.compute_sim_solution(solution,3)
clust_cmp.plot_sim_solution(sim_solution_3,solution.keys(),\
        mtr_p,feat_name_reorder,feat_use,feat_use_class,\
        solution_id_gwas,feat_gwas,data_gwas,\
        dir_fig + '/dend_concordance.'+fig_format,\
        dir_fig + '/sim_solution_concordance.csv',\
        dir_fig + '/solution_feat_p_concordance.'+fig_format,\
        dir_fig + '/solution_feat_p_concordance.csv',fig_format,True)

# compute cluster x cluster similarity matrix
# find the overlap between samples of two solutions as base set for computing
# concordance between two clusters
sim_clust_reorder_1,cluster_id_reorder_1 = \
        clust_cmp.compute_sim_clust(solution,\
        True,64,dir_fig + '/sim_clust_1.'+fig_format,\
        dir_fig + '/sim_clust_1.csv',fig_format)
# instead of finding the overlap between two solutions as base set, compute
# concordance between two clusters directly
sim_clust_reorder_2,cluster_id_reorder_2 = \
        clust_cmp.compute_sim_clust(solution,\
        False,64,dir_fig + '/sim_clust_2.'+fig_format,\
        dir_fig + '/sim_clust_2.csv',fig_format)

