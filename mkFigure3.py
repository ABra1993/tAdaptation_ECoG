# required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

from modelling_utils_paramInit import paramInit

"""

Author: A. Brands

"""

############################################################################################## ADAPT CODE HERE
##############################################################################################################
##############################################################################################################
##############################################################################################################

# define root directory
dir = '/home/amber/OneDrive/code/nAdaptation_ECoG_git/'
# dir = '/Users/a.m.brandsuva.nl/Library/CloudStorage/OneDrive-UvA/code/nAdaptation_ECoG_git/'

##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################

# visual areas (labels)
VA = ['V1-V3', 'VOTC', 'LOTC']

# import image classes
stim_cat = np.loadtxt(dir+'variables/cond_stim.txt', dtype=str)

# electrode coordinates
electrodes_visuallyResponsive = pd.read_csv(dir+'subject_data/electrodes_visuallyResponsive_manuallyAssigned.txt', header=0, index_col=0, delimiter=' ')
n_electrodes = len(electrodes_visuallyResponsive)

# extract electrode indices per visual area (i.e. V1-V3, LOTC, VOTC)
VA_name_idx_temp = {}
for i in range(n_electrodes):
    VA_name_current = electrodes_visuallyResponsive.loc[i, 'varea']
    if VA_name_current not in VA_name_idx_temp:
        VA_name_idx_temp[VA_name_current] = [i]
    else:
        VA_name_idx_temp[VA_name_current].append(i)
VA_name_idx = {}
VA_name_idx = {k: VA_name_idx_temp[k] for k in VA}
print(VA_name_idx, '\n')

# computational models
models = ['DN', 'csDN']

# iterate over electrode and retrieve cross-validated performance for DN models and cbDN (fitted for P and NP trials)
r2_values_mean      = np.zeros((len(VA), len(models)))
r2_values_variance  = np.zeros((len(VA), len(models)))

# iterate over electrode and retrieve cross-validated performance for csDN (fitted for all image classes seperately)
count_VA = 0
for _, value in VA_name_idx.items():

    # determine number of electrodes
    n_electrodes = len(value)

    # initiate dataframes
    r2_values_current = np.zeros((n_electrodes, len(models)))

    for i in range(n_electrodes):
    # for i in range(2):

        # retrieve info current electrode
        subject = electrodes_visuallyResponsive.subject[value[i]]
        electrode_name = electrodes_visuallyResponsive.electrode[value[i]]

        # cbdn_fine model (overall r2)
        for idx, model in enumerate(models):
            temp = pd.read_csv(dir+'modelFit/visuallyResponsive/' + subject + '_' + electrode_name + '/param_' + model + '.txt', header=0, delimiter=' ', index_col=0)
            r2_values_current[i, idx] = temp.loc[:, 'r2']

    # average values
    for i in range(len(models)):
        r2_values_mean[count_VA, i] = np.mean(r2_values_current[:, i])
        r2_values_variance[count_VA, i] = np.std(r2_values_current[:, i])/math.sqrt(n_electrodes)

    # increment count
    count_VA = count_VA + 1 


# plot cross validation
fig = plt.figure()
ax = plt.gca()

# seperate axes
sns.despine(offset=10)

# set barwidth
barWidth = 0.2

# fontsizes
fontsize_label  = 18
fontsize_tick   = 24
fontsize_legend = 15

# # set height of bar
model1 = r2_values_mean[:, 0]
model2 = r2_values_mean[:, 1]

# Set position of bar on X axis
br1 = np.arange(len(model1))
br2 = [x + barWidth for x in br1]
br = [br1, br2]

# visualize mean
ax.bar(br1, model1, color='white', width = barWidth,
        edgecolor = 'black', label = models[0], yerr=r2_values_variance[:, 0])
ax.bar(br2, model2, color = np.array([212, 0, 0])/255 , width = barWidth,
        edgecolor = np.array([212, 0, 0])/255, label = models[1], yerr=r2_values_variance[:, 0])

# axes
ax.set_ylabel(r'Cross-validated $R^{2}$', fontsize=fontsize_label)
ax.tick_params(axis='y', which='major', labelsize=fontsize_tick)
ax.set_xticks(np.arange(len(VA))+barWidth/2)
ax.set_xticklabels(['','',''])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.set_ylim(-0.15, 0.65)
plt.legend(fontsize=fontsize_legend, frameon=False, ncol=2)

# save figure
plt.savefig(dir+'/mkFigure/Fig3.svg', format='svg')
plt.savefig(dir+'/mkFigure/Fig3') 
# plt.show()

