# required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.utils import resample

from modelling_utils_paramInit import paramInit

"""

Author: A. Brands

"""

############################################################################################## ADAPT CODE HERE
##############################################################################################################
##############################################################################################################
##############################################################################################################

# define root directory
file = open('setDir.txt')
dir = file.readline()
dir = dir[:-1]

##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################

# visual areas (labels)
VA = ['V1-V3', 'VOTC', 'LOTC']

# determine confidence interval for plotting
CI                  = 68
CI_low              = 50 - (0.5*CI)
CI_high             = 50 + (0.5*CI)
B_repetitions       = 1000

# import image classes
stim_cat = np.loadtxt(dir+'variables/cond_stim.txt', dtype=str)

sf_labels = ['sf_bodies', 'sf_buildings', 'sf_faces', 'sf_objects', 'sf_scenes', 'sf_scrambled']

# electrode coordinates
electrodes_visuallyResponsive = pd.read_csv(dir+'data_subjects/electrodes_visuallyResponsive_manuallyAssigned.txt', header=0, index_col=0, delimiter=' ')
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
models              = ['DN']
models_labels       = ['DN']

# scaling
scaling             = ['none', 'S']
scaling_lbl         = ['omitted', 'included']

# iterate over electrode and retrieve cross-validated performance for DN models and cbDN (fitted for P and NP trials)
r2_values_medians       = np.zeros((B_repetitions, len(VA), len(models), len(scaling)))

r2_values_avg           = np.zeros((len(VA), len(models), len(scaling)))
r2_values_variance      = np.zeros((2, len(VA), len(models), len(scaling)))

# iterate over electrode and retrieve cross-validated performance for csDN (fitted for all image classes seperately)
count_VA = 0
for key, value in VA_name_idx.items():

    # determine number of electrodes
    n_electrodes = len(value)

    # initiate dataframes
    r2_values_current = np.zeros((n_electrodes, len(models), len(scaling)))

    for i in range(n_electrodes):
    # for i in range(2):

        # retrieve info current electrode
        subject = electrodes_visuallyResponsive.subject[value[i]]
        electrode_name = electrodes_visuallyResponsive.electrode[value[i]]

        # cbdn_fine model (overall r2)
        for idxM, model in enumerate(models):

            for idxS, scal in enumerate(scaling):
                temp = pd.read_csv(dir+'modelFit/visuallyResponsive/' + subject + '_' + electrode_name + '/param_' + model + '_' + scal + '.txt', header=0, delimiter=' ', index_col=0)
                r2_values_current[i, idxM, idxS] = temp.loc[:, 'r2']

    # average values
    for m in range(len(models)):
        for s in range(len(scaling)):
            
            # bootstrapping procedure
            for B in range(B_repetitions):

                # draw random sample
                idx_temp = np.arange(n_electrodes)
                n_samples = len(idx_temp)
                boot = resample(idx_temp, replace=True, n_samples=n_samples)

                # retrieve data
                data_boot = np.zeros(n_samples)
                for idxB, b in enumerate(boot):
                    data_boot[idxB] = r2_values_current[b, m, s]

                # compute median for current bootstrap sample
                r2_values_medians[B, count_VA, m, s] = np.median(data_boot)

            # compute CI
            # print(r2_values_medians[:, count_VA, i, s])
            r2_values_avg[count_VA, m, s] = np.median(r2_values_medians[:, count_VA, m, s])
            r2_values_variance[:, count_VA, m, s] = np.nanpercentile(r2_values_medians[:, count_VA, m, s], [CI_low, CI_high])

    # increment count
    count_VA = count_VA + 1

# save data
np.save(dir + '/data_figures/Fig3/r2_avg', r2_values_avg)
np.save(dir + '/data_figures/Fig3/r2_CI', r2_values_variance)

# plot cross validation
fig = plt.figure(figsize=(7, 5))
ax = plt.gca()

# seperate axes
sns.despine(offset=10)

# set barwidth
barWidth = 0.2

# fontsizes
fontsize_label  = 20
fontsize_tick   = 15
fontsize_legend = 15

# # set height of bar
model1 = r2_values_avg[:, 0, 0]
model2 = r2_values_avg[:, 0, 1]

# Set position of bar on X axis
br1 = np.arange(len(model1))
br2 = [x + barWidth for x in br1]
br = [br1, br2]
print(br)

# visualize mean
ax.bar(br1, model1, color='pink', width = barWidth,
        edgecolor = 'white', label = scaling_lbl[0])
ax.bar(br2, model2, color = np.array([212, 0, 0])/255 , width = barWidth,
        edgecolor = 'white', label = scaling_lbl[1])

# visualize spread
for iV in range(len(VA)):
    for iS in range(len(scaling_lbl)):
        ax.plot([br[iS][iV], br[iS][iV]], [r2_values_variance[0, iV, 0, iS], r2_values_variance[1, iV, 0, iS]], color='black')

# axes
ax.set_ylabel(r'Cross-validated $R^{2}$', fontsize=fontsize_label)
ax.tick_params(axis='y', which='major', labelsize=fontsize_tick)
ax.set_xticks(np.arange(len(VA))+barWidth/2)
ax.set_xticklabels(['','',''])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.set_ylim(-0.15, 0.65)
plt.legend(fontsize=fontsize_legend, frameon=False, loc='upper right', ncol=2, title='Category-dependent scaling', title_fontsize=fontsize_legend)

# save figure
plt.tight_layout()
plt.savefig(dir+'/mkFigure/Fig3.svg', format='svg')
plt.savefig(dir+'/mkFigure/Fig3') 
# plt.show()



