
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
models              = ['DN', 'TTC19', 'TTC17']
models_labels       = ['DN', 'A+S model (Stigliani2019)', 'L+Q (Stigliani2017)']
scaling             = ['none', 'S']
scaling_lbl         = ['omitted', 'included']

global sf

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

    if key == 'V1-V3':
        sf = np.zeros((n_electrodes, len(sf_labels)))

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
                if (model == 'DN') & (key == 'V1-V3') & (scal == 'S'):
                    sf[i, :] = temp.loc[:, sf_labels]

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

# fontsizes
fontsize_label  = 22
fontsize_tick   = 20
fontsize_legend = 17

# # initiate figure
# fig = plt.figure(figsize=(8,5))
# ax = plt.gca()

# # plot sf
# ax.bar(np.arange(len(sf_labels)), np.mean(sf, 0), color='grey', yerr=np.std(sf, 0)/math.sqrt(sf.shape[0]))

# # adjust axes
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xticklabels([' '] + stim_cat.tolist(), rotation=45)
# # ax.set_xlabel('Image categories', fontsize=fontsize_label)
# ax.set_ylabel('Scaling factor')

# # save figure
# plt.tight_layout()
# plt.savefig(dir+'/mkFigure/sf_V1V3.svg', format='svg')
# plt.savefig(dir+'/mkFigure/sf_V1V3') 
# plt.close()
# # plt.show()

# save data
np.save(dir + '/data_figures/SFig1/r2_avg', r2_values_avg)
np.save(dir + '/data_figures/SFig1/r2_CI', r2_values_variance)

# plot cross validation
fig, ax = plt.subplots(1, len(VA), figsize=(12, 6))

# seperate axes
sns.despine(offset=10)

# set barwidth
barWidth = 0.2

color = [np.array([(212,170,0)])/255, np.array([((51,102,153))])/255, np.array([(102,153,51)])/255]

for i in range(len(VA)):

    # # set height of bar
    model1 = r2_values_avg[i, 0, :]
    model2 = r2_values_avg[i, 1, :]
    model3 = r2_values_avg[i, 2, :]

    # Set position of bar on X axis
    br1 = np.arange(len(model1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br = [br1, br2, br3]

    # visualize mean
    ax[i].bar(br1, model1, color=color[0], edgecolor='white', width = barWidth,
            label=models_labels[0])
    ax[i].bar(br2, model2, color=color[1], edgecolor='white', width = barWidth,
            label=models_labels[1])
    ax[i].bar(br3, model3, color=color[2], edgecolor='white', width = barWidth,
            label = models_labels[2])
    
    # spread
    for iS in range(len(scaling)):
        for iM in range(len(models)):
            ax[i].plot([br[iM][iS], br[iM][iS]], [r2_values_variance[0, i, iM, iS], r2_values_variance[1, i, iM, iS]], color='black')

    # axes
    ax[i].tick_params(axis='y', which='major', labelsize=fontsize_tick)
    ax[i].set_xticks(np.arange(len(scaling))+barWidth)
    ax[i].set_xticklabels(scaling_lbl, fontsize=fontsize_tick)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    # ax[i].set_xlim(-0.5, 2.5)
    if i != 0:
        ax[i].set_yticks([])
    else:
        ax[i].set_ylabel(r'Cross-validated $R^{2}$', fontsize=fontsize_label)
    if i == 1:
        ax[i].set_xlabel(r'Category-selective scaling', fontsize=fontsize_label)

    std = 0.2
    ymin = np.min(r2_values_avg) + 0.2*np.min(r2_values_avg)
    ymax = np.max(r2_values_avg) + 0.2*np.max(r2_values_avg)
    ax[i].set_ylim(-0.05, 0.7)

# average (np.zeros((B_repetitions, len(VA), len(models), len(scaling))))
medians_avg = r2_values_medians.mean(1).mean(2)

# add legend
ax[0].legend(fontsize=fontsize_legend, frameon=False, loc='upper left')

# save figure
plt.tight_layout()
plt.savefig(dir+'/mkFigure/SuppFig1.svg', format='svg')
plt.savefig(dir+'/mkFigure/SuppFig1') 
# plt.show()

# ############################################################### STATISTICS avg
# ##############################################################################

# statistical testing
alpha   = 0.05
mc      = len(model)
# mc = 1


# # TTC17 vs. TTC19
# sample1 = medians_avg[:, 0]
# sample2 = medians_avg[:, 1]
# param_diffs = sample1 - sample2

# p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
# if p < (alpha/mc):
#     print('TTC17 vs. TTC19: ', p, ' SIGNIFICANT')
# else:
#     print('TTC17 vs. TTC19: ', p)

# # TTC17 vs. DN
# sample1 = medians_avg[:, 0]
# sample2 = medians_avg[:, 2]
# param_diffs = sample1 - sample2

# p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
# if p < (alpha/3):
#     print('TTC17 vs. DN: ', p, ' SIGNIFICANT')
# else:
#     print('TTC17 vs. DN: ', p)

# # TTC19 vs. DN
# sample1 = medians_avg[:, 1]
# sample2 = medians_avg[:, 2]
# param_diffs = sample1 - sample2

# p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
# if p < (alpha/3):
#     print('TTC19 vs. DN: ', p, ' SIGNIFICANT')
# else:
#     print('TTC19 vs. DN: ', p)

# print(60*'$')

############################################################ STATISTICS per VA
##############################################################################

for iV, va in enumerate(VA):

    print(30*'#')
    print(va)
    print(30*'#')

    # scaling vs. none
    for iM in range(len(models)):
        sample1 = r2_values_medians[:, iV, iM, 0]
        sample2 = r2_values_medians[:, iV, iM, 1]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < (alpha):
            print(models[iM], ', none vs. S: ', p, ' SIGNIFICANT')
        else:
            print(models[iM], ', none vs. S: ', p)

    for iS, s in enumerate(scaling):

        print(30*'-')
        print(s)
        print(30*'-')

        # TTC17 vs. TTC19
        sample1 = r2_values_medians[:, iV, 0, iS]
        sample2 = r2_values_medians[:, iV, 1, iS]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < (alpha/mc):
            print('DN vs. TT19: ', p, ' SIGNIFICANT')
        else:
            print('DNN vs. TT19: ', p)

        # TTC17 vs. DN
        sample1 = r2_values_medians[:, iV, 0, iS]
        sample2 = r2_values_medians[:, iV, 2, iS]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < (alpha/mc):
            print('DN vs. TTC17: ', p, ' SIGNIFICANT')
        else:
            print('DN vs. TTC17: ', p)

        # TTC19 vs. DN
        sample1 = r2_values_medians[:, iV, 1, iS]
        sample2 = r2_values_medians[:, iV, 2, iS]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < (alpha/mc):
            print('TTC19 vs. TTC17: ', p, ' SIGNIFICANT')
        else:
            print('TTC19 vs. TTC17: ', p)