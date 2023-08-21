# required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne.transforms import apply_trans
from mne.io.constants import FIFF


"""

Author: A. Brands

"""

############################################################################################## ADAPT CODE HERE
##############################################################################################################
##############################################################################################################
##############################################################################################################

# define root directory
file = open('setDir.txt')
dir = file.readline().strip('\n')
print(dir)

# define Freesurfer directory
file = open('setDir_FreeSurfer.txt')
dir_surfdrive = file.readline()

##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################


# views to save the plots
view = ['lat', 'ven', 'caudal']

# electrode coordinates
electrodes_visuallyResponsive = pd.read_csv(dir+'subject_data/electrodes_visuallyResponsive_manuallyAssigned.txt', header=0, index_col=0, delimiter=' ')
n_electrodes = len(electrodes_visuallyResponsive)

threshold_d_prime = 1.0
electrodes_categorySelective = pd.read_csv(dir+'subject_data/electrodes_categorySelective_' + str(threshold_d_prime).replace('.', '-') + '.txt', header=0, index_col=0, delimiter=' ')
print('Total number of category-selective electrodes: ', len(electrodes_categorySelective[electrodes_categorySelective.preferred_cat != 'SCRAMBLED']))

# initiate colours
col_none        = 'black'
RGB             = [[233, 167, 0], [48, 64, 141], [187, 38, 102]]
RGB_cat         = ['goldenrod', 'cornflowerblue', 'orchid']
VA              = ['V1-V3', 'VOTC', 'LOTC']

# define users
subject = 'fsaverage'

# initiate brain plot
brain = mne.viz.Brain(subject, hemi='lh', surf='pial', subjects_dir=dir_surfdrive, background='white', view_layout='vertical', alpha=0.5)
s = 0.3
s_small = 0.15

count_VA = np.zeros(3)
count_cat = 0

# plot electrodes
for i in range(n_electrodes):
# for i in range(6):

    # retrieve info current electrode
    subject = electrodes_visuallyResponsive.subject[i]
    electrode_name = electrodes_visuallyResponsive.electrode[i]

    # select coordinates
    xyz = np.zeros(3)
    idx = electrodes_visuallyResponsive[(electrodes_visuallyResponsive.subject == subject) & (electrodes_visuallyResponsive.electrode == electrode_name)].index
    coordinates_temp = electrodes_visuallyResponsive.loc[idx, ['x', 'y', 'z']]
    coordinates_temp.reset_index(inplace=True, drop=True)
    xyz = coordinates_temp.loc[0, ['x', 'y', 'z']]

    # add to count
    # print(electrodes_visuallyResponsive)
    if electrodes_visuallyResponsive.loc[i, 'varea'] == 'V1-V3':
        count_VA[0]+=1
    elif electrodes_visuallyResponsive.loc[i, 'varea'] == 'VOTC':
        count_VA[1]+=1
    elif electrodes_visuallyResponsive.loc[i, 'varea'] == 'LOTC':
        count_VA[2]+=1

    # control category-selectivity
    index_cat = electrodes_categorySelective[(electrodes_categorySelective.subject == subject) & (electrodes_categorySelective.electrode == electrode_name)].index
    cat_sel = False
    if len(index_cat != 0):
        if electrodes_categorySelective.loc[index_cat[0], 'preferred_cat'] != 'SCRAMBLED':
            cat_sel = True
            count_cat+=1

    # define color
    if (electrodes_visuallyResponsive.loc[i, 'varea'] == 'V1-V3'):
        color_temp1 = np.array(RGB[0][:])/255
        if cat_sel:
            brain.add_foci(xyz, hemi='lh', color=color_temp1, scale_factor=s)
        else:
            brain.add_foci(xyz, hemi='lh', color=RGB_cat[0], scale_factor=s)
        
    elif (electrodes_visuallyResponsive.loc[i, 'varea'] == 'VOTC'):
        color_temp2 = np.array(RGB[1][:])/255
        if cat_sel:
            brain.add_foci(xyz, hemi='lh', color=color_temp2, scale_factor=s)
        else:
            brain.add_foci(xyz, hemi='lh', color=RGB_cat[1], scale_factor=s)
        
    elif (electrodes_visuallyResponsive.loc[i, 'varea'] == 'LOTC'):
        color_temp3 = np.array(RGB[2][:])/255
        if (subject == 'sub-p11') & (electrode_name == 'GB034'): # wrong electrode coordinates
            continue
        if cat_sel:
            brain.add_foci(xyz, hemi='lh', color=color_temp3, scale_factor=s)
        else:
            brain.add_foci(xyz, hemi='lh', color=RGB_cat[2], scale_factor=s)
    else:
        brain.add_foci(xyz, hemi='lh', color='black', scale_factor=s_small)


print(count_VA)

for v in view:
    brain.show_view(v, distance=350)
    brain.save_image(dir+'/mkFigure/Fig2_' + v + '_' + str(threshold_d_prime).replace('.', '-') + '.png', mode='rgb')