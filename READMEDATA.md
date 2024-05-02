# Temporal dynamics of short-term adaptation across human visual cortex

This folder contains the values derived from data analysis and computational modelling. See below for more details.

------------
subfolder: **data_figures**

Contains the data used for plotting the metrcis:
* Fig. 3 (cross-validated $R^{2}$ values)
* Fig. 4 (time-to-peak and full-width-half-maximum)
* Fig. 6 (averge adaptation and long-term adaptation determined by the intercept)
* Fig. 7 (fitted parameter values)
* Fig. 8 (averge adaptation and long-term adaptation determined by the intercept)
* Supp. Fig. 1 (cross-validated $R^{2}$ values)
* Supp. Fig. 2 (time-to-peak and full-width-half-maximum)
* Supp. Fig. 4 (averge adaptation and long-term adaptation determined by the intercept)
* Supp. Fig. 5 (averge adaptation and long-term adaptation determined by the intercept)
* Supp. Fig. 7 (averge adaptation and long-term adaptation determined by the intercept)
* Supp. Fig. 8 (averge adaptation and long-term adaptation determined by the intercept)

Both the bootstrapped medians and confidence intervals are stored.

------------
folder: **data_subjects**

Contains broadband data and information regarding the electrode selection.

The files ```electrodes_categorySelective_0-5.txt```, ```electrodes_categorySelective_0-75.txt```, ```electrodes_categorySelective_1-0.txt``` contain the category-selective electrodes selected based on a threshold of $d'$ of 0.5, 0.75 and 1, respectively, and are used to derive the data in Fig. 8.

The file ```electrodes_visuallyResponsive_manuallyAssigend.txt``` contains the area-wise electrode selection and is used to derive the data in Fig. 4, 5 and 6.

Each of the folders ```sub-p11```, ```sub-p12```, ```sub-p13``` and ```sub-p14``` contain the following files:
* ```channels.txt```: contains information about each electrode (e.g. type of electrode and sampling frequency)
* ```coordinates.txt```: xyz-coordinates per electrode
* ```electrodes.txt```: contains information (e.g. xyz-coordinates, visual area)
* ```events.txt```: trial information
* ```excluded_epochs.txt```: epochs excluded during preprocessing
* ```t.txt```: timestamps
* ```epochs_b```: channel-wise broadband data (.txt) (raw and baseline corrected)

------------
folder: **model_fit**

Model fit for the electrodes used to study adaptation across areas (folder ```visaullyResponsive```) or across stimuli within category-selective areas (folder ```categorySelective```). Model fitting was performed on the level of the individual electrodes. For each electrode the following files were created after fitting:
* ```r_sq```: $R^{2}$ value per fold and averaged over all folds.
* ```param```: parameter values after fitting the model
