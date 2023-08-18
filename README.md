# Temporal dynamics of neural adaptation across human visual cortex
This repository is included in the following paper:

**Temporal dynamics of neural adaptation across human visual cortex (2023). A. M. Brands et al, bioRxiv**

and contains data analysis of intracranial electroencephalography (iEEG) data with the goal to investigate how neural responses adapt to prolonged and repeated stimulus presentations across the human visual cortex. In addition, to investigate the neural computations underlying adaptation, responses are predicted using a model of delayed divisive normalization (DN).

The neural data that is required to create the figures will be made available later.

Structure
------------
The main directory contains scripts used for preprocessing (```analysis_*.py```), including electrode and epochs, and baseline correction. Scripts used for fitting the computational models are denoted with ```modelling_*```  and the scripts used to create the figures as presented in the paper are denoted with ```mkFigure*.py```. The ```utils.py``` contains helper functions. In addition, the repository contains the following directories:

* **mkFigure**: contains the figures as presented in the paper (in *.png* and *.svg*) and figures (subfolder *modelFit*) showing the modelFit including example timecourses derived from the iEEG data and DN model.

* **models**: Implementation of the DN model receiving a stimulus timecourse and subsequently predicts the neural response. One model implementation was previously described in Zhou et al (2019) en Groen et al. (2022) (```DN.py```). In addition, an augmented version of the model (```csDN.py```) scales the stimulus course depending on the image category the stimulus belongs to (e.g. face, scene) to account for category-selectivity. A third implementation scales the stimulus according to image category, but omits the general scaling (```csDN_withoutGeneralScaling.py```).

* **variables**: files containing the information about the experimental settings. 

## Installation
Clone the git repository to create a local copy with the following command:

    $ git clone git@github.com:ABra1993/tAdaptation_ECoG.git

and set the directory where the folder is located in ```setDir.txt``` (e.g. *home/user/tAdaptation_ECoG/*)

References
------------
Groen, I. I., Piantoni, G., Montenegro, S., Flinker, A., Devore, S., Devinsky, O., ... & Winawer, J. (2022). Temporal dynamics of neural responses in human visual cortex. Journal of Neuroscience, 42(40), 7562-7580.

Zhou, J., Benson, N. C., Kay, K., & Winawer, J. (2019). Predicting neuronal dynamics with a delayed gain control model. PLoS computational biology, 15(11), e1007484.
