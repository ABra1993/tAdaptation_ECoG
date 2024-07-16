# Temporal dynamics of short-term neural adaptation across human visual cortex
This repository is included in the following paper:

**Brands, A. M., Devore, S., Devinsky, O., Doyle, W., Flinker, A., Friedman, D., ... & Groen, I. I. A. (2024). Temporal dynamics of short-term neural adaptation across human visual cortex. PLOS Computational Biology, 20(5), e1012161.**

and contains data analysis of intracranial electroencephalography (iEEG) data with the goal to investigate how neural responses adapt to prolonged and repeated stimulus presentations across the human visual cortex. In addition, to investigate the neural computations underlying adaptation, responses are predicted using a model of delayed divisive normalization (DN).

All code used for the purpose of this paper can be found at this GitHub repository and https://github.com/WinawerLab/ECoG_utils. Raw and processed iEEG data are publicly available as part of the ‘Visual ECoG dataset’ on OpenNeuro ( https://openneuro.org/datasets/ds004194).

Structure
------------
The main directory contains scripts used for preprocessing (```analysis_*.py```), including electrode and epochs, and baseline correction. Scripts used for fitting the computational models are denoted with ```modelling_*```  and the scripts used to create the figures as presented in the paper are denoted with ```mkFigure*.py```. The ```utils.py``` contains helper functions. In addition, the repository contains the following directories:

* **mkFigure**: contains the figures as presented in the paper (in *.png* and *.svg*) and figures (subfolder *modelFit*) showing the modelFit including example timecourses derived from the iEEG data and DN model.

* **models**: Implementation of the DN model receiving a stimulus timecourse and subsequently predicts the neural response. One model implementation was previously described in Zhou et al (2019) en Groen et al. (2022) (```DN.py```). In addition, an augmented version of the model (```csDN.py```) scales the stimulus course depending on the image category the stimulus belongs to (e.g. face, scene) to account for category-selectivity. A third implementation scales the stimulus according to image category, but omits the general scaling (```csDN_withoutGeneralScaling.py```, not included in the paper).

* **variables**: files containing the information about the experimental settings. 

## Installation
Clone the git repository to create a local copy with the following command:

    $ git clone git@github.com:ABra1993/tAdaptation_ECoG.git

and set the directory where the GitHub repository is located by

     echo -e DIR > setDir.txt

where DIR is the directory's location (e.g. ```DIR = home/Users/tAdaptation_ECoG/```). In addition, to create Figure 2 (```mkFigure2.py```) you also need acces to the FreeSurfer files which is done by:

     echo -e DIR > setDir_FreeSurfer.txt

where DIR is the FreeSurfer directory (e.g. ```DIR = home/Users/derivatives/freesurfer/```)

References
------------
Groen, I. I., Piantoni, G., Montenegro, S., Flinker, A., Devore, S., Devinsky, O., ... & Winawer, J. (2022). Temporal dynamics of neural responses in human visual cortex. Journal of Neuroscience, 42(40), 7562-7580.

Zhou, J., Benson, N. C., Kay, K., & Winawer, J. (2019). Predicting neuronal dynamics with a delayed gain control model. PLoS computational biology, 15(11), e1007484.
