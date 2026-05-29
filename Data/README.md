A signal detection theory approach to quantify contextual effects on perceived signal

Authors: Heshu Yin, Kanghee Kim, Joseph Busch, Vijay Singh
Affiliation: Department of Physics and Astronomy, Haverford College, Haverford, PA, USA
Correspondence: vsingh1@haverford.edu

Updated May 29, 2025 by Vijay Singh
-------------------------------------------------------------------------------

This folder contains the data for each subject. The data is stored using 
pseudonyms for each observer. 

Each folder contains four subfolders named condition_0, condition_1, condition_2,
condition_3, condition_demo. Each of these subfolders contain another subfolder
called Repetiton_1, which contains the data.

The subfolder Repetiton_1 contains four files. 

1. trials_condition_0_rep_1.txt: Contains the parameters used for this condition 
and the location of files on host computer.

2. conditions_condition_0_rep_1.txt: Has the RGB values used in the experiment.

3. trials_condition_0_rep_1_gamma_corrected.txt: Has the gamma corrected RGB values 
used to perform the experiment.

4. subject_apple_condition_0_Repetition_1.txt: Has the subject response data. 
Each row represents one trial. The rows are organized in the order in which 
the trials were performed. The columns represent:
Sort-Index, Expected-Response, Observer-Response, Square-1-RGB, Background-1-RGB, Square-2-RGB, Background-2-RGB

Sort-Index is a pseudo index for the trials. It can be used to sort the data 
according to the standard and comparison square luminance. This column can be ignored.

Expected-Response is the correct expected response.
Observer-Response is observer's response.

The other columns contain the input given to Unity in RGBA format.
