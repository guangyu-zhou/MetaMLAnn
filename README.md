# metaSUB
This repository includes the code of MetaMLAnn (see paper).

##File discription:
==================
0. MetaMLAnn folder contains the core code for our model. 
1. naive_model.py: Evaluate baseline models. It will print out the result for 4 baseline models under K fold. you can change it. 
parameters: 
'--KFold' Number of Fold.
'--embedding' 0 to use stations feature only; 1 to use embedding; 2 to use both (default)

2. feature_extraction.py: Functions to extract embedding features/ subway line features

3. gen_data.py: construct species matrix (label) from files or by loading from src/tmp/pic_spec_vec.data

4. get_s.py, plot*.py: Plot and relevant functions on GoogleMap

5. vectorize.py: exploration function for species distribuion similarity v.s. distance. Generate the matrix. Similar as part of function in the gen_data.py. 

==================
## Reference