conda activate mutagenesis ; \
python tcr_specific_classification.py --epitope SIINFEKL --activation AS --threshold 46.9 ; \
python cross-classification-educated.py --epitope SIINFEKL --activation AS --threshold 46.9 ; \
python tcr_specific_data_size.py --epitope SIINFEKL --activation AS --threshold 46.9 ; \
python tcr_stratified_classification.py --epitope SIINFEKL --activation AS --threshold 46.9 ; \
python tcr_stratified_lpo.py --epitope SIINFEKL --activation AS --threshold 46.9 ; \
python tcr_stratified_regression.py --epitope SIINFEKL --activation AS ; \
python permutation_feature_importance_regression.py --epitope SIINFEKL --activation AS --threshold 46.9 ; \

python tcr_specific_classification.py --epitope  VPSVWRSSL --activation pc --threshold 66.09 ; \
python tcr_specific_data_size.py --epitope  VPSVWRSSL --activation pc --threshold 66.09 ; \
python tcr_stratified_classification.py --epitope  VPSVWRSSL --activation pc --threshold 66.09 ; \
python tcr_stratified_lpo.py --epitope  VPSVWRSSL --activation pc --threshold 66.09 ; \
python tcr_stratified_regression.py --epitope  VPSVWRSSL --activation pc  ; \
python permutation_feature_importance_regression.py --epitope  VPSVWRSSL --activation pc --threshold 66.09
