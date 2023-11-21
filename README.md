# P-TEAM - Predicting T Cell Receptor (TCR) Functionality against Mutant Epitopes
![Graphical Abstract](https://github.com/SchubertLab/TcrPrediction_MutatedAPLs/blob/master/figures/manuscript_fig1_abstract-1.jpg)

P-TEAM is a Random Forest-based prediction model to estimate the effect of mutations in epitopes on the TCR activation. Depending on the provided label, the model can classify TCR activation binary, or predict a continuous binding score.
While we observed that a total of 25% (ca. n=38) of randomly sampled mutations are sufficient for high-performance prediction, the number of samples can further be reduced to 24 mutations by employing iterative experimental design and our active learning framework

## Installation
TBD - Under construction !!!

To recreate the prediction results of the paper:
```
git clone https://github.com/SchubertLab/TcrPrediction_MutatedAPLs.git
cd TCRPrediction_MutatedAPLs
conda create --name pteam --file=requirements.yml
sh ./activation-prediction/run_all.sh
```
For the active learning results, run `activation-prediction/active_learning/al_tcr_specific.ipynb` for both epitopes.

The positional distances were calculated within the [pymol software](https://pymol.org/2/) via the script `Modelling3D/PositonalDistances.py` on the structural models provided in the Supplementary Data 6-7 of the paper.

To recreate the baseline results, refer to the notebook `baseline/SOTA_Comparisson.ipynb` and the GitHub repositories of [ERGO-II](https://github.com/IdoSpringer/ERGO-II) link and [ImRex](https://github.com/pmoris/ImRex) link.

## Tutorials
TBD - Under construction !!!

Tutorials are provided for:
- novel predictions for a specific TCR (`tutorials/within_tcr.ipynb`)
- a full mutation profile of a novel TCR (`tutorials/across_tcr.ipynb`)
- the active learning framework within a TCR (`tutorials/active_learning.ipynb`)

## Citation
If P-TEAM is helpful in your research, please consider citing the following paper:

```
@article{dorigatti2023predicting,
  title={Predicting T Cell Receptor Functionality against Mutant Epitopes},
  author={Dorigatti, Emilio and Drost, Felix and Straub, Adrian and Hilgendorf, Philipp and Wagner, Karolin Isabel and Bischl, Bernd and Busch, Dirk and Schober, Kilian and Schubert, Benjamin},
  journal={bioRxiv},
  pages={2023--05},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```
