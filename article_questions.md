
#### Questions:
1. Hardware issues: How can we reproduce an experiment that ran for 3 weeks with 1900 unit LSTM's on ~24 mil training samples, using 4 Nvidia K80 GPU's? Access to DIKU cluster?

#### Intredasting snippets from article:

```"This suggests the network derived secondary structure 'from scratch', through purely unsupervised means, as a compressive principle for next amino-acid predictions."```

```
"We found other neuron correlations with biophysical parameters including solvent accessibility. Taken together, we conclude the UniRep vector space is semantically rich, and encodes structural, evolutionary and functional information."
```

```
"Together, these results suggests that UniRep approximates a set of fundamental biophysical features shared between all proteins."
```

```
"UniRep's consistently superior performance, despite each proteins's unique biology and measurement assay, suggests UniRep is not only robust, but also must encompass features that underpin the function of all of these proteins."
```

```
"Despite not overlapping in sequence space, training and test points were colocalized in UniRep space suggesting that UniRep discovered commonalities between training and test proteins that effectively transformed the problem of extrapolation into one of interpolation, providing a plausible mechanism for our performance."
```

```
"We additionally envision future work on data-free UniRep variant-effect prediction using sequence likelihood-based scoring."
```

#### Dictionary of words/concepts that are not understood:
- Structural Classification of Proteins (SCOP).
- Homologous sequences.
- Crystallography (crystallographic data).
- Evolutionary similarity.
- Levenshtein distance (which is equivalent to the standard Needleman-Wunsch with equal penalties).
- Protein primary and secondary structure.
- Alpha-helix, Beta-sheet.
- Lac repressor structure.
- Protein stability.
- Spearman correlation, Pearson's r, Welch's two-tailed t-test.
- Phenotype.
- Epistasis.
- Simulated annealing.
- Adaptive Sampling.
- Bayesian Optimization.
- Ab initio structure prediction.
