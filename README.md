## What remains
- [ ] mLSTM Visualize PCA (figure 2 a).
- [ ] mLSTM Visualize t-SNE (figure 2 b).
- [ ] mLSTM Visualize t-SNE. (figure 2 c).
- [ ] TAPE table results
- [ ] Write conclusion
- [ ] Write discussion
- [ ] Write results section: amino acids embedding
- [ ] Write results section: proteomes
- [ ] Write results section: SCOPe
- [ ] Write results section: TAPE ablation
- [ ] Write conclusion section
- [ ] review background section
- [ ] review unirep model section
- [ ] review reproduction section
- [ ] review conclusion
- [ ] .. complete this list

# unirep_project
Repository for project out of course scope on Unirep article.

## NOTER fra 2019-12-19
deep sequence. VAE. kig på en proteinfammilie ad gangen. alignment. find den bedste alignment. 
```
AHF--GH
-HFGWGH
```
kolonner skal være ens. identiske amino-syrer. man vil gerne have at positioner er ens, strukturelt.

hvis man kigger på en proteinfamilier og aligner. tag alignment, lær en model (VAE), i modsætning til os, absolutte positioner. det er absolut fordi vi har alignet. artikel fra 2017, deep generative models of genetic variation capture the effects of mutations. local vs global info.


## Last christmas questions for Wouter
- We are considering scaling down on data from ~37 mil to ~22 mil. What do you think? Should we do ~1 mil proteins instead?
- Large-scale data processing; hardware GPU vs. CPU; too much data vs. too little data; cluster and parallelism. Model parameter explosion, it cannot fit in GPU or memory.
- Thesis contract status.
- Cluster: How greedy can we be during the christmas break?
- Novo mentors.
- what figures should we include? RNN? UniRep? More specific figures? Results?
- rosetta negative total energy?
- buried non-polar surface area???


## UniRep Sources
- [The Nature Methods article.](https://www.nature.com/articles/s41592-019-0598-1)
- [Code for UniRep model training and inference with trained weights + links to data.](https://github.com/churchlab/UniRep)
- [Code to reproduce all analysis and regenerate figures.](https://github.com/churchlab/UniRep-analysis)
- [UNIVERSAL TRANSFORMING GEOMETRIC NETWORK](https://arxiv.org/pdf/1908.00723.pdf). The recurrent geometric network (RGN), the first end-to-end differentiable neural architecture for protein structure prediction, is a competitive alternative to existing models. However, the RGN’s use of recurrent neural networks (RNNs) as internal representations results in long training time and unstable gradients. And because of its sequential nature, it is less effective at learning global dependencies among amino acids than existing transformer architectures. We propose the Universal Transforming Geometric Network (UTGN), an end-to-end differentiable model that uses the encoder portion of the Universal Transformer architecture as an alternative for internal representations.
