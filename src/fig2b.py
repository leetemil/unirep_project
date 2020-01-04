import torch
from Bio import SeqIO

from unirep import UniRep
from torch import nn
from pathlib import Path
import numpy as np

from datahandling import ProteinDataset, getProteinDataLoader, idx2seq, seq2idx
# from constants import *
from unirep import UniRep
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

# Options
MLSTM = False
EMBED_SIZE = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 4

# Get hardware information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
MULTI_GPU = NUM_GPUS > 1
print(f"Found {NUM_GPUS} GPUs!")
print("CUDNN version:", torch.backends.cudnn.version())

# Define model
model = UniRep(EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, use_mlstm=MLSTM)

# Use DataParallel if more than 1 GPU!
if MULTI_GPU:
    model = nn.DataParallel(model)

model.to(device)

# Apply weight norm on LSTM
if not MLSTM:
    if MULTI_GPU:
        inner_model = model.module
    else:
        inner_model = model

    for i in range(inner_model.num_layers):
        nn.utils.weight_norm(inner_model.rnn, f"weight_ih_l{i}")
        nn.utils.weight_norm(inner_model.rnn, f"weight_hh_l{i}")
        nn.utils.weight_norm(inner_model.rnn, f"bias_ih_l{i}")
        nn.utils.weight_norm(inner_model.rnn, f"bias_hh_l{i}")

# Load saved model
d = torch.load('../data/models/model_LSTM_notrunc.torch', map_location=torch.device('cpu'))
modulestate = d['model_state_dict']

if True:
    # remove module from multi gpu dict
    from collections import OrderedDict
    od = OrderedDict()
    for key, value in modulestate.items():
        od[key[7:]] = value

    modulestate = od

model.load_state_dict(modulestate)

# Load proteomes
proteome_path = Path('../data/proteomes')
proteomes = [(f.stem, ProteinDataset(f, device)) for f in proteome_path.iterdir()]

BATCH_SIZE = 1024

representations = []

model.eval()

with torch.no_grad():
    m = len(proteomes)
    for k, (name, proteome) in enumerate(proteomes):

        representation = torch.zeros(HIDDEN_SIZE)

        n = len(list(proteome))
        print(f'{k + 1} of {m}: Getting representations for {name}. There are {n} proteins.')
        data = getProteinDataLoader(proteome, batch_size=BATCH_SIZE)
        for i, xb in enumerate(data):
            representation += model.get_representation(xb, device).sum(dim=0)
            print(f'{(i + 1) * BATCH_SIZE * 100 / n:7.2f}%', end='\r')

        print('')
        representations += [(name, representation.numpy() / n)]


PROTEOMEFILE = 'proteome_representations.pkl'

with open(PROTEOMEFILE, 'wb') as f:
    pickle.dump(representations, f)

# # load pickle:
# with open(PROTEOMEFILE, 'rb') as f:
#     representations = pickle.load(f)

print(representations)

VIRUS = 'virus'
ANIMAL = 'animalia'
MAMMAL = 'mammalia'
FUNGI = 'fungi'
PLANT = 'plantae'
EUKARYA = 'eukarya'
ARCHAEA = 'archaea'
CYANOBACTERIA = 'cyanobacteria'
BACTERIA = 'bacteria'

colorscheme = {
    MAMMAL : 'aqua',
    ANIMAL : 'darkblue',
    PLANT : 'darkgreen',
    FUNGI : 'khaki',
    EUKARYA : 'tan',
    ARCHAEA : 'purple',
    CYANOBACTERIA : 'torquoise',
    BACTERIA : 'salmon',
    VIRUS : 'maroon'
}

organisms = {
    'halobacterium_salinarum' : ARCHAEA,
    'haloferax_volcanii' : ARCHAEA,
    'methanococcus_maripaludis' : ARCHAEA,
    'methanosarcina_acetivorans' : ARCHAEA,
    'sulfolobus_solfataricus' : ARCHAEA,
    'thermococcus_kodakarensis' : ARCHAEA,
    'escherichia_coli' : BACTERIA,
    'aliivibrio_fischeri' : BACTERIA,
    'azotobacter_vinelandii' : BACTERIA,
    'bacillus_subtilis' : BACTERIA,
    'cyanothece' : CYANOBACTERIA,
    'mycoplasma genitalium' : BACTERIA,
    'mycobacterium tuberculosis' : BACTERIA,
    'prochlorococcus marinus' : CYANOBACTERIA,
    'streptomyces coelicolor' : BACTERIA,
    'synechocystis' : CYANOBACTERIA,
    'caenorhabditis_elegans' : ANIMAL,
    'drosophila_melanogaster' : ANIMAL,
    'homo_sapiens' : MAMMAL,
    'mus_musculus' : MAMMAL,
    'saccharomyces_cerevisiae' : FUNGI,
    'anolis_carolinensis' : ANIMAL,
    'aspergillus_nidulans' : FUNGI,
    'arabidopsis_thaliana' : PLANT,
    'cavia_porcellus' : MAMMAL,
    'gallus_gallus domesticus' : ANIMAL,
    'coprinopsis_cinerea' : FUNGI,
    'bos_taurus' : MAMMAL,
    'chlamydomonas_reinhardtii' : EUKARYA,
    'cryptococcus' : FUNGI,
    'canis_familiaris' : MAMMAL,
    'emiliania_huxleyi' : EUKARYA,
    'macaca_mulatta' : MAMMAL,
    'zea_mays' : PLANT,
    'heterocephalus_glaber' : MAMMAL,
    'neurospora_crassa' : FUNGI,
    'oryzias_latipes' : ANIMAL,
    'physcomitrella_patens' : PLANT,
    'columba_liva' : ANIMAL,
    'sus_scrofa' : MAMMAL,
    'pristionchus_pacificus' : ANIMAL,
    'oryza_sativa' : PLANT,
    'schizosaccharomyces_pombe' : FUNGI,
    'tetrahymena_thermophila' : EUKARYA,
    'thalassiosira_pseudonana' : EUKARYA,
    'ustilago_maydis' : FUNGI,
    'xenopus_tropicalis' : ANIMAL,
    'danio_rerio' : ANIMAL,
    'phage_lambda' : VIRUS,
    'sv40' : VIRUS,
    't4_phage' : VIRUS,
    't7_phage' : VIRUS,
    'vaccina_virus' : VIRUS
}

breakpoint()

# Plot
tsne = TSNE(n_components=2)
names, reps = list(zip(*representations))
c = [colorscheme[organisms[name]] for name in names]
prots = tsne.fit_transform(reps)
plt.grid()
plt.title('Proteome t-SNE')
plt.scatter(prots[:, 0], prots[:, 1], c=c, s=50)

plt.show()
