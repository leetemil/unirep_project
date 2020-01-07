mkdir -p ./data

# Download pfam
wget http://s3.amazonaws.com/proteindata/data_pytorch/pfam.tar.gz; tar -xzf pfam.tar.gz -C ./data; rm pfam.tar.gz; break;;

# Download Vocab/Model files
wget http://s3.amazonaws.com/proteindata/data_pytorch/pfam.model
wget http://s3.amazonaws.com/proteindata/data_pytorch/pfam.vocab

mv pfam.model data
mv pfam.vocab data

# Download Data Files
wget http://s3.amazonaws.com/proteindata/data_pytorch/secondary_structure.tar.gz
wget http://s3.amazonaws.com/proteindata/data_pytorch/proteinnet.tar.gz
wget http://s3.amazonaws.com/proteindata/data_pytorch/remote_homology.tar.gz
wget http://s3.amazonaws.com/proteindata/data_pytorch/fluorescence.tar.gz
wget http://s3.amazonaws.com/proteindata/data_pytorch/stability.tar.gz

tar -xzf secondary_structure.tar.gz -C ./data
tar -xzf proteinnet.tar.gz -C ./data
tar -xzf remote_homology.tar.gz -C ./data
tar -xzf fluorescence.tar.gz -C ./data
tar -xzf stability.tar.gz -C ./data

rm secondary_structure.tar.gz
rm proteinnet.tar.gz
rm remote_homology.tar.gz
rm fluorescence.tar.gz
rm stability.tar.gz
