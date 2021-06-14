# EmbedPVP
Prioritizing causative variants SNP/indls by integrating functional embedding and biological annotations for genes.

## Dataset
We train and evaluate our method using human genomic Single nucleotide variants (SNVs) collected from [clinvar](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/) dataset.

## Dependencies
The code was developed and tested using python 3.7. To install python dependencies run:  
 `pip install -r requirements.txt`

## Prediction the candidate SNPs workflow
We integrate the annotates from Gene ontology [GO](http://geneontology.org/docs/download-go-annotations/), Uber-anatomy ontology
 [UBERON](https://www.ebi.ac.uk/ols/ontologies/uberon), Mammalian Phenotype ontology [MP](http://www.informatics.jax.org/vocab/mp_ontology), and Human Phenotype Ontology [HPO](https://hpo.jax.org/app/download/annotation) using [DL2vec](https://github.com/bio-ontology-research-group/DL2Vec). We convert different types of Description Logic axioms into graph representation, and then generate an embedding for each node and edge type. In addition we collected genomics features using public prediction scores for the SNP. 

## Scripts
- Details for predicting gene-disease associations with DL2Vec can be found in the [experiment](https://github.com/bio-ontology-research-group/DL2Vec/tree/master/Experiment).
- ``download.sh``: This script is used to download the annotions databases.
- ``annotations.sh``: This script is used to annotate the varaints.
- ``data_preprocessing.py``: preprocessing the annotations and features selection. 
- ``DL2vec``: Details for generate DL2vec embeddings.
- ``training.py``: script to train and testing the model, with Hyperparameter optimization

## Running EmbedPVP using pretrained models
1. Download the distribution file in []()
2. Extract the distribution files EmbedPVP and `cd EmbedPVP`
3. Database requirements: Installing Annovar: 
  - To download and install the Annovar command line tool follow the [Annovar installation instructions](https://annovar.openbioinformatics.org/en/latest/user-guide/download/).
3. Download the required database by run:  `bash src/downloadDB.sh`
4. Run the command `embedpvp --help` to display help and parameters:

```
EmbedPVP: Prioritizing Causative Variants by Integrating Functional Embedding and Biological Annotations for Genes.

optional arguments:
  -h, --help            show this help message and exit
  -inputfile [INPUTFILE]
                        Path to VCF file
  -hpo [HPO]            List of phenotype ids separated by commas
  -outfile [OUTFILE]    Path to results file
  -model [MODEL]        Preferred model (go, mp, uberon, hp,
                        go_ppi,mp_ppi,uberon_ppi,hp_ppi) , default='hp'
```

### Example:
    embedPVP -inputfile data/example.vcf -hpo HP:0003701,HP:0001324,HP:0010628,HP:0003388,HP:0000774,HP:0002093,HP:0000508,HP:0000218,HP:0000007  -outfile example_output.txt -model hp 

 ```   
 Annotate VCF file (example.vcf) with the phenotypes (HP:0003701,HP:0001324,HP:0010628,HP:0003388,HP:0000774,HP:0002093,HP:0000508,HP:0000218,HP:0000007)...
 |========                        | 25% Annotated files generated successfully.
 |================                | 50% Phenotype prediction...
 |========================        | 75% SNP Prediction...
 |================================| 100%
The analysis is Done. You can find the priortize list in the output file: example_output.txt 

```
#### Output:
The script will output a ranking a score for the candidate caustive SNP. 


## Note
For any questions or comments please contact azza.althagafi@kaust.edu.sa
