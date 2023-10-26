# EmbedPVP Scripts

- Details for predicting gene-disease associations with DL2Vec can be found in the [Experiment](https://github.com/bio-ontology-research-group/DL2Vec/tree/master/Experiment).

- `download.sh`: 
This script is used to download the annotation databases from [Annovar](https://annovar.openbioinformatics.org/en/latest/) database.

- `af_annotations.sh`:
This script is used to annotate the variants with allele frequencies from different databases using [Annovar](https://annovar.openbioinformatics.org/en/latest/).

- `annotation_dataset.py`: 
Prepare the annotation dataset from different ontologies ([GO](http://geneontology.org/docs/download-go-annotations/), [MP](http://www.informatics.jax.org/vocab/mp_ontology), [HPO](https://hpo.jax.org/app/download/annotation), and [UBERON](https://www.ebi.ac.uk/ols/ontologies/uberon)

- `generate_mowl_dataset.py`: 
Script to prepare the input ontology to run [mOWL](https://github.com/bio-ontology-research-group/mowl) library and add annotations to the [uPheno](https://zenodo.org/records/3710690) ontology.

- `graph_based_embed.py, semantic_embed.py, and translation_embed.py`: 
Sripts for parameters optimization using embedding methods.

- `embedpvp_graph_based.py, embedpvp_semantic.py, and embedpvp_trans_method.py`: 
Scripts for generating the embeddings for all the genes using the optimized parameters.

- `embedpvp_preprocessing.py`: 
Script for preprocessing data related to annotations and features.

- `embedpvp_prediction.py`: 
Script for training and testing the model, including hyperparameter optimization.

