# EmbedPVP Scripts

- Details for predicting gene-disease associations with DL2Vec can be found in the [Experiment](https://github.com/bio-ontology-research-group/DL2Vec/tree/master/Experiment).

- `download.sh`: This script is used to download the annotation databases from [Annovar](https://annovar.openbioinformatics.org/en/latest/) database.

- `af_annotations.sh`: This script is used to annotate the variants with allele frequencies from different databases using [Annovar](https://annovar.openbioinformatics.org/en/latest/).

- `annotation_dataset.py`: Prepare the annotation dataset from different ontologies ([GO](http://geneontology.org/docs/download-go-annotations/), [MP](http://www.informatics.jax.org/vocab/mp_ontology), [HPO](https://hpo.jax.org/app/download/annotation), [UBERON](https://www.ebi.ac.uk/ols/ontologies/uberon), and [CL](https://www.nature.com/articles/s41586-018-0590-4)).

- `generate_mowl_dataset.py`: Script to prepare the input ontology to run [mOWL](https://github.com/bio-ontology-research-group/mowl) library and add annotations to the ontology.

- `graph_based_embed.py`: script for parameters optimization using graph-based embedding methods.

- `semantic_embed.py`: script for parameters optimization using graph-based embedding methods.

- `translation_embed.py`: script for parameters optimization using translation embedding methods.

- `EmbedPVP_embeddings.py`: Script for preprocessing dataset and the embeddings.

- `EmbedPVP_preprocessing.py`: Script for preprocessing data related to annotations and features.

- `training.py`: Script for training and testing the model, including hyperparameter optimization.

