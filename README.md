# EmbedPVP
Prioritizing genomic variants (SNP/InDls) through neuro-symbolic, knowledge-enhanced learning

## Annotation data sources (integrated in the candidate SNP prediction workflow)
We integrated the annotations from different sources:
- Gene ontology ([GO](http://geneontology.org/docs/download-go-annotations/))
- Mammalian Phenotype ontology ([MP](http://www.informatics.jax.org/vocab/mp_ontology))
- Human Phenotype Ontology ([HPO](https://hpo.jax.org/app/download/annotation))
- Uber-anatomy ontology ([UBERON](https://www.ebi.ac.uk/ols/ontologies/uberon))
- Gene expression in human Celltypes Ontology ([CL](https://www.nature.com/articles/s41586-018-0590-4))

## Dependencies

<div align="right">
<img src="https://raw.githubusercontent.com/bio-ontology-research-group/mowl/main/docs/source/mowl_black_background_colors_2048x2048px.png" alt="mOWL library" align="right" width="150">
</div>


- The code was developed and tested using python 3.7. To install python dependencies run:  
 `pip install -r requirements.txt`

- We used ([mOWL](https://github.com/bio-ontology-research-group/mowl)) library to process the input dataset as well as generated the embedding representation using different 
embedding-based methods.


## Note
For any questions or comments please contact azza.althagafi@kaust.edu.sa
