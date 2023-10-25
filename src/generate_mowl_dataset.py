import sys

# Add the path to the MOWL library
sys.path.append("../mowl/")

# Import the MOWL library
import mowl
mowl.init_jvm("2g")
from mowl.datasets.build_ontology import insert_annotations, create_from_triples

# Define directory paths
root = "data/"
root_mouse = "data_mouse/"
root_human = "data_human/"
root_go = "data_go/"
root_hp = "data_hp/"
root_uberon = "data_uberon/"
root_union = "data_union/"

# List of species to process
species = ["go"]  # Add more species here if needed, e.g. species = [ "go", "uberon", "union", "mp"]

for sp in species:
    # Determine the appropriate root directory for the current species
    if sp == "mp":
        root_sp = root_mouse
    elif sp == "go":
        root_sp = root_go
    elif sp == "uberon":
        root_sp = root_uberon
    elif sp == "hp":
        root_sp = root_hp
    elif sp == "union":
        root_sp = root_union
    else:
        raise ValueError("Species name not recognized")

    # Define annotations for ontology construction
    annotations = [
        (root + sp + "_annots.tsv", "http://has_annotation", True),
        (root + "disease_annots.tsv", "http://has_annotation", True),
        (root + f"gene_disease_assoc_human.tsv", "http://is_associated_with", False),
        (root + f"gene_disease_assoc_mouse.tsv", "http://is_associated_with", False)
    ]

    # Insert annotations into the ontology
    insert_annotations(root + "upheno_all_with_relations.owl", annotations, root_sp + f"mowl_{sp}.owl")

    # Define additional annotations for PPI (Protein-Protein Interaction) data
    annotations_ppi = [
        (root + sp + "_annots.tsv", "http://has_annotation", True),
        (root + "disease_annots.tsv", "http://has_annotation", True),
        (root + f"gene_disease_assoc_human.tsv", "http://is_associated_with", False),
        (root + f"gene_disease_assoc_mouse.tsv", "http://is_associated_with", False),
        (root + f"ppi_data.tsv", "http://is_interacted_with", False)
    ]

    # Insert PPI annotations into the ontology
    insert_annotations(root + "upheno_all_with_relations.owl", annotations_ppi, root_sp + f"mowl_ppi_{sp}.owl")

    '''
    # Create ontology from triples (optional)
    create_from_triples(
        root_sp + f"valid_assoc_data_{sp}.tsv",
        root_sp + f"valid_{sp}.owl",
        relation_name="is_associated_with",
        bidirectional=False,
    )

    create_from_triples(
        root_sp + f"test_assoc_data_{sp}.tsv",
        root_sp + f"test_{sp}.owl",
        relation_name="is_associated_with",
        bidirectional=False,
    )
    '''






