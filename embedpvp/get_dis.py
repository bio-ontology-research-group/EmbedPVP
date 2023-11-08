from mowl.datasets import PathDataset
from mowl.datasets.base import RemoteDataset, OWLClasses

class GDADataset(PathDataset):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    


    @property
    def ontology(self):
        """Training dataset

        :rtype: org.semanticweb.owlapi.model.OWLOntology
        """
        return self._ontology

    @property
    def validation(self):
        """Validation dataset

        :rtype: org.semanticweb.owlapi.model.OWLOntology
        """
        return self._validation

    @property
    def testing(self):
        """Testing ontology

        :rtype: org.semanticweb.owlapi.model.OWLOntology
        """
        return self._testing

        
    #@property
    def as_str(self):
        """Returns the list of entities as string names."""
        return list(self._name_owlobject.keys())


    def get_evaluation_property(self):
        return "http://is_associated_with"


    #@property
    def evaluation_classes(self):
        """List of classes used for evaluation. Depending on the dataset, this method could \
        return a single :class:`OWLClasses` object \
        (as in :class:`PPIYeastDataset <mowl.datasets.builtin.PPIYeastDataset>`) \
        or a tuple of :class:`OWLClasses` objects \
        (as in :class:`GDAHumanDataset <mowl.datasets.builtin.GDAHumanDataset>`). If not \
        overriden, this method returns the classes in the testing ontology obtained from the \
        OWLAPI method ``getClassesInSignature()`` as a :class:`OWLClasses` object.
        """

        if self._evaluation_classes is None:
            genes = set()
            diseases = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if owl_name[7:].isnumeric():
                    genes.add(owl_cls)
                if "OMIM_" in owl_name:
                    diseases.add(owl_cls)

            genes = OWLClasses(genes)
            diseases = OWLClasses(diseases)
            self._evaluation_classes = (genes, diseases)

        return self._evaluation_classes


    def evaluation_class(self): #_get_evaluation_classes
        """Classes that are used in evaluation
        """
        genes = set()
        diseases = set()
        for owl_cls in self.classes.as_str:
            if owl_cls[20:].isnumeric():        
                genes.add(owl_cls)
            u = owl_cls.split('/')[-1]  
            if u.isnumeric():     
                genes.add(owl_cls)    
            if 'OMIM' in owl_cls:
                diseases.add(owl_cls)

        
        #print('d',list(diseases)[1])
        #print('g',list(genes)[1])

        #print(genes)
        #print(diseases)
        
        return genes, diseases
        
        '''
        """Classes that are used in evaluation
        """
        genes = set()
        diseases = set()
        edit_omim = set()
        for owl_cls in self.classes.as_str:
            #print(owl_cls)
            if 'http://ontology.com/' in owl_cls:
                #print(owl_cls[20:])
                if owl_cls[20:].isnumeric():
                    genes.add(owl_cls)
                else:
                    diseases.add(owl_cls)
            if "OMIM" in owl_cls:
                u = owl_cls.split('OMIM:')[1]                
                r = 'OMIM:'+u
                diseases.add(r)
            if "ORPHA" in owl_cls:
                u = owl_cls.split('ORPHA:')[1]
                r = 'ORPHA:'+u
                diseases.add(r)
                        
        return genes, diseases
        '''

    def evaluation_class2(self): #_get_evaluation_classes
        """Classes that are used in evaluation
        """
        genes = set()
        diseases = set()
        for owl_cls in self.classes.as_str:
            if owl_cls[20:].isnumeric():        
                genes.add(owl_cls)
            u = owl_cls.split('/')[-1]  
            if u.isnumeric():     
                genes.add(owl_cls)    
            if 'OMIM' in owl_cls:
                diseases.add(owl_cls)

        
        #print('d',list(diseases)[1])
        #print('g',list(genes)[1])

        #print(genes)
        #print(diseases)
        
        return genes, diseases
        