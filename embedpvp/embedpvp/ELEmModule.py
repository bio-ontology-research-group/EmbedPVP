import mowl
mowl.init_jvm("10g")
from mowl.base_models.elmodel import EmbeddingELModel
from mowl.models.elembeddings.evaluate import ELEmbeddingsPPIEvaluator
from mowl.nn import ELModule
import numpy as np
import torch as th
from torch import nn
#import mowl.models.elboxembeddings.losses as L
import math
import logging
from mowl.models.elboxembeddings.evaluate import ELBoxEmbeddingsPPIEvaluator
from mowl.projection import TaxonomyWithRelationsProjector
from tqdm import trange, tqdm
from mowl.projection.factory import projector_factory


class ELBoxModule(ELModule):

    def __init__(self, nb_ont_classes, nb_rels, embed_dim=50, margin=0.1):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels

        self.embed_dim = embed_dim

        self.class_embed = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_embed.weight, a=-1, b=1)

        weight_data = th.linalg.norm(self.class_embed.weight.data, axis=1).reshape(-1, 1)
        self.class_embed.weight.data /= weight_data

        self.class_offset = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_offset.weight, a=-1, b=1)
        weight_data = th.linalg.norm(self.class_offset.weight.data, axis=1).reshape(-1, 1)
        self.class_offset.weight.data /= weight_data

        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, a=-1, b=1)
        weight_data = th.linalg.norm(self.rel_embed.weight.data, axis=1).reshape(-1, 1)
        self.rel_embed.weight.data /= weight_data

        self.margin = margin

    def gci0_loss(self, data, neg=False):
        c = self.class_embed(data[:, 0])
        d = self.class_embed(data[:, 1])

        off_c = th.abs(self.class_offset(data[:, 0]))
        off_d = th.abs(self.class_offset(data[:, 1]))

        euc = th.abs(c - d)
        dst = th.reshape(th.linalg.norm(th.relu(euc + off_c - off_d + self.margin), axis=1),
                         [-1, 1])

        return dst

    def gci1_loss(self, data, neg=False):
        c = self.class_embed(data[:, 0])
        d = self.class_embed(data[:, 1])
        e = self.class_embed(data[:, 2])
        off_c = th.abs(self.class_offset(data[:, 0]))
        off_d = th.abs(self.class_offset(data[:, 1]))
        off_e = th.abs(self.class_offset(data[:, 2]))

        startAll = th.maximum(c - off_c, d - off_d)
        endAll = th.minimum(c + off_c, d + off_d)

        new_offset = th.abs(startAll - endAll) / 2

        cen1 = (startAll + endAll) / 2
        euc = th.abs(cen1 - e)

        dst = th.reshape(th.linalg.norm(th.relu(euc + new_offset - off_e + self.margin), axis=1),
                         [-1, 1]) + th.linalg.norm(th.relu(startAll - endAll), axis=1)
        return dst

    def gci1_bot_loss(self, data, neg=False):
        c = self.class_embed(data[:, 0])
        d = self.class_embed(data[:, 1])

        off_c = th.abs(self.class_offset(data[:, 0]))
        off_d = th.abs(self.class_offset(data[:, 1]))

        euc = th.abs(c - d)
        dst = th.reshape(th.linalg.norm(th.relu(-euc + off_c + off_d + self.margin), axis=1),
                         [-1, 1])
        return dst

    def gci2_loss(self, data, neg=False):
        if neg:
            return self.gci2_loss_neg(data)
        else:
            c = self.class_embed(data[:, 0])
            r = self.rel_embed(data[:, 1])
            d = self.class_embed(data[:, 2])

            off_c = th.abs(self.class_offset(data[:, 0]))
            off_d = th.abs(self.class_offset(data[:, 2]))

            euc = th.abs(c + r - d)
            dst = th.reshape(th.linalg.norm(th.relu(euc + off_c - off_d + self.margin), axis=1),
                             [-1, 1])
            return dst

    def gci2_loss_neg(self, data):
        c = self.class_embed(data[:, 0])
        r = self.rel_embed(data[:, 1])

        rand_index = np.random.choice(self.class_embed.weight.shape[0], size=len(data))
        rand_index = th.tensor(rand_index).to(self.class_embed.weight.device)
        d = self.class_embed(rand_index)

        off_c = th.abs(self.class_offset(data[:, 0]))
        off_d = th.abs(self.class_offset(rand_index))

        euc = th.abs(c + r - d)
        dst = th.reshape(th.linalg.norm(th.relu(euc - off_c - off_d - self.margin), axis=1),
                         [-1, 1])
        return dst

    def gci3_loss(self, data, neg=False):
        r = self.rel_embed(data[:, 0])
        c = self.class_embed(data[:, 1])
        d = self.class_embed(data[:, 2])

        off_c = th.abs(self.class_offset(data[:, 1]))
        off_d = th.abs(self.class_offset(data[:, 2]))

        euc = th.abs(c - r - d)
        dst = th.reshape(th.linalg.norm(th.relu(euc - off_c - off_d + self.margin), axis=1),
                         [-1, 1])
        return dst


class ELBoxEmbeddings(EmbeddingELModel):

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0,
                 reg_norm=1,
                 learning_rate=0.001,
                 epochs=1000,
                 batch_size=4096 * 8,
                 model_filepath=None,
                 device='cpu'
                 ):
        super().__init__(dataset, batch_size=batch_size,embed_dim=embed_dim,  extended=True, model_filepath=model_filepath)

        self.embed_dim = embed_dim
        self.margin = margin
        self.reg_norm = reg_norm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self._loaded = False
        self._loaded_eval = False
        self.extended = False
        self.init_model()
        self._testing_set = None
        self._training_set = None
        self._head_entities = None
        self._tail_entities = None

    @property
    def testing_set(self):
        if self._testing_set is None:
            projector = TaxonomyWithRelationsProjector(taxonomy=False,
                                                       relations=["http://is_associated_with"]
                                                       )
            #projector = projector_factory('dl2vec', taxonomy=False, relations=["http://is_associated_with"])

            self._testing_set = projector.project(self.dataset.testing)
        return self._testing_set

    @property
    def training_set(self):
        if self._training_set is None:
            projector = TaxonomyWithRelationsProjector(taxonomy=False,
                                                       relations=["http://is_associated_with"]
                                                       )
            #projector = projector_factory('dl2vec', taxonomy=False, relations=["http://is_associated_with"])

            self._training_set = projector.project(self.dataset.ontology)
        return self._training_set

    @property
    def head_entities(self):
        if self._head_entities is None:
            #self._head_entities,_ = self.dataset.evaluation_classes
            self._head_entities,_ = self.dataset.evaluation_class2()
        return self._head_entities #.as_str

    @property
    def tail_entities(self):
        if self._tail_entities is None:
            #_ , self._tail_entities = self.dataset.evaluation_classes
            _ , self._tail_entities = self.dataset.evaluation_class2()
        return self._tail_entities #.as_str

    def init_model(self):
        self.model = ELBoxModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            embed_dim=self.embed_dim,
            margin=self.margin
        ).to(self.device)

    def load_best_model(self):
        self.model.load_state_dict(th.load(self.model_filepath))


    def train(self):
        criterion = nn.MSELoss()
        optimizer = th.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_loss = float('inf')
        _, disease = self.dataset.evaluation_class()
        
        training_datasets = {k: v.data for k, v in
                             self.training_datasets.items()}
        validation_dataset = self.validation_datasets["gci2"][:]

        for epoch in trange(self.epochs):
            self.model.train()

            train_loss = 0
            loss = 0
            for gci_name, gci_dataset in training_datasets.items():
                if len(gci_dataset) == 0:
                    continue
                rand_index = np.random.choice(len(gci_dataset), size=512)
                dst = self.model(gci_dataset[rand_index], gci_name)
                mse_loss = criterion(dst, th.zeros(dst.shape, requires_grad=False).to(self.device))
                loss += mse_loss

                if gci_name == "gci2":
                    rand_index = np.random.choice(len(gci_dataset), size=512)
                    gci_batch = gci_dataset[rand_index]
                
                    #prots = [self.class_index_dict[p] for p in self.dataset.evaluation_classes.as_str]
                    prots = [self.class_index_dict[d] for d in disease if d in self.class_index_dict] #.as_str]
                    idxs_for_negs = np.random.choice(prots, size=len(gci_batch), replace=True)
                    rand_prot_ids = th.tensor(idxs_for_negs).to(self.device)
                    neg_data = th.cat([gci_batch[:, :2], rand_prot_ids.unsqueeze(1)], dim=1)

                    dst = self.model(neg_data, gci_name, neg=True)
                    mse_loss = criterion(dst,
                                         th.ones(dst.shape, requires_grad=False).to(self.device))
                    loss += mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            with th.no_grad():
                self.model.eval()
                valid_loss = 0
                gci2_data = validation_dataset
                dst = self.model(gci2_data, "gci2")
                loss = criterion(dst, th.zeros(dst.shape, requires_grad=False).to(self.device))
                valid_loss += loss.detach().item()

            checkpoint = 500
            if best_loss > valid_loss:
                best_loss = valid_loss
                th.save(self.model.state_dict(), self.model_filepath)
            #if (epoch + 1) % checkpoint == 0:
            #    print(f'\nEpoch {epoch+1}: Train loss: {train_loss:.4f} Valid loss: {valid_loss:.4f}')

    def evaluate_ppi(self):
        self.init_model()
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        with th.no_grad():
            self.model.eval()

            eval_method = self.model.gci2_loss

            evaluator = ELBoxEmbeddingsPPIEvaluator(
                self.dataset.testing, eval_method, self.dataset.ontology,
                self.class_index_dict, self.object_property_index_dict, device=self.device)
            evaluator()
            evaluator.print_metrics()

    def get_embeddings(self):
        self.init_model()
        
        #print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()

        ent_embeds = {k:v for k,v in zip(self.class_index_dict.keys(), self.model.class_embed.weight.cpu().detach().numpy())}
        rel_embeds = {k:v for k,v in zip(self.object_property_index_dict.keys(), self.model.rel_embed.weight.cpu().detach().numpy())}
        
        return ent_embeds, rel_embeds


class ELEmModule(ELModule):

    def __init__(self, nb_ont_classes, nb_rels, embed_dim=50, margin=0.1):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels

        self.embed_dim = embed_dim

        # Embedding layer for classes centers.
        self.class_embed = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_embed.weight, a=-1, b=1)
        weight_data = th.linalg.norm(self.class_embed.weight.data, axis=1).reshape(-1, 1)
        self.class_embed.weight.data /= weight_data

        # Embedding layer for classes radii.
        self.class_rad = nn.Embedding(self.nb_ont_classes, 1)
        nn.init.uniform_(self.class_rad.weight, a=-1, b=1)
        weight_data = th.linalg.norm(self.class_rad.weight.data, axis=1).reshape(-1, 1)
        self.class_rad.weight.data /= weight_data

        # Embedding layer for ontology object properties
        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, a=-1, b=1)
        weight_data = th.linalg.norm(self.rel_embed.weight.data, axis=1).reshape(-1, 1)
        self.rel_embed.weight.data /= weight_data

        self.margin = margin

    # Regularization method to force n-ball to be inside unit ball
    def class_reg(self, x):
        res = th.abs(th.linalg.norm(x, axis=1) - 1)
        res = th.reshape(res, [-1, 1])
        return res

    # Loss function for normal form :math:`C \sqsubseteq D`
    def gci0_loss(self, data, neg=False):
        c = self.class_embed(data[:, 0])
        d = self.class_embed(data[:, 1])
        rc = th.abs(self.class_rad(data[:, 0]))
        rd = th.abs(self.class_rad(data[:, 1]))
        dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
        loss = th.relu(dist - self.margin)
        return loss + self.class_reg(c) + self.class_reg(d)

    # Loss function for normal form :math:`C \sqcap D \sqsubseteq E`
    def gci1_loss(self, data, neg=False):
        c = self.class_embed(data[:, 0])
        d = self.class_embed(data[:, 1])
        e = self.class_embed(data[:, 2])
        rc = th.abs(self.class_rad(data[:, 0]))
        rd = th.abs(self.class_rad(data[:, 1]))

        sr = rc + rd
        dst = th.linalg.norm(d - c, dim=1, keepdim=True)
        dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
        dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
        loss = th.relu(dst - sr - self.margin) + th.relu(dst2 - rc - self.margin)
        loss += th.relu(dst3 - rd - self.margin)

        return loss + self.class_reg(c) + self.class_reg(d) + self.class_reg(e)

    # Loss function for normal form :math:`C \sqcap D \sqsubseteq \bot`
    def gci1_bot_loss(self, data, neg=False):
        c = self.class_embed(data[:, 0])
        d = self.class_embed(data[:, 1])
        rc = self.class_rad(data[:, 0])
        rd = self.class_rad(data[:, 1])

        sr = rc + rd
        dst = th.reshape(th.linalg.norm(d - c, axis=1), [-1, 1])
        return th.relu(sr - dst + self.margin) + self.class_reg(c) + self.class_reg(d)

    # Loss function for normal form :math:`C \sqsubseteq \exists R. D`
    def gci2_loss(self, data, neg=False):

        if neg:
            return self.gci2_loss_neg(data)

        else:
            # C subSelf.ClassOf R some D
            c = self.class_embed(data[:, 0])
            rE = self.rel_embed(data[:, 1])
            d = self.class_embed(data[:, 2])

            rc = th.abs(self.class_rad(data[:, 0]))
            rd = th.abs(self.class_rad(data[:, 2]))

            dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
            loss = th.relu(dst + rc - rd - self.margin)
            return loss + self.class_reg(c) + self.class_reg(d)

    # Loss function for normal form :math:`C \nsqsubseteq \exists R. D`
    def gci2_loss_neg(self, data):

        c = self.class_embed(data[:, 0])
        rE = self.rel_embed(data[:, 1])

        d = self.class_embed(data[:, 2])
        rc = th.abs(self.class_rad(data[:, 1]))
        rd = th.abs(self.class_rad(data[:, 2]))

        dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
        loss = th.relu(rc + rd - dst + self.margin)
        return loss + self.class_reg(c) + self.class_reg(d)

    # Loss function for normal form :math:`\exists R. C \sqsubseteq D`
    def gci3_loss(self, data, neg=False):

        rE = self.rel_embed(data[:, 0])
        c = self.class_embed(data[:, 1])
        d = self.class_embed(data[:, 2])
        rc = th.abs(self.class_rad(data[:, 1]))
        rd = th.abs(self.class_rad(data[:, 2]))

        euc = th.linalg.norm(c - rE - d, dim=1, keepdim=True)
        loss = th.relu(euc - rc - rd - self.margin)
        return loss + self.class_reg(c) + self.class_reg(d)



class ELEmbeddings(EmbeddingELModel):

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0,
                 reg_norm=1,
                 learning_rate=0.001,
                 epochs=1000,
                 batch_size=4096 * 8,
                 model_filepath=None,
                 device='cpu'
                 ):
        super().__init__(dataset, batch_size=batch_size,embed_dim=embed_dim,  extended=True, model_filepath=model_filepath)

        self.embed_dim = embed_dim
        self.batch_size=batch_size
        self.margin = margin
        self.reg_norm = reg_norm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self._loaded = False
        self._loaded_eval = False
        self.extended = False
        self.init_model()
        self._testing_set = None
        self._training_set = None
        self._head_entities = None
        self._tail_entities = None
        

    def init_model(self):
        self.model = ELEmModule(
            len(self.class_index_dict),  # number of ontology classes
            len(self.object_property_index_dict),  # number of ontology object properties
            embed_dim=self.embed_dim,
            margin=self.margin
        ).to(self.device)

    def train(self):
        optimizer = th.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_loss = float('inf')

        #_, disease = self.dataset.evaluation_classes #on example work good
        _, disease = self.dataset.evaluation_class2()

        #for i,j in self.class_index_dict.items():
        #    print(i,j)
            

        for epoch in trange(self.epochs):
            self.model.train()

            train_loss = 0
            loss = 0

            # Notice how we use the ``training_datasets`` variable directly
            # and every element of it is a pair (GCI name, GCI tensor data).
            for gci_name, gci_dataset in self.training_datasets.items():
                if len(gci_dataset) == 0:
                    continue

                loss += th.mean(self.model(gci_dataset[:], gci_name))
                if gci_name == "gci2":
                    #prots = [self.class_index_dict[p] for p in self.dataset.evaluation_classes.as_str]
                    #print(self.class_index_dict)
                    
                    prots = [self.class_index_dict[d] for d in disease ]#if d in self.class_index_dict] #.as_str]
                    idxs_for_negs = np.random.choice(prots, size=len(gci_dataset), replace=True)
                    rand_index = th.tensor(idxs_for_negs).to(self.device)
                    data = gci_dataset[:]
                    neg_data = th.cat([data[:, :2], rand_index.unsqueeze(1)], dim=1)
                    loss += th.mean(self.model(neg_data, gci_name, neg=True))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            loss = 0
            with th.no_grad():
                self.model.eval()
                valid_loss = 0
                gci2_data = self.validation_datasets["gci2"][:]
                loss = th.mean(self.model(gci2_data, "gci2"))
                valid_loss += loss.detach().item()

            checkpoint = 1
            if best_loss > valid_loss:
                best_loss = valid_loss
                th.save(self.model.state_dict(), self.model_filepath)
            #if (epoch + 1) % checkpoint == 0:
            #    print(f'\nEpoch {epoch}: Train loss: {train_loss:4f} Valid loss: {valid_loss:.4f}')

    def evaluate_ppi(self):
        self.init_model()
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        with th.no_grad():
            self.model.eval()

            eval_method = self.model.gci2_loss

            evaluator = ELEmbeddingsPPIEvaluator(
                self.dataset.testing, eval_method, self.dataset.ontology,
                self.class_index_dict, self.object_property_index_dict, device=self.device)
            evaluator()
            evaluator.print_metrics()

    def get_embeddings(self):
        self.init_model()
        
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()

        ent_embeds = {k:v for k,v in zip(self.class_index_dict.keys(), self.model.class_embed.weight.cpu().detach().numpy())}
        rel_embeds = {k:v for k,v in zip(self.object_property_index_dict.keys(), self.model.rel_embed.weight.cpu().detach().numpy())}
        
        return ent_embeds, rel_embeds

    def load_eval_data(self):
        
        if self._loaded_eval:
            return

        eval_property = self.dataset.get_evaluation_property
        #g,d = self.dataset.evaluation_classes
        g,d = self.dataset.evaluation_class2()
        
        self._head_entities = g
        self._tail_entities = d


        eval_projector = projector_factory('dl2vec', taxonomy=False, relations=[eval_property])

        self._training_set = eval_projector.project(self.dataset.ontology)
        self._testing_set = eval_projector.project(self.dataset.testing)
        
        
        self._loaded_eval = True

    @property
    def eval_method(self, data):
        return self.model.gci2_loss(data)

    @property
    def testing_set(self):
        if self._testing_set is None:
            projector = TaxonomyWithRelationsProjector(taxonomy=False,
                                                       relations=["http://is_associated_with"]
                                                       )
            #projector = projector_factory('dl2vec', taxonomy=False, relations=["http://is_associated_with"])
            self._testing_set = projector.project(self.dataset.testing)
        return self._testing_set

    @property
    def training_set(self):
        if self._training_set is None:
            projector = TaxonomyWithRelationsProjector(taxonomy=False,
                                                       relations=["http://is_associated_with"]
                                                       )
            #projector = projector_factory('dl2vec', taxonomy=False, relations=["http://is_associated_with"])

            self._training_set = projector.project(self.dataset.ontology)
        return self._training_set

    @property
    def head_entities(self):
        if self._head_entities is None:
            #self._head_entities,_ = self.dataset.evaluation_classes
            self._head_entities,_ = self.dataset.evaluation_class2()
        return self._head_entities #.as_str

    @property
    def tail_entities(self):
        if self._tail_entities is None:
            #_ , self._tail_entities = self.dataset.evaluation_classes
            _ , self._tail_entities = self.dataset.evaluation_class2()
        return self._tail_entities #.as_str
        

    def load_best_model(self):
        self.model.load_state_dict(th.load(self.model_filepath))

