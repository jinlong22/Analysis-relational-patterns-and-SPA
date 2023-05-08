from numpy.random.mtrand import normal
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict as ddict
import random
from .DataPreprocess import *
from IPython import embed
import torch.nn.functional as F
import time
import queue

class UniSampler(BaseSampler):
    """Random negative sampling 
    Filtering out positive samples and selecting some samples randomly as negative samples.

    Attributes:
        cross_sampling_flag: The flag of cross sampling head and tail negative samples.
    """
    def __init__(self, args):
        super().__init__(args)
        self.cross_sampling_flag = 0

    def sampling(self, data):
        """Filtering out positive samples and selecting some samples randomly as negative samples.
        
        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        batch_data = {}
        neg_ent_sample = []
        subsampling_weight = []
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        if self.cross_sampling_flag == 0:
            batch_data['mode'] = "head-batch"
            for h, r, t in data:
                neg_head = self.head_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_head)
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r-1)]
                    subsampling_weight.append(weight)
        else:
            batch_data['mode'] = "tail-batch"
            for h, r, t in data:
                neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_tail)
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r-1)]
                    subsampling_weight.append(weight)

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_sample'] = torch.LongTensor(np.array(neg_ent_sample))
        if self.args.use_weight:
            batch_data["subsampling_weight"] = torch.sqrt(1/torch.tensor(subsampling_weight))
        return batch_data
    
    def uni_sampling(self, data):
        batch_data = {}
        neg_head_list = []
        neg_tail_list = []
        for h, r, t in data:
            neg_head = self.head_batch(h, r, t, self.args.num_neg)
            neg_head_list.append(neg_head)
            neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
            neg_tail_list.append(neg_tail)

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_head'] = torch.LongTensor(np.arrary(neg_head_list))
        batch_data['negative_tail'] = torch.LongTensor(np.arrary(neg_tail_list))
        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'mode']

class AdvSampler(BaseSampler):
    """Self-adversarial negative sampling, in math:
    
    p\left(h_{j}^{\prime}, r, t_{j}^{\prime} \mid\left\{\left(h_{i}, r_{i}, t_{i}\right)\right\}\right)=\frac{\exp \alpha f_{r}\left(\mathbf{h}_{j}^{\prime}, \mathbf{t}_{j}^{\prime}\right)}{\sum_{i} \exp \alpha f_{r}\left(\mathbf{h}_{i}^{\prime}, \mathbf{t}_{i}^{\prime}\right)}
    
    Attributes:
        freq_hr: The count of (h, r) pairs.
        freq_tr: The count of (t, r) pairs.
    """
    def __init__(self, args):
        super().__init__(args)
        self.freq_hr, self.freq_tr = self.calc_freq()
    def sampling(self, pos_sample):
        """Self-adversarial negative sampling.
    
        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        data = pos_sample.numpy().tolist()
        adv_sampling = []
        for h, r, t in data:
            weight = self.freq_hr[(h, r)] + self.freq_tr[(t, r)]
            adv_sampling.append(weight)
        adv_sampling = torch.tensor(adv_sampling, dtype=torch.float32).cuda()
        adv_sampling = torch.sqrt(1 / adv_sampling)
        return adv_sampling
    def calc_freq(self):
        """Calculating the freq_hr and freq_tr.
        
        Returns:
            freq_hr: The count of (h, r) pairs.
            freq_tr: The count of (t, r) pairs.
        """
        freq_hr, freq_tr = {}, {}
        for h, r, t in self.train_triples:
            if (h, r) not in freq_hr:
                freq_hr[(h, r)] = self.args.freq_init
            else:
                freq_hr[(h, r)] += 1
            if (t, r) not in freq_tr:
                freq_tr[(t, r)] = self.args.freq_init
            else:
                freq_tr[(t, r)] += 1
        return freq_hr, freq_tr

class TestSampler(object):
    """Sampling triples and recording positive triples for testing.
    Attributes:
        sampler: The function of training sampler.
        hr2t_all: Record the tail corresponding to the same head and relation.
        rt2h_all: Record the head corresponding to the same tail and relation.
        num_ent: The count of entities.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.num_ent = sampler.args.num_ent

    def get_hr2t_rt2h_from_all(self):
        """Get the set of hr2t and rt2h from all datasets(train, valid, and test), the data type is tensor.
        Update:
            self.hr2t_all: The set of hr2t.
            self.rt2h_all: The set of rt2h.
        """
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        """Sampling triples and recording positive triples for testing.
        Args:
            data: The triples used to be sampled.
        Returns:
            batch_data: The data used to be evaluated.
        """
        batch_data = {}
        head_label = torch.zeros(len(data), self.num_ent)
        tail_label = torch.zeros(len(data), self.num_ent)
        for idx, triple in enumerate(data):
            head, rel, tail = triple
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        return batch_data

class TestSampler1(object):
    """Sampling triples and recording positive triples for testing.
    Attributes:
        sampler: The function of training sampler.
        hr2t_all: Record the tail corresponding to the same head and relation.
        rt2h_all: Record the head corresponding to the same tail and relation.
        num_ent: The count of entities.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.num_ent = sampler.args.num_ent

    def get_hr2t_rt2h_from_all(self):
        """Get the set of hr2t and rt2h from all datasets(train, valid, and test), the data type is tensor.
        Update:
            self.hr2t_all: The set of hr2t.
            self.rt2h_all: The set of rt2h.
        """
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))
    
    def sampling(self, data):
        """Sampling triples and recording positive triples for testing.
        Args:
            data: The triples used to be sampled.
        Returns:
            batch_data: The data used to be evaluated.
        """
        
        # import pdb;pdb.set_trace()
        batch_data = {}
        head_label = torch.zeros(len(data), self.num_ent)
        tail_label = torch.zeros(len(data), self.num_ent)
        
        if self.sampler.args.use_sym_weight: 
            sym_data = []                                                       
            sym_weight = torch.zeros(len(data), 2)                              
        
        if self.sampler.args.use_inv_weight: 
            inv_data = []                                                       
            inv_weight = torch.zeros(len(data), self.sampler.max_inv, 2)        

        if self.sampler.args.use_sub_weight: 
            sub_data = []                                                       
            sub_weight = torch.zeros(len(data), self.sampler.max_sub, 2)        

        if self.sampler.args.use_comp2_weight: 
            comp2_data1 = []                                                    
            comp2_data2 = []                                                    
            comp2_weight = torch.zeros(len(data), self.sampler.max_comp2, 2)    
            comp2_rel_inv= torch.zeros(len(data), self.sampler.max_comp2, 2)    
        
        if self.sampler.args.use_comp3_weight: 
            comp3_data1 = []                                                     
            comp3_data2 = []                                                     
            comp3_data3 = []                                                     
            comp3_weight = torch.zeros(len(data), self.sampler.max_comp3, 2)     
        
        for idx, triple in enumerate(data):
            head, rel, tail = triple
            
            #filter
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0

            if self.sampler.args.use_sym_weight:
                sym_data.append([tail, rel, head])
                sym_weight[idx] = torch.tensor(self.sampler.sym_weight[rel])

            if self.sampler.args.use_inv_weight:
                tmp_data = []
                for inv_rel in self.sampler.inv_rule[rel]:
                    tmp_data.append([tail, inv_rel, head])
                inv_data.append(tmp_data)
                inv_weight[idx] = torch.tensor(self.sampler.inv_weight[rel])
            
            if self.sampler.args.use_sub_weight:
                tmp_data = []
                for sub_rel in self.sampler.sub_rule[rel]:
                    tmp_data.append([head, sub_rel, tail])
                sub_data.append(tmp_data)
                sub_weight[idx] = torch.tensor(self.sampler.sub_weight[rel])
            
            if self.sampler.args.use_comp2_weight:
                tmp_data1 = []
                tmp_data2 = []
                for comp2_rel in self.sampler.comp2_rule[rel]:
                    # import pdb;pdb.set_trace()
                    comp2_rel1, comp2_rel2 = comp2_rel
                    # import pdb;pdb.set_trace()
                    tmp_data1.append([head, comp2_rel1, tail])
                    tmp_data2.append([head, comp2_rel2, tail])
                comp2_data1.append(tmp_data1)
                comp2_data2.append(tmp_data2)
                comp2_weight[idx] = torch.tensor(self.sampler.comp2_weight[rel])
                comp2_rel_inv[idx] = torch.tensor(self.sampler.comp2_rel_inv[rel])
            
            if self.sampler.args.use_comp3_weight:
                tmp_data1 = []
                tmp_data2 = []
                tmp_data3 = []
                for comp3_rel in self.sampler.comp3_rule[rel]:
                    # import pdb;pdb.set_trace()
                    comp3_rel1, comp3_rel2, comp3_rel3 = comp3_rel
                    # import pdb;pdb.set_trace()
                    tmp_data1.append([head, comp3_rel1, tail])
                    tmp_data2.append([head, comp3_rel2, tail])
                    tmp_data3.append([head, comp3_rel3, tail])
                # import pdb;pdb.set_trace()
                comp3_data1.append(tmp_data1)
                comp3_data2.append(tmp_data2)
                comp3_data3.append(tmp_data3)
                
                comp3_weight[idx] = torch.tensor(self.sampler.comp3_weight[rel])

        # print(torch.tensor(inv_data).shape)
        # import pdb;pdb.set_trace()
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        
        if self.sampler.args.use_sym_weight:
            batch_data["symmetric_sample"] = torch.tensor(sym_data)     
            batch_data["sym_weight"] = sym_weight                       
        
        if self.sampler.args.use_inv_weight:
            batch_data["inverse_sample"] = torch.tensor(inv_data)       
            batch_data["inv_weight"] = inv_weight                       
            batch_data["max_inv"] = self.sampler.max_inv

        if self.sampler.args.use_sub_weight:
            batch_data["subrelation_sample"] = torch.tensor(sub_data)   
            batch_data["sub_weight"] = sub_weight                       
            batch_data["max_sub"] = self.sampler.max_sub
        
        if self.sampler.args.use_comp2_weight:
            batch_data["comp2_sample1"] = torch.tensor(comp2_data1)     
            batch_data["comp2_sample2"] = torch.tensor(comp2_data2)     
            batch_data["comp2_weight"] = comp2_weight                   
            batch_data["max_comp2"] = self.sampler.max_comp2
            batch_data["comp2_rel_inv"] = comp2_rel_inv
        
        if self.sampler.args.use_comp3_weight:
            batch_data["comp3_sample1"] = torch.tensor(comp3_data1)     
            batch_data["comp3_sample2"] = torch.tensor(comp3_data2)     
            batch_data["comp3_sample3"] = torch.tensor(comp3_data3)     
            batch_data["comp3_weight"] = comp3_weight                   
            batch_data["max_comp3"] = self.sampler.max_comp3
        
        return batch_data

class GraphTestSampler(object):
    """Sampling graph for testing.

    Attributes:
        sampler: The function of training sampler.
        hr2t_all: Record the tail corresponding to the same head and relation.
        rt2h_all: Record the head corresponding to the same tail and relation.
        num_ent: The count of entities.
        triples: The training triples.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.num_ent = sampler.args.num_ent
        self.triples = sampler.train_triples

    def get_hr2t_rt2h_from_all(self):
        """Get the set of hr2t and rt2h from all datasets(train, valid, and test), the data type is tensor.

        Update:
            self.hr2t_all: The set of hr2t.
            self.rt2h_all: The set of rt2h.
        """
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        """Sampling graph for testing.

        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The data used to be evaluated.
        """
        batch_data = {}
        head_label = torch.zeros(len(data), self.num_ent)
        tail_label = torch.zeros(len(data), self.num_ent)
        for idx, triple in enumerate(data):
            # from IPython import embed;embed();exit()
            head, rel, tail = triple
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        
        head, rela, tail = np.array(self.triples).transpose()
        graph, rela, norm = self.sampler.build_graph(self.num_ent, (head, rela, tail), -1)
        batch_data["graph"]  = graph
        batch_data["rela"]   = rela
        batch_data["norm"]   = norm
        batch_data["entity"] = torch.arange(0, self.num_ent, dtype=torch.long).view(-1,1)
        
        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "head_label", "tail_label",\
             "graph", "rela", "norm", "entity"]

class CompGCNTestSampler(object):
    """Sampling graph for testing.

    Attributes:
        sampler: The function of training sampler.
        hr2t_all: Record the tail corresponding to the same head and relation.
        rt2h_all: Record the head corresponding to the same tail and relation.
        num_ent: The count of entities.
        triples: The training triples.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.num_ent = sampler.args.num_ent
        self.triples = sampler.t_triples

    def get_hr2t_rt2h_from_all(self):
        """Get the set of hr2t and rt2h from all datasets(train, valid, and test), the data type is tensor.

        Update:
            self.hr2t_all: The set of hr2t.
            self.rt2h_all: The set of rt2h.
        """
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        """Sampling graph for testing.

        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The data used to be evaluated.
        """
        batch_data = {}
        
        head_label = torch.zeros(len(data), self.num_ent)
        tail_label = torch.zeros(len(data), self.num_ent)
        
        for idx, triple in enumerate(data):
            # from IPython import embed;embed();exit()
            head, rel, tail = triple
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        
        graph, relation, norm = \
            self.sampler.build_graph(self.num_ent, np.array(self.triples).transpose(), -0.5)
    
        batch_data["graph"]  = graph
        batch_data["rela"]   = relation
        batch_data["norm"]   = norm
        batch_data["entity"] = torch.arange(0, self.num_ent, dtype=torch.long).view(-1,1)
        
        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "head_label", "tail_label",\
             "graph", "rela", "norm", "entity"]

class BernSampler(BaseSampler):
    """Using bernoulli distribution to select whether to replace the head entity or tail entity.
    
    Attributes:
        lef_mean: Record the mean of head entity
        rig_mean: Record the mean of tail entity
    """
    def __init__(self, args):
        super().__init__(args)
        self.lef_mean, self.rig_mean = self.calc_bern()
    def __normal_batch(self, h, r, t, neg_size):
        """Generate replace head/tail list according to Bernoulli distribution.
        
        Args:
            h: The head of triples.
            r: The relation of triples.
            t: The tail of triples.
            neg_size: The number of negative samples corresponding to each triple

        Returns:
             numpy.array: replace head list and replace tail list.
        """
        neg_size_h = 0
        neg_size_t = 0
        prob = self.rig_mean[r] / (self.rig_mean[r] + self.lef_mean[r])
        for i in range(neg_size):
            if random.random() > prob:
                neg_size_h += 1
            else:
                neg_size_t += 1

        res = []

        neg_list_h = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_h:
            neg_tmp_h = self.corrupt_head(t, r, num_max=(neg_size_h - neg_cur_size) * 2)
            neg_list_h.append(neg_tmp_h)
            neg_cur_size += len(neg_tmp_h)
        if neg_list_h != []:
            neg_list_h = np.concatenate(neg_list_h)
        
        for hh in neg_list_h[:neg_size_h]:
            res.append((hh, r, t))
        
        neg_list_t = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_t:
            neg_tmp_t = self.corrupt_tail(h, r, num_max=(neg_size_t - neg_cur_size) * 2)
            neg_list_t.append(neg_tmp_t)
            neg_cur_size += len(neg_tmp_t)
        if neg_list_t != []:
            neg_list_t = np.concatenate(neg_list_t)
        
        for tt in neg_list_t[:neg_size_t]:
            res.append((h, r, tt))

        return res

    def sampling(self, data):
        """Using bernoulli distribution to select whether to replace the head entity or tail entity.
    
        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        batch_data = {}
        neg_ent_sample = []

        batch_data['mode'] = 'bern'
        for h, r, t in data:
            neg_ent = self.__normal_batch(h, r, t, self.args.num_neg)
            neg_ent_sample += neg_ent
        
        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data["negative_sample"] = torch.LongTensor(np.array(neg_ent_sample))

        return batch_data
    
    def calc_bern(self):
        """Calculating the lef_mean and rig_mean.
        
        Returns:
            lef_mean: Record the mean of head entity.
            rig_mean: Record the mean of tail entity.
        """
        h_of_r = ddict(set)
        t_of_r = ddict(set)
        freqRel = ddict(float)
        lef_mean = ddict(float)
        rig_mean = ddict(float)
        for h, r, t in self.train_triples:
            freqRel[r] += 1.0
            h_of_r[r].add(h)
            t_of_r[r].add(t)
        for r in h_of_r:
            lef_mean[r] = freqRel[r] / len(h_of_r[r])
            rig_mean[r] = freqRel[r] / len(t_of_r[r])
        return lef_mean, rig_mean

    @staticmethod
    def sampling_keys():
        return ['positive_sample', 'negative_sample', 'mode']


'''继承torch.Dataset'''
class KGDataset(Dataset):

    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]