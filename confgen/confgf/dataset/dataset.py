import os, sys
import pickle
import copy
import json
from collections import defaultdict

import numpy as np
import random

import torch
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_networkx
from torch_scatter import scatter
from torch.nn.utils.rnn import pad_sequence

#from torch.utils.data import Dataset

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from rdkit import RDLogger
import networkx as nx
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')

from confgf import utils

from torch_geometric.datasets import qm9
from torch.utils.data import random_split
from typing import Optional, List
from torch.utils.data import DataLoader
import torch
from torch_geometric.data import Data
from typing import Callable
import random

def collate_(batch, y_index=0):
    # avoid forming the batch indice.
    data = batch[0]
    try:
        data.y = data.y.squeeze(dim=1)[:, y_index]
    except:
        pass
    return data



def group_same_size(
    dataset: Data,
):
    data_list=dataset
    #data_list = list(dataset) #is this a problem?
    data_list.sort(key=lambda data: data.atom_type.shape[0])
    # grouped dataset by size
    grouped_dataset = []
    for i in range(len(data_list)):
        data = data_list[i]
        if i == 0:
            group = [data]
        else:
            last_data = data_list[i-1]
            if data.atom_type.shape[0] == last_data.atom_type.shape[0]:
                group.append(data)
            else:
                grouped_dataset.append((last_data.atom_type.shape[0], group))
                group = [data]
    return grouped_dataset

def batch_same_size_Data(
    grouped_dataset: Data,
    batch_size: int,
):
    # batched dataset, according to the batch size. 
    batched_dataset = []
    for size, group in grouped_dataset:
        batch_num_in_group = (len(group) // batch_size) + 1 if len(group) % batch_size != 0 else len(group) // batch_size
        for i in range(batch_num_in_group):
            lower_bound = i * batch_size
            upper_bound = min((i+1) * batch_size, len(group))
            #create empty tensor for edge index
            pad = torch.full((2*(upper_bound-lower_bound), group[lower_bound].atom_type.shape[0]), -1) #all elements in group have identical graph size
            
            batch = group[lower_bound:upper_bound]
            #y = torch.cat([batch[i].y.unsqueeze(0) for i in range(len(batch))], dim=0)
            atom_type = torch.cat([batch[i].atom_type.unsqueeze(0) for i in range(len(batch))], dim=0)
            pos = torch.cat([batch[i].pos.unsqueeze(0) for i in range(len(batch))], dim=0)
            #boltzmannweight = torch.cat([batch[i].boltzmannweight.unsqueeze(0) for i in range(len(batch))], dim=0)
            #bond_edge_index = torch.cat([batch[i].bond_edge_index.unsqueeze(0) for i in range(len(batch))], dim=0)
            #insert edge_index inside padded tensor

            edge_index = pad_sequence([batch[i].edge_index.t() for i in range(len(batch))], batch_first=False, padding_value= 0).permute(1,2,0) #TODO:change padding value
            edge_index = edge_index.clone().view(1,-1, edge_index.size(2)).squeeze()
            #edge_index = torch.cat([batch[i].edge_index.unsqueeze(0) for i in range(len(batch))], dim=0)
            #edge_length = torch.cat([batch[i].edge_length.unsqueeze(0) for i in range(len(batch))], dim=0)
            #edge_order = torch.cat([batch[i].edge_order.unsqueeze(0) for i in range(len(batch))], dim=0)
            
            edge_type = pad_sequence([batch[i].edge_type.t() for i in range(len(batch))], batch_first=False, padding_value= 0).permute(1,0)#TODO:change padding value
            #print(edge_index.size())
            #sys.exit("above is edge type size")
            #idx = torch.cat([batch[i].idx.unsqueeze(0) for i in range(len(batch))], dim=0)
            #is_bond = torch.cat([batch[i].is_bond.unsqueeze(0) for i in range(len(batch))], dim=0)
            #pert_pos = torch.cat([batch[i].pert_pos.unsqueeze(0) for i in range(len(batch))], dim=0)
            #rdmol = torch.cat([batch[i].rdmol.unsqueeze(0) for i in range(len(batch))], dim=0)
            #smiles = torch.cat([batch[i].smiles.unsqueeze(0) for i in range(len(batch))], dim=0)
            #totalenergy = torch.cat([batch[i].totalenergy.unsqueeze(0) for i in range(len(batch))], dim=0)
            #data_batch = torch.cat([batch[i].batch.unsqueeze(0) for i in range(len(batch))], dim=0)

            batched_dataset.append(
                Data( #this looks like Batch from pyg that inherits from Data 
                    atom_type=atom_type, 
                    pos=pos,
                    #boltzmannweight = boltzmannweight,
                    #bond_edge_index = bond_edge_index,
                    edge_index = edge_index,
                    #edge_length = edge_length,
                    #edge_order = edge_order,
                    edge_type = edge_type,
                    #idx = idx,
                    #is_bond = is_bond,
                    #pert_pos = pert_pos,
                    #rdmol = rdmol,
                    #smiles = smiles,
                    #totalenergy = totalenergy,
                    #batch = data_batch,
                    batch_size=atom_type.shape[0],
                    graph_size=atom_type.shape[1],
                )
            )
    return batched_dataset

#new batch_same_size
def batch_same_size( 
    grouped_dataset: Data,
    batch_size: int,
):
    # batched dataset, according to the batch size. 
    batched_dataset = []
    for size, group in grouped_dataset:
        batch_num_in_group = (len(group) // batch_size) + 1 if len(group) % batch_size != 0 else len(group) // batch_size
        for i in range(batch_num_in_group):
            lower_bound = i * batch_size
            upper_bound = min((i+1) * batch_size, len(group))

            batch = group[lower_bound:upper_bound]
            batch_obj = Batch.from_data_list(data_list=batch)
            batch_obj.batch_size = upper_bound - lower_bound
            batched_dataset.append(
                batch_obj
            )
    return batched_dataset #TODO: run and see that works in scorenet with all the features


def rdmol_to_data(mol:Mol, smiles=None):
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()

    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)

    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [utils.BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float32)

    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)

    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type,
                rdmol=copy.deepcopy(mol), smiles=smiles)
    #data.nx = to_networkx(data, to_undirected=True)

    return data

def smiles_to_data(smiles):
    """
    Convert a SMILES to a pyg object that can be fed into ConfGF for generation
    """
    try:    
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    except:
        return None
        
    N = mol.GetNumAtoms()
    pos = torch.rand((N, 3), dtype=torch.float32)

    atomic_number = []
    aromatic = []

    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [utils.BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index

    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type,
                rdmol=copy.deepcopy(mol), smiles=smiles)
    
    transform = Compose([
        utils.AddHigherOrderEdges(order=3),
        utils.AddEdgeLength(),
        utils.AddPlaceHolder(),
        utils.AddEdgeName()
    ])
    
    return transform(data)


def preprocess_iso17_dataset(base_path):
    train_path = os.path.join(base_path, 'iso17_split-0_train.pkl')
    test_path = os.path.join(base_path, 'iso17_split-0_test.pkl')
    with open(train_path, 'rb') as fin:
        raw_train = pickle.load(fin)
    with open(test_path, 'rb') as fin:
        raw_test = pickle.load(fin)

    smiles_list_train = [utils.mol_to_smiles(mol) for mol in raw_train]
    smiles_set_train = list(set(smiles_list_train))
    smiles_list_test = [utils.mol_to_smiles(mol) for mol in raw_test]
    smiles_set_test = list(set(smiles_list_test))

    print('preprocess train...')
    all_train = []
    for i in tqdm(range(len(raw_train))):
        smiles = smiles_list_train[i]
        data = rdmol_to_data(raw_train[i], smiles=smiles)
        all_train.append(data)

    print('Train | find %d molecules with %d confs' % (len(smiles_set_train), len(all_train)))    
    
    print('preprocess test...')
    all_test = []
    for i in tqdm(range(len(raw_test))):
        smiles = smiles_list_test[i]
        data = rdmol_to_data(raw_test[i], smiles=smiles)
        all_test.append(data)

    print('Test | find %d molecules with %d confs' % (len(smiles_set_test), len(all_test)))  

    return all_train, all_test


    


def preprocess_GEOM_dataset(base_path, dataset_name, conf_per_mol=5, train_size=0.8, tot_mol_size=50000, seed=None):
    """
    base_path: directory that contains GEOM dataset
    dataset_name: dataset name in [qm9, drugs]
    conf_per_mol: keep mol that has at least conf_per_mol confs, and sampling the most probable conf_per_mol confs
    train_size ratio, val = test = (1-train_size) / 2
    tot_mol_size: max num of mols. The total number of final confs should be tot_mol_size * conf_per_mol
    seed: rand seed for RNG
    """

    # set random seed
    if seed is None:
        seed = 2021
    np.random.seed(seed)
    random.seed(seed)
    

    # read summary file
    assert dataset_name in ['qm9', 'drugs']
    summary_path = os.path.join(base_path, 'summary_%s.json' % dataset_name)
    with open(summary_path, 'r') as f:
        summ = json.load(f)

    # filter valid pickle path
    smiles_list = []
    pickle_path_list = []
    num_mols = 0    
    num_confs = 0    
    for smiles, meta_mol in tqdm(summ.items()):
        u_conf = meta_mol.get('uniqueconfs')
        if u_conf is None:
            continue
        pickle_path = meta_mol.get('pickle_path')
        if pickle_path is None:
            continue
        if u_conf < conf_per_mol:
            continue
        num_mols += 1
        num_confs += conf_per_mol
        smiles_list.append(smiles)
        pickle_path_list.append(pickle_path)

    random.shuffle(pickle_path_list)
    assert len(pickle_path_list) >= tot_mol_size, 'the length of all available mols is %d, which is smaller than tot mol size %d' % (len(pickle_path_list), tot_mol_size)

    pickle_path_list = pickle_path_list[:tot_mol_size]

    print('pre-filter: find %d molecules with %d confs, use %d molecules with %d confs' % (num_mols, num_confs, tot_mol_size, tot_mol_size*conf_per_mol))


    # 1. select the most probable 'conf_per_mol' confs of each 2D molecule
    # 2. split the dataset based on 2D structure, i.e., test on unseen graphs
    train_data, val_data, test_data = [], [], []
    val_size = test_size = (1. - train_size) / 2

    # generate train, val, test split indexes
    split_indexes = list(range(tot_mol_size))
    random.shuffle(split_indexes)
    index2split = {}
    for i in range(0, int(len(split_indexes) * train_size)):
        index2split[split_indexes[i]] = 'train'
    for i in range(int(len(split_indexes) * train_size), int(len(split_indexes) * (train_size + val_size))):
        index2split[split_indexes[i]] = 'val'
    for i in range(int(len(split_indexes) * (train_size + val_size)), len(split_indexes)):
        index2split[split_indexes[i]] = 'test'        


    num_mols = np.zeros(4, dtype=int) # (tot, train, val, test)
    num_confs = np.zeros(4, dtype=int) # (tot, train, val, test)


    bad_case = 0

    for i in tqdm(range(len(pickle_path_list))):
        
        with open(os.path.join(base_path, pickle_path_list[i]), 'rb') as fin:
            mol = pickle.load(fin)
        
        if mol.get('uniqueconfs') > len(mol.get('conformers')):
            bad_case += 1
            continue
        if mol.get('uniqueconfs') <= 0:
            bad_case += 1
            continue

        datas = []
        smiles = mol.get('smiles')

        if mol.get('uniqueconfs') == conf_per_mol:
            # use all confs
            conf_ids = np.arange(mol.get('uniqueconfs'))
        else:
            # filter the most probable 'conf_per_mol' confs
            all_weights = np.array([_.get('boltzmannweight', -1.) for _ in mol.get('conformers')])
            descend_conf_id = (-all_weights).argsort()
            conf_ids = descend_conf_id[:conf_per_mol]

        for conf_id in conf_ids:
            conf_meta = mol.get('conformers')[conf_id]
            data = rdmol_to_data(conf_meta.get('rd_mol'), smiles=smiles)
            labels = {
                'totalenergy': conf_meta['totalenergy'],
                'boltzmannweight': conf_meta['boltzmannweight'],
            }
            for k, v in labels.items():
                data[k] = torch.tensor([v], dtype=torch.float32)
            data['idx'] = torch.tensor([i], dtype=torch.long)
            datas.append(data)
        assert len(datas) == conf_per_mol

        if index2split[i] == 'train':
            train_data.extend(datas)
            num_mols += [1, 1, 0, 0]
            num_confs += [len(datas), len(datas), 0, 0]
        elif index2split[i] == 'val':    
            val_data.extend(datas)
            num_mols += [1, 0, 1, 0]
            num_confs += [len(datas), 0, len(datas), 0]
        elif index2split[i] == 'test': 
            test_data.extend(datas)
            num_mols += [1, 0, 0, 1]
            num_confs += [len(datas), 0, 0, len(datas)] 
        else:
            raise ValueError('unknown index2split value.')                         

    print('post-filter: find %d molecules with %d confs' % (num_mols[0], num_confs[0]))    
    print('train size: %d molecules with %d confs' % (num_mols[1], num_confs[1]))    
    print('val size: %d molecules with %d confs' % (num_mols[2], num_confs[2]))    
    print('test size: %d molecules with %d confs' % (num_mols[3], num_confs[3]))    
    print('bad case: %d' % bad_case)
    print('done!')

    return train_data, val_data, test_data, index2split


def get_GEOM_testset(base_path, dataset_name, block, tot_mol_size=200, seed=None, confmin=50, confmax=500):
    """
    base_path: directory that contains GEOM dataset
    dataset_name: dataset name, should be in [qm9, drugs]
    block: block the training and validation set
    tot_mol_size: size of the test set
    seed: rand seed for RNG
    confmin and confmax: range of the number of conformations
    """

    #block smiles in train / val 
    block_smiles = defaultdict(int)
    for block_ in block:
        for i in range(len(block_)):
            block_smiles[block_[i].smiles] = 1

    # set random seed
    if seed is None:
        seed = 2021
    np.random.seed(seed)
    random.seed(seed)
    

    # read summary file
    assert dataset_name in ['qm9', 'drugs']
    summary_path = os.path.join(base_path, 'summary_%s.json' % dataset_name)
    with open(summary_path, 'r') as f:
        summ = json.load(f)

    # filter valid pickle path
    smiles_list = []
    pickle_path_list = []
    num_mols = 0    
    num_confs = 0    
    for smiles, meta_mol in tqdm(summ.items()):
        u_conf = meta_mol.get('uniqueconfs')
        if u_conf is None:
            continue
        pickle_path = meta_mol.get('pickle_path')
        if pickle_path is None:
            continue
        if u_conf < confmin or u_conf > confmax:
            continue
        if block_smiles[smiles] == 1:
            continue

        num_mols += 1
        num_confs += u_conf
        smiles_list.append(smiles)
        pickle_path_list.append(pickle_path)


    random.shuffle(pickle_path_list)
    assert len(pickle_path_list) >= tot_mol_size, 'the length of all available mols is %d, which is smaller than tot mol size %d' % (len(pickle_path_list), tot_mol_size)

    pickle_path_list = pickle_path_list[:tot_mol_size]

    print('pre-filter: find %d molecules with %d confs' % (num_mols, num_confs))


    bad_case = 0
    all_test_data = []
    num_valid_mol = 0
    num_valid_conf = 0

    for i in tqdm(range(len(pickle_path_list))):
        
        with open(os.path.join(base_path, pickle_path_list[i]), 'rb') as fin:
            mol = pickle.load(fin)
        
        if mol.get('uniqueconfs') > len(mol.get('conformers')):
            bad_case += 1
            continue
        if mol.get('uniqueconfs') <= 0:
            bad_case += 1
            continue

        datas = []
        smiles = mol.get('smiles')

        conf_ids = np.arange(mol.get('uniqueconfs'))
      
        for conf_id in conf_ids:
            conf_meta = mol.get('conformers')[conf_id]
            data = rdmol_to_data(conf_meta.get('rd_mol'), smiles=smiles)
            labels = {
                'totalenergy': conf_meta['totalenergy'],
                'boltzmannweight': conf_meta['boltzmannweight'],
            }
            for k, v in labels.items():
                data[k] = torch.tensor([v], dtype=torch.float32)
            data['idx'] = torch.tensor([i], dtype=torch.long)
            datas.append(data)

      
        all_test_data.extend(datas)
        num_valid_mol += 1
        num_valid_conf += len(datas)

    print('poster-filter: find %d molecules with %d confs' % (num_valid_mol, num_valid_conf))


    return all_test_data



class GEOMDataset(Dataset):

    def __init__(self, data=None, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform
        self.atom_types = self._atom_types()
        self.edge_types = self._edge_types()

    def __getitem__(self, idx):

        data = self.data[idx].clone()
        pos_center = data.pos.mean(dim=0)
        data.pos = data.pos - pos_center #TODO: what the hell is this? dont think we need this
        if self.transform is not None:
            data = self.transform(data)        
        return data

    def __len__(self):
        return len(self.data)

        
    def _atom_types(self):
        """All atom types."""
        atom_types = set()
        for graph in self.data:
            atom_types.update(graph.atom_type.tolist())
        return sorted(atom_types)

    def _edge_types(self):
        """All edge types."""
        edge_types = set()
        for graph in self.data:
            edge_types.update(graph.edge_type.tolist())
        return sorted(edge_types)




class GEOMDataset_PackedConf(GEOMDataset):

    def __init__(self, data=None, transform=None):
        super(GEOMDataset_PackedConf, self).__init__(data, transform)
        self._pack_data_by_mol()

    def _pack_data_by_mol(self):
        """
        pack confs with same mol into a single data object
        """
        self._packed_data = defaultdict(list)
        if hasattr(self.data, 'idx'):
            for i in range(len(self.data)):
                self._packed_data[self.data[i].idx.item()].append(self.data[i])
        else:
            for i in range(len(self.data)):
                self._packed_data[self.data[i].smiles].append(self.data[i])
        print('got %d molecules with %d confs' % (len(self._packed_data), len(self.data)))

        new_data = []
        # logic
        # save graph structure for each mol once, but store all confs 
        cnt = 0
        for k, v in self._packed_data.items():
            data = copy.deepcopy(v[0])
            all_pos = []
            for i in range(len(v)):
                pos_center = v[i].pos.mean(dim=0)
                pos = v[i].pos - pos_center
                all_pos.append(pos)
            data.pos_ref = torch.cat(all_pos, 0) # (num_conf*num_node, 3)
            data.num_pos_ref = torch.tensor([len(all_pos)], dtype=torch.long)
            #del data.pos

            if hasattr(data, 'totalenergy'):
                del data.totalenergy
            if hasattr(data, 'boltzmannweight'):
                del data.boltzmannweight
            new_data.append(data)
        self.new_data = new_data
        

    def __getitem__(self, idx):

        data = self.new_data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)        
        return data

    def __len__(self):
        return len(self.new_data)
        
        
class batched_QM9(Dataset):
    def __init__(self,  data=None, batch_size=100, transform=None):
        self.batch_size = batch_size
        # To batch train/val/test data respectively. For using, please set the indices of the data.
        self.subdataset = data #convert to list?
        self.grouped_data = group_same_size(self.subdataset)
        self.batched_data = batch_same_size(self.grouped_data, self.batch_size)
        #self.data_num = self.size
        self.size = len(self.batched_data) # batched version, a datapoint is a batch.

    def __getitem__(self, index):
        # to get the whole batched data.
        #if not self.flag:
        #    return super().__getitem__(index)
        return self.batched_data[index]
        
    def __len__(self):
        return self.size

    def __repr__(self) -> str:
        return f"Batched_QM9(batch_size={self.batch_size}, size={self.size})"
    
    def processed_file_names(self) -> str:
        return 'data_v3.pt'

    def reshuffle_grouped_dataset(self):
        for _, group in self.grouped_data:
            random.shuffle(group)
        self.batched_data = batch_same_size(self.grouped_data, self.batch_size)

if __name__ == '__main__':
    pass