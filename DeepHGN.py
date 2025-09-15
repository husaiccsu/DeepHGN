import os
import warnings
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import networkx as nx
import torch
from node2vec import Node2Vec
import random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import sys
from sklearn.model_selection import KFold
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from collections import Counter
from transformers import EsmModel, EsmTokenizer
from transformers import EsmForMaskedLM 
from functools import reduce

proteinlist=[]
PPMatrix=np.empty_like([])
PGMatrix=np.empty_like([]) 
GoList=[]
PGONum=[]
GPList=[]
GOType='F'
Species='Saccharomyces_cerevisiae'
#Species='Homo_sapiens'

def normalize_matrix(matrix):
    min_val = matrix.min()
    max_val = matrix.max()
    if max_val - min_val > 0:
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
    else:
        normalized_matrix = matrix
    return normalized_matrix

def max_path_product(G, source, target):
    all_paths = list(nx.all_simple_paths(G, source, target))
    if not all_paths:
        return 0    
    products = [reduce(lambda x, y: x * y, [G[u][v]['weight'] for u, v in zip(path, path[1:])], 1) for path in all_paths]
    max_product=max(products)
    return max_product

def build_heterogeneous_network():
    num_proteins = len(proteinlist)
    num_go = len(GoList)  
    PPI_adj = sp.lil_matrix((num_proteins, num_proteins))
    for i in range(num_proteins):
        neighbors_i = set(np.where(PPMatrix[i] > 0)[0])
        for j in range(i+1, num_proteins):
            if PPMatrix[i][j] > 0:
                neighbors_j = set(np.where(PPMatrix[j] > 0)[0])
                intersection = len(neighbors_i & neighbors_j)
                union = len(neighbors_i | neighbors_j)
                if union >=0:
                    jaccard = intersection / union
                    #PPI_adj[i,j] = jaccard
                    PPI_adj[i,j] =PPMatrix[i][j]
                    PPI_adj[j,i] = PPI_adj[i,j]
    PG_adj = PGMatrix.copy()    
    GO_GO_adj = sp.lil_matrix((num_go, num_go))
    G = readDAG()     
    ic_values = np.zeros(num_go)
    for i in range(num_go):
        p_g = np.sum(PGMatrix[:, i])
        ic_values[i] = -np.log(p_g / num_proteins) if p_g > 0 else 0    
    GO_GO_adj = GO_GO_adj.tocsr()    
    for i in range(num_go):        
        print(i,'%',num_go,end='\r')
        if GoList[i] not in G:
            continue           
        for j in range(i+1, num_go):
            if GoList[j] not in G:
                continue                
            try:
                ic=max_path_product(G, GoList[i],GoList[j])                
                GO_GO_adj[i, j] = ic
                GO_GO_adj[j, i] = ic
            except (nx.NetworkXError, ValueError):
                GO_GO_adj[i, j] = 0
                GO_GO_adj[j, i] = 0   
    GO_GO_adj = GO_GO_adj.tolil()    
    hetero_adj = sp.bmat([[PPI_adj, sp.csr_matrix(PG_adj)], [sp.csr_matrix(PG_adj).T, GO_GO_adj]])
    hetero_adj=normalize_matrix(hetero_adj)     
    return hetero_adj, GO_GO_adj                    

def heterogeneous_hits(hetero_adj, GO_GO_adj,alpha=0.3, max_iter=100, tolerance=1e-6):
    num_proteins = len(proteinlist)
    num_go = len(GoList)
    hetero_adj = hetero_adj.tocsr()  
    protein_auth = np.zeros(num_proteins)
    protein_hub = np.zeros(num_proteins)
    degrees = np.sum(PPMatrix, axis=1)
    total_degree = np.sum(degrees)
    go_counts = np.sum(PGMatrix, axis=1)
    total_go_counts = np.sum(go_counts)  
    for i in range(num_proteins):
        protein_auth[i] = degrees[i] / total_degree if total_degree > 0 else 0
        protein_hub[i] = go_counts[i] / total_go_counts if total_go_counts > 0 else 0   
    go_auth = np.zeros(num_go)
    go_hub = np.zeros(num_go)    
    protein_counts = np.sum(PGMatrix, axis=0)
    total_protein_counts = np.sum(protein_counts)    
    similarity_sums = np.asarray(np.sum(GO_GO_adj, axis=1)).flatten()
    total_similarity = np.sum(similarity_sums)
    for j in range(num_go):
        go_auth[j] = protein_counts[j] / total_protein_counts if total_protein_counts > 0 else 0
        go_hub[j] = similarity_sums[j] / total_similarity if total_similarity > 0 else 0    
    authority = np.concatenate([protein_auth, go_auth])
    hub = np.concatenate([protein_hub, go_hub])    
    authority = authority / np.linalg.norm(authority)
    hub = hub / np.linalg.norm(hub)
    initial_auth = authority.copy()
    initial_hub = hub.copy()

    for i in range(max_iter):
        old_authority = authority.copy()
        old_hub = hub.copy()     
        authority = alpha * hetero_adj.T.dot(hub) + (1 - alpha) * initial_auth
        authority /= np.linalg.norm(authority, ord=1)
        hub = alpha * hetero_adj.T.dot(authority) + (1 - alpha) * initial_hub
        hub /= np.linalg.norm(hub, ord=1)        
        authority_diff = np.linalg.norm(authority - old_authority)
        hub_diff = np.linalg.norm(hub - old_hub)        
        if authority_diff < tolerance and hub_diff < tolerance:
            break  
    authority_p = authority[:num_proteins]
    authority_g = authority[num_proteins:]
    hub_p = hub[:num_proteins]
    hub_g = hub[num_proteins:]  
    return authority_p, authority_g, hub_p, hub_g
    
def ordered_unique(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]
    
def LoadData():
  global proteinlist
  TempList=[]
  CAFAList=[]
  with open('./data/'+Species+'_CAFA3.txt', 'r') as file:
    for line in file:
      line = line.strip()
      beginstr,endstr,types=line.split('\t')
      if types!=GOType:
         continue      
      if beginstr not in TempList:
         TempList.append(beginstr)

  PList=[]
  with open('./data/'+Species+'_PPI_StringV12.txt', 'r') as file:
    for line in file:
      line = line.strip()
      beginstr,endstr,scores=line.split('\t')
      if beginstr not in PList:
         PList.append(beginstr)
      if endstr not in PList:
         PList.append(endstr)        

    list1 = ordered_unique(PList)  
    list2 = ordered_unique(TempList) 
    proteinlist = [item for item in list1 if item in list2] 
    listlen=len(proteinlist)
    file.seek(0)
    global PPMatrix
    PPMatrix = np.zeros((listlen,listlen))
    global NList
    NList=[[0 for j in range(0)] for i in range(listlen)]
    global N2List
    N2List=[[0 for j in range(0)] for i in range(listlen)]
    for line in file:
        line = line.strip()
        beginstr,endstr,scores=line.split('\t')
        if (beginstr not in TempList) or (endstr not in TempList):
          continue
        Ipos=proteinlist.index(beginstr)
        JPos=proteinlist.index(endstr)
        PPMatrix[Ipos][JPos]=scores
        PPMatrix[JPos][Ipos]=scores  

  with open('./data/'+Species+'_CAFA3.txt', 'r') as file:
    for line in file:
      line = line.strip()
      beginstr,endstr,types=line.split('\t')
      if types!=GOType:
         continue
      if beginstr not in proteinlist:
         continue 
      global GoList
      if endstr not in GoList:
         GoList.append(endstr)          
         
  global PGMatrix 
  PGMatrix=np.zeros((listlen,len(GoList)))
  global PGONum
  PGONum=[0 for i in range(listlen)] 
  global GPList
  GPList=[[0 for j in range(0)] for i in range(len(GoList))]
  with open('./data/'+Species+'_CAFA3.txt', 'r') as file:
   for line in file:
      line = line.strip()
      beginstr,endstr,types=line.split('\t')
      try:
          Ipos=proteinlist.index(beginstr)
      except ValueError:
          Ipos=-1
      if types!=GOType:
         continue    
      if Ipos==-1:
         continue     
      JPos=GoList.index(endstr)
      PGMatrix[Ipos][JPos]=1
      PGONum[Ipos]=PGONum[Ipos]+1
      GPList[JPos].append(Ipos)         
  
def GetESMSequenceFeature():
    save_file = './data/'+Species+'_'+GOType+'_sequence_features_esm1b.npz' 
    if not os.path.exists(save_file):
        sequences_dict = read_sequences()
        full_features = np.zeros((len(proteinlist), 1280))          
        valid_ids = [pid for pid in proteinlist if pid in sequences_dict]
        valid_indices = [proteinlist.index(pid) for pid in valid_ids]        
        model_dir = "./local_models/esm1b_t33_650M_UR50S"
        tokenizer = EsmTokenizer.from_pretrained(model_dir)
        model = EsmForMaskedLM.from_pretrained(model_dir).esm.eval().cpu()
        batch_size = 4
        features = []
        for i in range(0, len(valid_ids), batch_size):
            print(i+1,'|',len(valid_ids),end='\r')
            batch_ids = valid_ids[i:i+batch_size]
            batch_seqs = [sequences_dict[pid] for pid in batch_ids]            
            MAX_SEQ_LENGTH = 1024
            batch_seqs = [seq[:MAX_SEQ_LENGTH] for seq in batch_seqs]           
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True,  truncation=True,  max_length=MAX_SEQ_LENGTH).to(model.device)           
            with torch.no_grad():
                outputs = model(**inputs)
                residue_embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]                
                mask_expanded = attention_mask.unsqueeze(-1).expand(residue_embeddings.size()).float()
                sum_embeddings = torch.sum(residue_embeddings * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)               
                protein_embeddings = sum_embeddings / sum_mask
                features.append(protein_embeddings.cpu().numpy())
        if features:
            valid_features = np.vstack(features)
            for idx, feat in zip(valid_indices, valid_features):
                full_features[idx] = feat        
        np.savez(save_file, features=full_features, protein_ids=np.array(proteinlist))    
    data = np.load(save_file)
    features = data['features']
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return {'features': normalized_features,'protein_ids': data['protein_ids']}

def clean_sequence(seq):
    INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', '*'])
    return ''.join([aa if aa not in INVALID_ACIDS else 'X' for aa in seq])

def read_sequences_with_cleaning():
    sequences = {}
    file_path = f'./data/{Species}_Sequence_Uniprot.txt'
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                protein_id = parts[0]
                sequence = parts[1]
                sequence = clean_sequence(sequence)
                if protein_id in proteinlist:
                    sequences[protein_id] = sequence
    return sequences
    
def read_sequences():
    sequences = {}
    file_path = './data/'+Species+'_Sequence_Uniprot.txt'
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) != 2:
                    continue 
                protein_id = parts[0]
                sequence = parts[1]
                if protein_id not in proteinlist:
                    continue
                sequences[protein_id] = sequence
    return sequences

def readDAG():
    n_GO=len(GoList) 
    GO_DAG_Matrix = np.zeros((n_GO, n_GO))
    G=nx.DiGraph()
    with open('./data/'+'Total_DAG.txt', 'r') as file:  
     for line in file:
      line = line.strip()
      beginstr,endstr,types,relation=line.split('\t')
      if beginstr not in GoList or endstr not in GoList:
         continue
      if relation!='is_a' and relation!='part_of':
         continue
      if types!=GOType:
         continue       
      if  relation=='is_a':        
        G.add_edge(beginstr,endstr,weight=0.4)
      else:
        G.add_edge(beginstr,endstr,weight=0.3)  
    return G

def GetGOFeature():
    save_file = './data/'+Species+'_'+GOType+'_GO_features.npz' 
    if not os.path.exists(save_file):
        G = readDAG()        
        subgraph_nodes = [node for node in GoList if node in G]
        G_sub = G.subgraph(subgraph_nodes).copy()      
        missing_nodes = set(GoList) - set(subgraph_nodes)
        for node in missing_nodes:
            G_sub.add_node(node)
        dimensions = 256
        go_embedding = dag2vec_embedding(G_sub, GoList, dimensions)       
        np.savez(save_file, features=go_embedding)
    else:
        data = np.load(save_file)
        go_embedding = data['features']    
    return go_embedding

def dag2vec_embedding(G, go_list, dimensions=256):
    node_to_idx = {node: idx for idx, node in enumerate(go_list)}
    embedding = np.zeros((len(go_list), dimensions))   
    depth_groups = {} 
    roots = [node for node, deg in G.in_degree() if deg == 0]   
    depth_map = {}  
    for node in G.nodes():
        depth_map[node] = 0    
    for node in nx.topological_sort(G):
        for predecessor in G.predecessors(node):         
            if depth_map[predecessor] + 1 > depth_map[node]:
                depth_map[node] = depth_map[predecessor] + 1
        depth_groups.setdefault(depth_map[node], []).append(node)    
    for node in G.nodes():
        if G.in_degree(node) == 0 and G.out_degree(node) == 0:
            depth_groups.setdefault(0, []).append(node)
       
    for depth in sorted(depth_groups.keys()):
        layer_nodes = depth_groups[depth]       
        layer_graph = nx.DiGraph()
        for node in layer_nodes:    
            for parent in G.predecessors(node):
                if parent in G: 
                    weight = G[parent][node].get('weight', 0.4)
                    layer_graph.add_edge(parent, node, weight=weight)            
            for child in G.successors(node):
                if child in G: 
                    weight = G[node][child].get('weight', 0.4)
                    layer_graph.add_edge(node, child, weight=weight)       
        if layer_graph.number_of_edges() == 0:
            for node in layer_nodes:
                layer_graph.add_edge(node, node, weight=1.0)       
        if len(layer_graph.nodes) > 0:           
            p_value = 1.0 / (depth + 1)  
            q_value = 0.5 + depth * 0.1             
            node2vec = Node2Vec(layer_graph,dimensions=dimensions,walk_length=min(30, len(layer_graph.nodes)*2), 
                num_walks=min(15, len(layer_graph.nodes)),p=p_value,q=q_value,weight_key='weight',quiet=True)                      
            try: 
                model = node2vec.fit(window=min(8, len(layer_graph.nodes)),min_count=1,workers=2)
            except Exception as e:               
                for node in layer_graph.nodes:
                    idx = node_to_idx[node]
                    embedding[idx] = np.random.normal(scale=0.6, size=(dimensions,))
                continue                    
            for node in layer_graph.nodes: 
                if hasattr(model, 'wv') and node in model.wv:
                    idx = node_to_idx[node]
                    embedding[idx] = model.wv[node]
                else:                 
                    idx = node_to_idx[node]
                    embedding[idx] = np.random.normal(scale=0.6, size=(dimensions,))        
    for node in G.nodes:
        if node in node_to_idx:
            idx = node_to_idx[node]       
            try: 
                ancestors = list(nx.ancestors(G, node))
                if ancestors:                   
                    if len(ancestors) > 50:
                        ancestors = random.sample(ancestors, 50)                    
                    ancestor_embs = []
                    for a in ancestors:
                        if a in node_to_idx:
                            ancestor_embs.append(embedding[node_to_idx[a]])                    
                    if ancestor_embs:
                        embedding[idx] += np.mean(ancestor_embs, axis=0) * 0.3
            except nx.NetworkXError:
                pass                       
            try: 
                descendants = list(nx.descendants(G, node))
                if descendants:                   
                    if len(descendants) > 30:
                        descendants = random.sample(descendants, 30)                   
                    descendant_embs = []
                    for d in descendants:
                        if d in node_to_idx:
                            descendant_embs.append(embedding[node_to_idx[d]])                  
                    if descendant_embs:
                        embedding[idx] += np.mean(descendant_embs, axis=0) * 0.2
            except nx.NetworkXError:
                pass   
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1  
    embedding = embedding / norms   
    return embedding

def predict(model, X_all_tensor, hetero_adj_tensor):
    model.eval()
    with torch.no_grad():
        row_sum = torch.sum(hetero_adj_tensor, dim=1)
        row_sum_inv_sqrt = torch.pow(row_sum, -0.5)
        row_sum_inv_sqrt[torch.isinf(row_sum_inv_sqrt)] = 0.
        D_inv_sqrt = torch.diag(row_sum_inv_sqrt)
        normalized_adj = torch.mm(torch.mm(D_inv_sqrt, hetero_adj_tensor), D_inv_sqrt)
        adj_with_selfloops = normalized_adj + torch.eye(normalized_adj.shape[0])
        edge_index = adj_with_selfloops.nonzero().t()
        predictions = model(X_all_tensor, edge_index)
        predictions = torch.clamp(predictions, min=-10, max=10)  
    return predictions    

def incorporate_supervision(protein_emb, PGMatrix_full_tensor, authority_p_tensor, hub_p_tensor, authority_g_tensor, train_indices, tele_prob=0.2):
    num_proteins = protein_emb.size(0)    
    go_weights = PGMatrix_full_tensor / (PGMatrix_full_tensor.sum(dim=1, keepdim=True) + 1e-8)
    if authority_g_tensor.dim() == 1:
        authority_g_tensor = authority_g_tensor.unsqueeze(1)
    go_signals = torch.mm(go_weights, authority_g_tensor)   
    hits_weights = (authority_p_tensor + hub_p_tensor).unsqueeze(1)
    supervised_emb = protein_emb.clone()
    supervised_emb[train_indices] = (protein_emb[train_indices] * (1 - tele_prob) + go_signals[train_indices] * tele_prob * hits_weights[train_indices])
    return supervised_emb

def final_prediction(embeddings, S_protein, authority, hub, alpha=0.15, iterations=10, tolerance=1e-6):
    num_nodes = embeddings.size(0)
    device = embeddings.device    
    row_sum = S_protein.sum(dim=1, keepdim=True)
    row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
    S_norm = S_protein / row_sum    
    auth_norm = authority / (authority.max() + 1e-8)
    hub_norm = hub / (hub.max() + 1e-8)
    hits_state = (auth_norm + hub_norm).unsqueeze(1) * embeddings    
    current = hits_state.clone()
    for _ in range(iterations):
        prev = current
        current = alpha * hits_state + (1 - alpha) * torch.mm(S_norm, current)
        if torch.norm(current - prev) < tolerance:
            break   
    return current   
     
def FeatureCombination():
    hetero_adj, GO_GO_adj = build_heterogeneous_network()
    authority_p, authority_g, hub_p, hub_g = heterogeneous_hits(hetero_adj, GO_GO_adj)
    num_proteins = len(proteinlist)     
    X_sequence = GetESMSequenceFeature()
    X_sequence_features = X_sequence['features']
    X_go = GetGOFeature()    
    seq_feature_dim = X_sequence_features.shape[1]
    go_feature_dim = X_go.shape[1]
    X_protein=X_sequence_features.copy() 
    go_ancestor_features = np.zeros((len(GoList), X_go.shape[1]))
    G = readDAG()
    for i, go_id in enumerate(GoList):
        if go_id in G:
            ancestors = list(nx.ancestors(G, go_id))
            ancestor_indices = [GoList.index(anc) for anc in ancestors if anc in GoList]
            if ancestor_indices:
                go_ancestor_features[i] = np.mean(X_go[ancestor_indices], axis=0)    
    X_go = np.concatenate([X_go, go_ancestor_features], axis=1)    
    protein_scaler = StandardScaler()
    X_protein = protein_scaler.fit_transform(X_protein)    
    go_scaler = StandardScaler()
    X_go = go_scaler.fit_transform(X_go)    
    total_dim = X_protein.shape[1] + X_go.shape[1]
    X_all = np.zeros((num_proteins + len(GoList), total_dim))
    X_all[:num_proteins, :X_protein.shape[1]] = X_protein
    X_all[num_proteins:, X_protein.shape[1]:] = X_go    
    X_all += np.random.normal(0, 1e-5, X_all.shape)    
    return X_all, hetero_adj, authority_p, authority_g, hub_p, hub_g

def HITSGCN(X_all,hetero_adj,authority_p,authority_g,hub_p,hub_g,train_index, test_index):
    S_protein = authority_p + hub_p 
    num_go_terms = len(GoList)
    num_proteins = len(proteinlist)
    X_all_tensor = torch.tensor(X_all, dtype=torch.float)    
    hetero_adj_csr = hetero_adj.tocsr()
    ppi_adj = hetero_adj_csr[:num_proteins, :num_proteins]
    ppi_adj_coo = ppi_adj.tocoo()   
    indices = torch.tensor(np.vstack((ppi_adj_coo.row, ppi_adj_coo.col)), dtype=torch.long)
    values = torch.tensor(ppi_adj_coo.data, dtype=torch.float)
    shape = torch.Size(ppi_adj_coo.shape)    
    S_matrix = torch.sparse_coo_tensor(indices, values, shape).coalesce()    
    row_sum = torch.sparse.sum(S_matrix, dim=1).to_dense()
    row_sum_inv_sqrt = torch.pow(row_sum, -0.5)
    row_sum_inv_sqrt[torch.isinf(row_sum_inv_sqrt)] = 0   
    D_inv_sqrt = torch.diag(row_sum_inv_sqrt)
    norm_S = torch.sparse.mm(torch.sparse.mm(D_inv_sqrt, S_matrix), D_inv_sqrt)    
    identity = torch.eye(num_proteins, dtype=torch.float)
    adj_with_selfloops = norm_S.to_dense() + identity  
    edge_index = adj_with_selfloops.nonzero().t().contiguous()   
    S_protein_tensor = torch.tensor(S_protein, dtype=torch.float32)
    authority_p_tensor = torch.tensor(authority_p, dtype=torch.float32)
    hub_p_tensor = torch.tensor(hub_p, dtype=torch.float32)
    authority_g_tensor = torch.tensor(authority_g, dtype=torch.float32)
    train_indices = torch.tensor(train_index, dtype=torch.long)
    class HITSGCNModel(nn.Module):
      def __init__(self, in_dim, hidden_dim, num_go_terms):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)        
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)        
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)        
        self.conv5 = GCNConv(hidden_dim, hidden_dim//2)
        self.bn5 = nn.BatchNorm1d(hidden_dim//2)        
        self.att = nn.Linear(hidden_dim//2, 1)        
        self.res1 = nn.Linear(in_dim, hidden_dim)
        self.res3 = nn.Linear(hidden_dim, hidden_dim)
        self.res5 = nn.Linear(in_dim, hidden_dim//2)       
        self.predict_layer = nn.Sequential(nn.Linear(hidden_dim//2, 256),nn.ReLU(),nn.Dropout(0.4),nn.Linear(256, num_go_terms) )

      def forward(self, x, edge_index):
        identity1 = self.res1(x)
        x1 = F.leaky_relu(self.bn1(self.conv1(x, edge_index)), negative_slope=0.2)
        x1 = x1 + identity1[:x1.size(0)]
        x2 = F.leaky_relu(self.bn2(self.conv2(x1, edge_index)), negative_slope=0.2)
        identity3 = self.res3(x1)
        x3 = F.leaky_relu(self.bn3(self.conv3(x2, edge_index)), negative_slope=0.2)
        x3 = x3 + identity3
        x4 = F.leaky_relu(self.bn4(self.conv4(x3, edge_index)), negative_slope=0.2)
        identity5 = self.res5(x)
        x5 = F.leaky_relu(self.bn5(self.conv5(x4, edge_index)), negative_slope=0.2)
        x5 = x5 + identity5[:x5.size(0)]
        att_weights = torch.sigmoid(self.att(x5))
        x_att = x5 * att_weights       
        return self.predict_layer(x_att)
      
    model = HITSGCNModel(X_all.shape[1], 256, num_go_terms)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)        
    max_grad_norm = 1.0
    pos_weight = torch.tensor([5.0] * num_go_terms)  
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()        
        all_output = model(X_all_tensor, edge_index)
        protein_output = all_output[:num_proteins]        
        supervised_emb = incorporate_supervision( protein_output,torch.tensor(PGMatrix, dtype=torch.float32),authority_p_tensor,hub_p_tensor,authority_g_tensor,train_indices)        
        propagated_emb = final_prediction(supervised_emb,norm_S[:num_proteins, :num_proteins], authority_p_tensor,hub_p_tensor)        
        train_labels = torch.tensor(PGMatrix[train_index], dtype=torch.float32)
        loss = criterion(protein_output[train_indices], train_labels)        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()    
    model.eval()
    with torch.no_grad():
        test_pred = model(X_all_tensor, edge_index)[:num_proteins][test_index]
        return test_pred.cpu().numpy() 
    
def evaluate_predictions(true_labels, predictions,true_go_terms):
    pred_z = (predictions - np.mean(predictions)) / np.std(predictions)
    pred_normalized = (pred_z - np.min(pred_z)) / (np.max(pred_z) - np.min(pred_z)) 
    sorted_indices = np.argsort(-pred_normalized, axis=1)
    sorted_scores = np.array([pred_normalized[i][indices] for i, indices in enumerate(sorted_indices)])
    sorted_labels = np.array([true_labels[i][indices] for i, indices in enumerate(sorted_indices)])   
    best_fmax = 0.0
    best_threshold = 0.0
    for threshold in np.linspace(0, 1, 1000)[::-1]:
        binary_preds = (sorted_scores >= threshold).astype(int)
        tp = np.sum((binary_preds == 1) & (sorted_labels == 1))
        fp = np.sum((binary_preds == 1) & (sorted_labels == 0))
        fn = np.sum((binary_preds == 0) & (sorted_labels == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0  
        if f1 > best_fmax:
            best_fmax = f1
            best_threshold = threshold        
    y_true = true_labels.flatten()
    y_score = pred_normalized.flatten()
    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    return best_fmax, auroc, aupr

def single_evaluate(methods,true_labels, predictions,true_go_terms):
    fmax, auroc, aupr = evaluate_predictions(true_labels, predictions,true_go_terms)
    s="{:<15} {:<10.3f} {:<10.3f} {:<10.3f}".format(methods, fmax, auroc, aupr)
    print("{:<15} {:<10.3f} {:<10.3f} {:<10.3f}".format(methods, fmax, auroc, aupr))
 
def ten_fold_cross_validation():
    global IsTenFold
    IsTenFold=1
    print("Loading Data:"+Species+'_'+GOType, datetime.now().strftime('%Y-%m-%d %H:%M:%S')) 
    LoadData()  
    method_results = {}
    kfnum=10
    kf = KFold(n_splits=kfnum, shuffle=True, random_state=42) 
    print('KFold Num:',kfnum)   
    splits = list(kf.split(proteinlist))    
    all_predictions_HITSGCN = np.zeros((len(proteinlist), len(GoList)))   
    X_all,hetero_adj,authority_p,authority_g,hub_p,hub_g=FeatureCombination()           
    for fold_idx, (train_index, test_index) in enumerate(splits):        
        print(f'Fold {fold_idx+1}/10') 
        s='The '+str(fold_idx)+' Fold'    
        tenfold_labels=PGMatrix[test_index, : ] 
        true_go_terms = {}
        for i in range(len(proteinlist)):
          if  i not in (test_index):
            continue
          protein_id =proteinlist[i]
          associated_go_terms = [GoList[j] for j in range(len(GoList)) if PGMatrix[i][j] == 1]
          true_go_terms[protein_id] = associated_go_terms        
        test_predictions_HITSGCN = HITSGCN(X_all, hetero_adj, authority_p, authority_g, hub_p, hub_g,train_index, test_index)
        all_predictions_HITSGCN[test_index] = test_predictions_HITSGCN
        single_evaluate('HITSGCN',tenfold_labels,test_predictions_HITSGCN,true_go_terms) 
    print("Performance Metrics:") 
    single_evaluate('HITSGCN',PGMatrix,all_predictions_HITSGCN,true_go_terms) 
    print('Over', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
def main():
   ten_fold_cross_validation()
   
if __name__ == "__main__":
	main()