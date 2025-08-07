import os
import time
import numpy as np
import scanpy as sc
import scipy.sparse as sps
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import LinearEncoder, NonlinearEncoder, Classifier
from losses import InfoNCE, L1Regularization, non_corr, zero_center, max_var
from utils import *
from dataset import ClsDataset, get_pos_ind

class scSHEFT:
    
    def __init__(self,
                 encoder_type='linear',
                 use_struct=False,
                 n_latent=64,
                 bn=False,
                 dr=0.2,
                 l1_w=0.1,
                 ortho_w=0.1,
                 cont_w=0.0,
                 cont_tau=0.4,
                 cont_cutoff=0.,
                 align_w=0.2,
                 align_p=0.8,
                 align_cutoff=0.,
                 anchor_w=0.0,
                 anchor_sample_ratio=0.10,
                 anchor_top_ratio=0.10,
                 center_w=0.0,
                 center_cutoff=0.,
                 stc_w=0.0,
                 stc_cutoff=0.,
                 momentum=0.9,
                 ce_w=1.0,
                 seed=1234):
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        
        # Model parameters
        self.n_latent = n_latent
        self.encoder_type = encoder_type
        self.use_struct = use_struct
        self.bn = bn
        self.dr = dr
        self.l1_w = l1_w
        self.ortho_w = ortho_w
        self.cont_w = cont_w
        self.cont_tau = cont_tau
        self.cont_cutoff = cont_cutoff
        self.align_w = align_w
        self.anchor_w = anchor_w
        self.center_w = center_w
        self.center_cutoff = center_cutoff
        self.momentum = momentum
        self.stc_w = stc_w
        self.stc_cutoff = stc_cutoff
        self.align_p = align_p
        self.align_cutoff = align_cutoff
        self.ce_w = ce_w
        self.anchor_sample_ratio = anchor_sample_ratio
        self.anchor_top_ratio = anchor_top_ratio
        
        # Evaluation results record
        self.evaluation_results = []
    
    def preprocess(self, exp_id, adata_inputs, target_raw_emb, pp_dict, 
                   adata_adt_inputs=None, stc_emb_inputs=False):
        source_adata = adata_inputs[0].copy()
        target_adata = adata_inputs[1].copy()
        
        # Process structural information
        if self.use_struct:
            if not stc_emb_inputs:
                raw_target_adata = adata_inputs[2].copy()
                sc.tl.pca(raw_target_adata, n_comps=500, svd_solver="auto")
                self.struct_target_emb = raw_target_adata.obsm['X_pca']
                # Save structural PCA embedding to data directory
                save_path = f'./data/{exp_id}_struct_pca.npy'
                np.save(save_path, self.struct_target_emb)
            else:
                # Load pre-computed structural PCA embedding
                load_path = f'./data/{exp_id}_struct_pca.npy'
                self.struct_target_emb = np.load(load_path)
        
        n_source, n_target = source_adata.shape[0], target_adata.shape[0]
        self.type_label = pp_dict['type_label']
        
        # Data preprocessing
        source_adata, target_adata = preprocess_data(
            source_adata, target_adata, 
            hvg_num=pp_dict['hvg_num'],
            binz=pp_dict['binz'],
            lognorm=pp_dict['lognorm'],
            scale_per_batch=pp_dict['scale_per_batch']
        )
        
        # Convert to sparse matrix
        self.source_data = sps.csr_matrix(source_adata.X)
        self.target_data = sps.csr_matrix(target_adata.X)
        self.target_embeddings = target_raw_emb
        
        # Process ADT data
        if adata_adt_inputs is not None:
            print('Concating adt features...')
            adt_matrices = [sps.csr_matrix(adata_adt_inputs[i].X) for i in range(2)]
            self.source_data = sps.csr_matrix(sps.hstack([self.source_data, adt_matrices[0]]))
            self.target_data = sps.csr_matrix(sps.hstack([self.target_data, adt_matrices[1]]))
        
        self.n_input = self.source_data.shape[1]
        self.n_source, self.n_target = n_source, n_target
        self.source_metadata = source_adata.obs.copy()
        self.target_metadata = target_adata.obs.copy()
        
        # Label processing
        source_labels = self.source_metadata[self.type_label].values
        self.relabel(source_labels)
        
        # Data shuffling and KNN
        self.shuffle_data()
        self.get_nns(pp_dict['knn'])
        
        # Label mapping
        source_label_names = source_adata.obs[self.type_label]
        source_label_ranks = source_label_names.rank(method="dense", ascending=True).astype(int) - 1
        self.source_label_names = source_label_names.values
        self.source_label_ranks = source_label_ranks.values
        
        self.label_map = {k: source_label_names[source_label_ranks == k][0] for k in range(source_label_ranks.max() + 1)}
        
        self.type_num = len(self.label_map)
        self.source_prototypes = torch.zeros(self.type_num, self.n_latent).to(self.device)
        self.target_prototypes = torch.zeros(self.type_num, self.n_latent).to(self.device)
        
        # Find anchor points
        if self.anchor_w != 0:
            # Compute PCA embeddings
            for adata, name in [(source_adata, 'source'), (target_adata, 'target')]:
                sc.tl.pca(adata, n_comps=self.target_embeddings.shape[1], svd_solver="auto")
            
            source_emb, target_emb = source_adata.obsm['X_pca'], target_adata.obsm['X_pca']
            
            print("Finding anchors...")
            start_time = time.time()
            self.A_B, self.B_A, self.anchor_set = find_anchor_points(source_emb, target_emb)
            end_time = time.time()
            print(f"Finding anchors executed in {end_time - start_time} seconds.")
            
    
    def init_train(self, opt, lr, lr2, weight_decay):
        encoder_class = LinearEncoder if self.encoder_type == 'linear' else NonlinearEncoder
        encoder_args = (self.n_input, self.n_latent) if self.encoder_type == 'linear' else (self.n_input, self.n_latent, self.bn, self.dr)
        
        self.encoder = torch.nn.DataParallel(encoder_class(*encoder_args).to(self.device))
        self.head = torch.nn.DataParallel(Classifier(self.n_latent, self.n_class).to(self.device))
        
        if self.use_struct:
            self.stc_encoder = torch.nn.DataParallel(
                NonlinearEncoder(self.struct_target_emb.shape[1], self.n_latent, 
                               self.bn, self.dr).to(self.device))
        
        # Optimizer
        optimizer_class = optim.Adam if opt == 'adam' else optim.SGD
        optimizer_kwargs = {'lr': lr, 'weight_decay': weight_decay, 'momentum': 0.9 if opt == 'sgd' else None}
        optimizer_kwargs = {k: v for k, v in optimizer_kwargs.items() if v is not None}
        
        optimizer_G = optimizer_class(self.encoder.parameters(), **optimizer_kwargs)
        optimizer_C = optimizer_class(self.head.parameters(), 
                                    lr=lr2 if lr2 is not None else lr, 
                                    **{k: v for k, v in optimizer_kwargs.items() if k != 'lr'})
        
        return optimizer_G, optimizer_C
    
    def compute_alignment_loss(self, source_features, target_features, top_ratio=0.8):
        batch_size = target_features.size(0)
        source_normalized = F.normalize(source_features, p=2, dim=1)
        target_normalized = F.normalize(target_features, p=2, dim=1)
        
        source_detached = source_normalized.detach()
        target_detached = target_normalized.detach()
        
        cosine_similarity = torch.matmul(target_detached, source_detached.t())
        similarity_values, similarity_indices = torch.max(cosine_similarity, dim=1)
        _, top_target_indices = torch.topk(similarity_values, int(batch_size * top_ratio))
        top_source_indices = similarity_indices[top_target_indices]
        
        top_target_features = target_normalized[top_target_indices]
        top_source_features = source_normalized[top_source_indices]
        
        return -torch.mean(torch.sum(top_source_features * top_target_features, dim=1))
    
    def train_step(self, step, batch_size, optimizer_G, optimizer_C, cls_crit, reg_crit, 
                   reg_cont, criterion, log_step=100, eval_target=False, eval_top_k=1):
        """Training step"""
        self.encoder.train()
        self.head.train()
        
        N_A, N_B = self.n_source, self.n_target
        
        # Sample data
        source_indices = np.random.choice(np.arange(N_A), size=batch_size)
        source_data = torch.from_numpy(self.source_data_shuffle[source_indices, :].A).float().to(self.device)
        source_labels = torch.from_numpy(self.source_label_ids_shuffle[source_indices]).long().to(self.device)
        
        target_indices = np.random.choice(np.arange(N_B), size=batch_size)
        target_data = torch.from_numpy(self.target_data_shuffle[target_indices, :].A).float().to(self.device)
        
        if self.use_struct:
            target_struct_data = torch.from_numpy(self.struct_target_emb[target_indices, :]).float().to(self.device)
        
        # Forward propagation
        source_features = self.encoder(source_data)
        source_predictions = self.head(source_features)
        
        target_features = self.encoder(target_data)
        
        target_struct_features = self.stc_encoder(target_struct_data) if self.use_struct else None
        target_features = target_features * self.momentum + target_struct_features * (1 - self.momentum) if self.use_struct else target_features
        
        target_predictions = self.head(target_features)
        target_pred_labels = np.argmax(target_predictions.detach().cpu().numpy(), axis=1)
        
        optimizer_G.zero_grad()
        optimizer_C.zero_grad()
        
        # Reduction loss
        source_center_loss = zero_center(source_features)
        source_corr_loss = non_corr(source_features)
        source_var_loss = max_var(source_features)
        
        target_center_loss = zero_center(target_features)
        target_corr_loss = non_corr(target_features)
        target_var_loss = max_var(target_features)
        
        reduction_loss = source_center_loss + target_center_loss + source_corr_loss + target_corr_loss + target_var_loss
        
        if self.use_struct:
            target_struct_center_loss = zero_center(target_struct_features)
            target_struct_corr_loss = non_corr(target_struct_features)
            target_struct_var_loss = max_var(target_struct_features)
            reduction_loss += target_struct_center_loss + target_struct_corr_loss + target_struct_var_loss
        
        # Contrastive learning loss
        contrastive_loss = 0
        if self.cont_w != 0 and (step >= self.cont_cutoff):
            target_pos_indices = get_pos_ind(target_indices, self.knn_ind)
            target_pos_data = torch.from_numpy(self.target_data_shuffle[target_pos_indices, :].A).float().to(self.device)
            target_pos_features = self.encoder(target_pos_data)
            contrastive_loss = reg_cont(target_features, target_pos_features)
        
        # Alignment loss
        align_loss = self.compute_alignment_loss(source_features, target_features, self.align_p) if (self.align_w != 0) and (step >= self.align_cutoff) else 0.
        
        # Structural information alignment loss
        stc_align_loss = self.compute_alignment_loss(source_features, target_struct_features, self.align_p) if (self.stc_w != 0) and (step >= self.stc_cutoff) and self.use_struct else 0.
        
        # Anchor loss
        anchor_align_loss = 0.
        if self.anchor_w != 0:
            # Sample anchor pairs
            anchor_pairs = list(self.anchor_set)
            num_anchors = int(len(anchor_pairs) * self.anchor_sample_ratio)
            selected_anchors = [anchor_pairs[i] for i in np.random.choice(len(anchor_pairs), size=num_anchors, replace=False)]
            
            # Extract anchor indices
            anchor_indices = np.array(selected_anchors)
            source_anchor_indices, target_anchor_indices = anchor_indices[:, 0], anchor_indices[:, 1]
            
            # Encode anchor features
            anchor_data = [
                torch.from_numpy(self.source_data_shuffle[source_anchor_indices, :].toarray()).float().to(self.device),
                torch.from_numpy(self.target_data_shuffle[target_anchor_indices, :].toarray()).float().to(self.device)
            ]
            anchor_features = [self.encoder(data) for data in anchor_data]
            anchor_normalized = [F.normalize(feat, p=2, dim=1) for feat in anchor_features]
            
            anchor_similarities = F.cosine_similarity(anchor_normalized[0], anchor_normalized[1], dim=1)
            num_top_pairs = int(anchor_features[0].size(0) * self.anchor_top_ratio)
            top_similarity_values = torch.topk(anchor_similarities, k=num_top_pairs, dim=0).values
            anchor_align_loss = torch.mean(1 - top_similarity_values)
        
        # Center alignment loss
        center_align_loss = 0
        if self.center_w != 0 and (step > self.center_cutoff):
            self.source_prototypes.detach_()
            self.target_prototypes.detach_()
            
            # Update prototype centers
            for k in range(self.type_num):
                # Source domain prototype update
                source_mask = source_labels == k
                if source_mask.sum() != 0:
                    source_mean = source_features[source_mask].mean(dim=0)
                    self.source_prototypes[k] = self.momentum * self.source_prototypes[k] + (1 - self.momentum) * source_mean
                
                # Target domain prototype update
                target_mask = target_pred_labels == k
                if target_mask.sum() != 0 and torch.abs(self.target_prototypes[k]).sum() > 1e-7:
                    target_mean = target_features[target_mask].mean(dim=0)
                    self.target_prototypes[k] = self.momentum * self.target_prototypes[k] + (1 - self.momentum) * target_mean
            
            # Compute alignment loss
            center_align_loss = criterion(
                F.normalize(self.source_prototypes, dim=-1),
                F.normalize(self.target_prototypes, dim=-1)
            )
        
        # Total loss
        classification_loss = self.ce_w * cls_crit(source_predictions, source_labels)
        l1_regularization_loss = reg_crit(self.encoder) + reg_crit(self.head)
        
        total_loss = (classification_loss + l1_regularization_loss + self.ortho_w * reduction_loss + 
                     self.cont_w * contrastive_loss + self.align_w * align_loss + 
                     self.anchor_w * anchor_align_loss + self.center_w * center_align_loss + 
                     self.stc_w * stc_align_loss)
        
        total_loss.backward()
        optimizer_G.step()
        optimizer_C.step()
        
        # Logging
        if not (step % log_step):
            print(f"step {step}, classification_loss={classification_loss:.3f}, "
                  f"contrastive_loss={self.cont_w*contrastive_loss:.3f}, "
                  f"anchor_loss={self.anchor_w*anchor_align_loss:.3f}, "
                  f"center_align_loss={self.center_w*center_align_loss:.3f}")
            
            # Evaluation
            source_features_eval, target_features_eval, source_predictions_eval, target_predictions_eval = self.eval(inplace=False)
            target_pred_labels_eval = np.argmax(target_predictions_eval, axis=1)
            
            # Initialize prototype centers
            if step == self.center_cutoff:
                source_features_tensor = torch.from_numpy(source_features_eval).to(self.device)
                target_features_tensor = torch.from_numpy(target_features_eval).to(self.device)
                
                # Initialize source domain prototypes
                for k in range(self.type_num):
                    source_mask = self.source_label_ranks == k
                    if source_mask.sum() > 0:
                        self.source_prototypes[k] = source_features_tensor[source_mask].mean(dim=0)
                
                # Initialize target domain prototypes
                for k in range(self.type_num):
                    target_mask = target_pred_labels_eval == k
                    if target_mask.sum() > 0:
                        self.target_prototypes[k] = target_features_tensor[target_mask].mean(dim=0)
            
            # Evaluate performance
            if eval_target and (self.type_label in self.target_metadata.columns):
                target_true_labels = self.target_metadata[self.type_label].to_numpy()
                target_label_ids = np.array([self.trainlabel2id.get(label, -1) for label in target_true_labels])
                
                shared_mask = np.in1d(target_true_labels, self.source_classes)
                known_pred_labels = target_pred_labels_eval[shared_mask]
                known_true_labels = target_label_ids[shared_mask]
                closed_accuracy, f1_score = evaluator(known_pred_labels, known_true_labels)
                    
        
        return classification_loss.item()
    
    def train(self, opt='sgd', batch_size=500, training_steps=2000, lr=0.001, lr2=None, 
              weight_decay=5e-4, log_step=100, eval_target=False, eval_top_k=1):
        begin_time = time.time()
        print(f"Beginning time: {time.asctime(time.localtime(begin_time))}")
        
        # Initialize model
        optimizer_G, optimizer_C = self.init_train(opt, lr, lr2, weight_decay)
        reg_crit = L1Regularization(self.l1_w).to(self.device)
        reg_cont = InfoNCE(batch_size, self.cont_tau).to(self.device)
        cls_crit = nn.CrossEntropyLoss().to(self.device)
        criterion = nn.MSELoss()
        
        self.loss_cls_history = []
        for step in range(training_steps):
            loss_cls = self.train_step(
                step, batch_size, optimizer_G, optimizer_C, cls_crit, reg_crit, reg_cont,
                criterion, log_step, eval_target, eval_top_k
            )
            self.loss_cls_history.append(loss_cls)
        
        end_time = time.time()
        print(f"Ending time: {time.asctime(time.localtime(end_time))}")
        self.train_time = end_time - begin_time
        print(f"Training takes {self.train_time:.2f} seconds")
    
    def eval(self, batch_size=500, inplace=False):
        # Test data loader
        src_ds = ClsDataset(self.source_data, self.source_label_ids, binz=False, train=False)
        tgt_ds = ClsDataset(self.target_data, np.ones(self.n_target, dtype='int32'), binz=False, train=False)
        
        self.src_dl = DataLoader(src_ds, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False)
        self.tgt_dl = DataLoader(tgt_ds, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False)
        
        self.encoder.eval()
        self.head.eval()
        
        # Evaluation function
        def evaluate_domain(dataloader):
            features, predictions = [], []
            for x, y in dataloader:
                x = x.to(self.device)
                z = self.encoder(x)
                h = nn.Softmax(dim=1)(self.head(z))
                features.append(z.detach().cpu().numpy())
                predictions.append(h.detach().cpu().numpy())
            return np.vstack(features), np.vstack(predictions)
        
        # Source and target domain evaluation
        source_features_eval, source_predictions_eval = evaluate_domain(self.src_dl)
        target_features_eval, target_predictions_eval = evaluate_domain(self.tgt_dl)
        
        if inplace:
            self.source_features, self.target_features = source_features_eval, target_features_eval
            self.source_predictions, self.target_predictions = source_predictions_eval, target_predictions_eval
            self.features_combined = np.vstack([source_features_eval, target_features_eval])
            self.predictions_combined = np.vstack([source_predictions_eval, target_predictions_eval])
        return (source_features_eval, target_features_eval, source_predictions_eval, target_predictions_eval) if not inplace else None
    
    def annotate(self):
        try:
            self.target_predictions
        except AttributeError:
            self.eval(inplace=True)
        
        target_pred_labels = np.argmax(self.target_predictions, axis=1)
        return np.array([self.id2trainlabel[_] for _ in target_pred_labels])
    
    def relabel(self, source_labels):
        self.source_labels = source_labels
        self.source_classes = np.unique(self.source_labels)
        self.trainlabel2id = {v: i for i, v in enumerate(self.source_classes)}
        self.id2trainlabel = {v: k for k, v in self.trainlabel2id.items()}
        self.source_label_ids = np.array([self.trainlabel2id[label] for label in self.source_labels]).astype('int32')
        self.n_class = len(self.source_classes)
    
    def shuffle_data(self):
        def shuffle_domain_data(n_samples, data, meta, labels, id_labels, emb=None, struct_emb=None):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            shuffled_data = {
                'data': data[indices],
                'meta': meta.iloc[indices],
                'labels': labels[indices],
                'id_labels': id_labels[indices].astype('int32')
            }
            if emb is not None:
                shuffled_data['emb'] = emb[indices]
            if struct_emb is not None:
                shuffled_data['struct_emb'] = struct_emb[indices]
            return shuffled_data, indices
        
        # Source and target domain shuffling
        source_shuffled, _ = shuffle_domain_data(
            self.n_source, self.source_data, self.source_metadata, self.source_labels, self.source_label_ids
        )
        target_shuffled, _ = shuffle_domain_data(
            self.n_target, self.target_data, self.target_metadata, 
            np.zeros(self.n_target), np.zeros(self.n_target), 
            self.target_embeddings, self.struct_target_emb if self.use_struct else None
        )
        
        # Assign shuffled data
        self.source_data_shuffle = source_shuffled['data']
        self.source_metadata_shuffle = source_shuffled['meta']
        self.source_labels_shuffle = source_shuffled['labels']
        self.source_label_ids_shuffle = source_shuffled['id_labels']
        
        self.target_data_shuffle = target_shuffled['data']
        self.target_metadata_shuffle = target_shuffled['meta']
        self.target_embeddings_shuffle = target_shuffled['emb']
        if self.use_struct:
            self.struct_target_emb = target_shuffled['struct_emb']
    
    def get_nns(self, k=15):
        knn_ind = find_nearest_neighbors(self.target_embeddings_shuffle, query=self.target_embeddings_shuffle, k=k+1)[:, 1:]
        knn_ind = knn_ind.astype('int64')
        
        # Calculate KNN accuracy
        if self.type_label in self.target_metadata_shuffle.columns:
            y_ = self.target_metadata_shuffle[self.type_label].to_numpy()
            y_knn = y_[knn_ind.ravel()].reshape(knn_ind.shape)
            ratio = (y_.reshape(-1, 1) == y_knn).mean(axis=1).mean()
            print('=' * 50)
            print(f'knn correct ratio = {ratio:.4f}')
            print('=' * 50)
        
        self.knn_ind = knn_ind
    