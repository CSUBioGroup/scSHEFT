import torch.utils.data.dataloader as dataloader
from tool import hvg_binz_lognorm_scale, NN, anchor_point_distance, kNN_approx, knn_classifier_eval
from metrics import osr_evaluator
from loss import L1regularization, InfoNCE
from dataset import ClsDataset
from model import *
import torch.optim as optim
import torch.nn.functional as F
import scipy.sparse as sps
import torch.nn as nn
import torch
import numpy as np
import time
import os
import sys
import scanpy as sc

sys.path.append('../')

class BuildscSHEFT(object):
    def __init__(self,
                 encoder_type='linear', use_struct=False, n_latent=64, bn=False, dr=0.2,
                 l1_w=0.1, ortho_w=0.1,
                 cont_w=0.0, cont_tau=0.4, cont_cutoff=0.,
                 align_w=0.0, align_p=0.8, align_cutoff=0.,
                 anchor_w=0.0,
                 center_w=0.0, center_cutoff=0., momentum=0.9,
                 stc_w=0.0, stc_cutoff=0.,
                 clamp=None,
                 seed=1234,
                 novel_detection=False, novel_threshold=0.
                 ):

        # add device
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.n_latent = n_latent
        self.encoder_type = encoder_type
        self.use_struct = use_struct
        self.novel_detection=novel_detection
        self.novel_threshold=novel_threshold

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
        self.clamp = clamp

    def preprocess(self,
                   exp_id,
                   adata_inputs,   # list of 'anndata' object
                   atac_raw_emb,
                   pp_dict,
                   adata_adt_inputs=None,  # list of adata_adt
                   stc_emb_inputs=False,
                   ):
        '''
        Performing preprocess for a pair of datasets.
        '''
        rna = adata_inputs[0].copy()
        atac = adata_inputs[1].copy()
        if self.use_struct and stc_emb_inputs == False:  # for speed and space
            raw_atac = adata_inputs[2].copy()
            sc.tl.pca(raw_atac, n_comps=500, svd_solver="auto")
            self.struct_atac_emb = raw_atac.obsm['X_pca']
            np.save(f'/media/asus/data16t/huangzt/Data/data_lab/{exp_id}_struct_pca.npy', self.struct_atac_emb)
        if self.use_struct and stc_emb_inputs == True:  # for speed and space
            self.struct_atac_emb = np.load("/mnt/second19T/huangzt/scSHEFT/DataDir/bm_struct_pca.npy")

        n_rna, n_atac = rna.shape[0], atac.shape[0]
        n_feature1, n_feature2 = rna.shape[1], atac.shape[1]
        self.type_label = pp_dict['type_label']

        rna, atac = hvg_binz_lognorm_scale(rna, atac, pp_dict['hvg_num'], pp_dict['binz'],
                                                      pp_dict['lognorm'], pp_dict['scale_per_batch'])

        self.data_A = sps.csr_matrix(rna.X)
        self.data_B = sps.csr_matrix(atac.X)

        self.emb_B = atac_raw_emb

        if adata_adt_inputs is not None:
            print('Concating adt features...')
            csr_adt_a = sps.csr_matrix(adata_adt_inputs[0].X)
            self.data_A = sps.csr_matrix(sps.hstack([self.data_A, csr_adt_a]))

            csr_adt_b = sps.csr_matrix(adata_adt_inputs[1].X)
            self.data_B = sps.csr_matrix(sps.hstack([self.data_B, csr_adt_b]))

        self.n_input = self.data_A.shape[1]

        self.n_rna, self.n_atac = n_rna, n_atac
        self.meta_A = rna.obs.copy()
        self.meta_B = atac.obs.copy()

        y_A = self.meta_A[self.type_label].values
        self.relabel(y_A)

        if self.novel_detection:
            y_B = self.meta_B[self.type_label].values
            self.share_mask = np.in1d(y_B, self.class_A)
            self.share_class_name = np.unique(y_B[self.share_mask])

        self.shuffle_data()
        self.get_nns(pp_dict['knn'])


        rna_label = rna.obs[self.type_label]
        rna_label_int = rna_label.rank(
            method="dense", ascending=True).astype(int) - 1
        self.rna_label = rna_label.values           
        self.rna_label_int = rna_label_int.values   

        self.label_map = dict()
        for k in range(rna_label_int.max() + 1):
            self.label_map[k] = rna_label[rna_label_int == k][0]

        self.type_num = len(self.label_map)
        self.source_prototypes = torch.zeros(
            self.type_num, self.n_latent).cuda()
        self.target_prototypes = torch.zeros(
            self.type_num, self.n_latent).cuda()

        if self.anchor_w != 0:
            sc.tl.pca(rna, n_comps=self.emb_B.shape[1], svd_solver="auto")
            emb_rna = rna.obsm['X_pca']
            sc.tl.pca(atac, n_comps=self.emb_B.shape[1], svd_solver="auto")
            emb_atac = atac.obsm['X_pca']
            print("Finding anchors...")
            start_time = time.time()
            self.A_B, self.B_A, self.anchor_set = anchor_point_distance(
                emb_rna, emb_atac)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Finding anchors executed in {execution_time} seconds.")

    def init_train(self, opt, lr, lr2, weight_decay):
        if self.encoder_type == 'linear':
            self.encoder = torch.nn.DataParallel(
                Net_encoder(self.n_input, self.n_latent).cuda())
            self.head = torch.nn.DataParallel(
                Net_cell(self.n_latent, self.n_class).cuda())
        else:
            self.encoder = torch.nn.DataParallel(Nonlinear_encoder(
                self.n_input, self.n_latent, self.bn, self.dr).cuda())
            self.head = torch.nn.DataParallel(
                Net_cell(self.n_latent, self.n_class).cuda())

        if self.use_struct:
            self.stc_encoder = torch.nn.DataParallel(Nonlinear_encoder(
                self.struct_atac_emb.shape[1], self.n_latent, self.bn, self.dr).cuda())

        if opt == 'adam':
            optimizer_G = optim.Adam(
                self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer_C = optim.Adam(self.head.parameters(
            ), lr=lr2 if lr2 is not None else lr, weight_decay=weight_decay)
        elif opt == 'sgd':
            optimizer_G = optim.SGD(self.encoder.parameters(
            ), lr=lr, momentum=0.9, weight_decay=weight_decay)
            optimizer_C = optim.SGD(self.head.parameters(
            ), lr=lr2 if lr2 is not None else lr, momentum=0.9, weight_decay=weight_decay)
        return optimizer_G, optimizer_C

    # @profile
    def train_step(
        self,
        step, batch_size,
        optimizer_G, optimizer_C,
        cls_crit, reg_crit, reg_cont, criterion,
        log_step=100,
        eval_atac=False, eval_top_k=1, eval_open=False
    ):
        self.encoder.train()
        self.head.train()

        N_A = self.n_rna
        N_B = self.n_atac

        index_A = np.random.choice(np.arange(N_A), size=batch_size)
        x_A = torch.from_numpy(
            self.data_A_shuffle[index_A, :].A).float().cuda()
        y_A = torch.from_numpy(self.y_id_A_shuffle[index_A]).long().cuda()

        index_B = np.random.choice(np.arange(N_B), size=batch_size)
        x_B = torch.from_numpy(
            self.data_B_shuffle[index_B, :].A).float().cuda()
        if self.use_struct:
            x_B_stc = torch.from_numpy(
                self.struct_atac_emb[index_B, :]).float().cuda()

        f_A = self.encoder(x_A)
        if self.clamp:
            f_A = torch.clamp(f_A, min=-self.clamp, max=self.clamp)
        p_A = self.head(f_A)

        f_B = self.encoder(x_B)
        if self.clamp:
            f_B = torch.clamp(f_B, min=-self.clamp, max=self.clamp)

        if self.use_struct:
            f_B_stc = self.stc_encoder(x_B_stc)
            if self.clamp:
                f_B_stc = torch.clamp(f_B_stc, min=-self.clamp, max=self.clamp)
            f_B = f_B * self.momentum + f_B_stc * (1-self.momentum)

        p_B = self.head(f_B)
        pr_B = np.argmax(p_B.detach().cpu().numpy(), axis=1)

        optimizer_G.zero_grad()
        optimizer_C.zero_grad()

        #  Adapted NNDR loss for reduction (source:scJoint)
        A_center_loss = zero_center(f_A)
        A_corr_loss = non_corr(f_A)
        A_var_loss = max_var(f_A)

        B_center_loss = zero_center(f_B)
        B_corr_loss = non_corr(f_B)
        B_var_loss = max_var(f_B)
        adapted_NNDR_loss = A_center_loss+B_center_loss+A_corr_loss+B_corr_loss+B_var_loss

        if self.use_struct:
            B_stc_center_loss = zero_center(f_B_stc)
            B_stc_corr_loss = non_corr(f_B_stc)
            B_stc_var_loss = max_var(f_B_stc)
            adapted_NNDR_loss += B_stc_center_loss+B_stc_corr_loss+B_stc_var_loss

        cont_loss = 0
        if self.cont_w != 0 and (step >= self.cont_cutoff):
            B_pos_ind = get_pos_ind(index_B, self.knn_ind)
            x_B_pos = torch.from_numpy(
                self.data_B_shuffle[B_pos_ind, :].A).float().cuda()

            f_B_pos = self.encoder(x_B_pos)
            cont_loss = reg_cont(f_B, f_B_pos) 

        align_loss = 0.
        if (self.align_w != 0) and (step >= self.align_cutoff):
            bs = f_B.size(0)
            f_A_norm = F.normalize(f_A, p=2, dim=1)
            f_B_norm = F.normalize(f_B, p=2, dim=1)

            f_A_norm_detach, f_B_norm_detach = f_A_norm.detach(), f_B_norm.detach()

            cos_sim = torch.matmul(f_B_norm_detach, f_A_norm_detach.t())
            vals, inds = torch.max(cos_sim, dim=1) 
            vals, top_B_inds = torch.topk(vals, int(bs * self.align_p))
            top_B_A_inds = inds[top_B_inds]

            f_B_norm_top = f_B_norm[top_B_inds]
            f_A_norm_top = f_A_norm[top_B_A_inds]

            align_loss = - \
                torch.mean(torch.sum(f_A_norm_top * f_B_norm_top, dim=1))

        stc_align_loss = 0.
        if (self.stc_w != 0) and (step >= self.stc_cutoff):
            bs = f_B_stc.size(0)
            f_A_norm = F.normalize(f_A, p=2, dim=1)
            f_B_norm = F.normalize(f_B_stc, p=2, dim=1)
            f_A_norm_detach, f_B_norm_detach = f_A_norm.detach(), f_B_norm.detach()
            cos_sim = torch.matmul(f_B_norm_detach, f_A_norm_detach.t())
            vals, inds = torch.max(cos_sim, dim=1)  
            vals, top_B_inds = torch.topk(vals, int(bs * 0.8))
            top_B_A_inds = inds[top_B_inds]
            f_B_norm_top = f_B_norm[top_B_inds]
            f_A_norm_top = f_A_norm[top_B_A_inds]
            stc_align_loss = - \
                torch.mean(torch.sum(f_A_norm_top * f_B_norm_top, dim=1))

        Anchor_align_loss = 0.
        if self.anchor_w != 0:
            data_list = list(self.anchor_set)
            K = int(len(data_list) * 0.10) 
            indices = np.random.choice(len(data_list), size=K, replace=False)
            selected_elements = [data_list[i] for i in indices]
            anchor_Aidx = np.array([x[0] for x in selected_elements])
            anchor_Bidx = np.array([x[1] for x in selected_elements])
            anchor_A = torch.from_numpy(
                self.data_A_shuffle[anchor_Aidx, :].toarray()).float().cuda()
            anchor_B = torch.from_numpy(
                self.data_B_shuffle[anchor_Bidx, :].toarray()).float().cuda()
            anc_f_A = self.encoder(anchor_A)
            anc_f_B = self.encoder(anchor_B)
            anc_f_A_norm = F.normalize(anc_f_A, p=2, dim=1)
            anc_f_B_norm = F.normalize(anc_f_B, p=2, dim=1)
            cos_sims = F.cosine_similarity(anc_f_A_norm, anc_f_B_norm, dim=1)
            bs = anc_f_A.size(0)
            num_top_pairs = int(bs * 0.10) 
            top_sim_indices = torch.topk(
                cos_sims, k=num_top_pairs, dim=0).indices
            top_sim_values = cos_sims[top_sim_indices]
            Anchor_align_loss = torch.mean(1 - top_sim_values)


        center_align_loss = 0
        if self.center_w != 0 and (step > self.center_cutoff):
            self.source_prototypes.detach_()
            self.target_prototypes.detach_()
            for k in range(self.type_num):
                source_index = y_A == k
                if source_index.sum() != 0:        
                    self.source_prototypes[k] = self.momentum * self.source_prototypes[
                        k
                    ] + (1 - self.momentum) * f_A[y_A == k].mean(dim=0)
                target_index = pr_B == k
                if target_index.sum() != 0:  
                    if torch.abs(self.target_prototypes[k]).sum() > 1e-7:
                        self.target_prototypes[k] = self.momentum * self.target_prototypes[
                            k
                        ] + (1 - self.momentum) * f_B[pr_B == k].mean(dim=0)
            loss = criterion(
                F.normalize(self.source_prototypes, dim=-1),
                F.normalize(self.target_prototypes, dim=-1),
            )
            center_align_loss += loss

        loss_cls = cls_crit(p_A, y_A)
        l1_reg_loss = reg_crit(self.encoder) + reg_crit(self.head)

        loss = loss_cls + l1_reg_loss + self.ortho_w*adapted_NNDR_loss + self.cont_w*cont_loss + \
            self.align_w*align_loss + self.anchor_w*Anchor_align_loss + self.center_w * \
            center_align_loss + self.stc_w*stc_align_loss 

        loss.backward()
        optimizer_G.step()
        optimizer_C.step()

        # logging info
        if not (step % log_step):
            print("step %d, loss_cls=%.3f, loss_l1_reg=%.3f, loss_cont=%.3f, loss_align=%.3f, anchor_loss=%.3f, type_center_loss=%.3f, stc_loss=%.3f" %
                  (
                      step, loss_cls, l1_reg_loss,
                      self.cont_w*cont_loss, self.align_w *
                      align_loss, self.anchor_w*Anchor_align_loss, self.center_w*center_align_loss, self.stc_w*stc_align_loss
                  )
                  )

            feat_A, feat_B, head_A, head_B = self.eval(inplace=False)
            pr_A = np.argmax(head_A, axis=1)
            pr_B = np.argmax(head_B, axis=1)
            pr_B_top_k = np.argsort(-1 * head_B, axis=1)[:, :eval_top_k]

            if (step == self.center_cutoff):
                feat_A = torch.from_numpy(feat_A).cuda()
                feat_B = torch.from_numpy(feat_B).cuda()
                for k in range(self.type_num):
                    self.source_prototypes[k] = feat_A[self.rna_label_int == k].mean(
                        dim=0)
                    target_index = pr_B == k
                    if target_index.sum() != 0:
                        self.target_prototypes[k] = feat_B[target_index].mean(
                            dim=0)

            if eval_atac and (self.type_label in self.meta_B.columns):
                y_B = self.meta_B[self.type_label].to_numpy()
                y_id_B = np.array([self.trainlabel2id.get(_, -1) for _ in y_B])

                share_mask = np.in1d(y_B, self.class_A)
                pr_B_top_acc = knn_classifier_eval(
                    pr_B_top_k, y_id_B, True, share_mask)

                if not eval_open:  
                    print("Overall acc={:.5f}".format(pr_B_top_acc))
                else:             
                    closed_score = np.max(head_B, axis=1)
                    open_score = 1 - closed_score

                    kn_data_pr = pr_B[share_mask]
                    kn_data_gt = y_id_B[share_mask]
                    kn_data_open_score = open_score[share_mask]
                    unk_data_open_score = open_score[np.logical_not(
                        share_mask)]

                    closed_acc, os_auroc, os_aupr, oscr = osr_evaluator(
                        kn_data_pr, kn_data_gt, kn_data_open_score, unk_data_open_score)

        return loss_cls.item()

    def train(self,
              opt='sgd',
              batch_size=500, training_steps=2000,
              lr=0.001, lr2=None, weight_decay=5e-4,
              log_step=100, eval_atac=False, eval_top_k=1, eval_open=False,
              ):
        # torch.manual_seed(1)
        begin_time = time.time()
        print("Beginning time: ", time.asctime(time.localtime(begin_time)))
        # init model
        optimizer_G, optimizer_C = self.init_train(opt, lr, lr2, weight_decay)

        reg_crit = L1regularization(self.l1_w).cuda()
        reg_cont = InfoNCE(batch_size, self.cont_tau).cuda()
        cls_crit = nn.CrossEntropyLoss().cuda()
        criterion = nn.MSELoss()

        self.loss_cls_history = []
        for step in range(training_steps):
            loss_cls = self.train_step(
                step, batch_size,
                optimizer_G=optimizer_G, optimizer_C=optimizer_C,
                cls_crit=cls_crit, reg_crit=reg_crit, reg_cont=reg_cont,
                log_step=log_step, criterion=criterion,
                eval_atac=eval_atac, eval_top_k=eval_top_k, eval_open=eval_open
            )

            self.loss_cls_history.append(loss_cls)

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

    def eval(self, batch_size=500, inplace=False):
        src_ds = ClsDataset(self.data_A, self.y_id_A,
                            binz=False, train=False)   
        tgt_ds = ClsDataset(self.data_B, np.ones(
            self.n_atac, dtype='int32'), binz=False, train=False)
        self.src_dl = dataloader.DataLoader(
            src_ds, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False)
        self.tgt_dl = dataloader.DataLoader(
            tgt_ds, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False)

        self.encoder.eval()
        self.head.eval()
        feat_A, head_A = [], []
        for x, y in self.src_dl:  # consier batch remain
            x = x.cuda()
            z_A = self.encoder(x)
            h_A = nn.Softmax(dim=1)(self.head(z_A))
            feat_A.append(z_A.detach().cpu().numpy())
            head_A.append(h_A.detach().cpu().numpy())
        feat_B, head_B = [], []

        for x, y in self.tgt_dl:
            x = x.cuda()
            z_B = self.encoder(x)
            h_B = nn.Softmax(dim=1)(self.head(z_B))
            feat_B.append(z_B.detach().cpu().numpy())
            head_B.append(h_B.detach().cpu().numpy())
        feat_A, feat_B = np.vstack(feat_A), np.vstack(feat_B)
        head_A, head_B = np.vstack(head_A), np.vstack(head_B)

        if inplace:
            self.feat_A = feat_A
            self.feat_B = feat_B
            self.head_A = head_A
            self.head_B = head_B

            self.feat_AB = np.vstack([feat_A, feat_B])
            self.head_AB = np.vstack([head_A, head_B])
        else:
            return feat_A, feat_B, head_A, head_B

    def load_ckpt(self, path):
        self.encoder.load_state_dict(torch.load(path)['encoder'])
        self.head.load_state_dict(torch.load(path)['head'])
        print(f'loaded checkpoints from {path}')

    def annotate(self, label_prop=False, prop_knn=10):
        try:
            self.head_B
        except:
            self.eval(inplace=True)

        atac_pr = np.argmax(self.head_B, axis=1)
        if label_prop:
            atac_pr = kNN_approx(self.feat_B, self.feat_B,
                                 atac_pr, n_sample=None, knn=prop_knn)

        atac_pr = np.array([self.id2trainlabel[_] for _ in atac_pr])

        if self.novel_detection:
            open_score = 1 - np.max(self.head_B, axis=1)
            unk_data_open_score = open_score[np.logical_not(self.share_mask)]
            novel_indices = np.logical_not(self.share_mask)[np.where(unk_data_open_score > self.novel_threshold)]
            atac_pr[novel_indices] = "Novel"

        return atac_pr

    def relabel(self, y_A):
        """
        label <==> 索引idx
        """
        self.y_A = y_A

        self.class_A = np.unique(self.y_A)
        # self.class_B = np.unique(self.y_B)

        self.trainlabel2id = {v: i for i, v in enumerate(self.class_A)}
        self.id2trainlabel = {v: k for k, v in self.trainlabel2id.items()}

        self.y_id_A = np.array([self.trainlabel2id[_]
                               for _ in self.y_A]).astype('int32')
        # self.y_id_B = np.array([self.trainlabel2id.get(_, -1) for _ in self.y_B]).astype('int32')
        self.n_class = len(self.class_A) - (1 if self.novel_detection else 0)
        

    def shuffle_data(self):
        # shuffle source domain
        rand_idx_ai = np.arange(self.n_rna)
        np.random.shuffle(rand_idx_ai)
        self.data_A_shuffle = self.data_A[rand_idx_ai]
        self.meta_A_shuffle = self.meta_A.iloc[rand_idx_ai]
        self.y_A_shuffle = self.y_A[rand_idx_ai]
        self.y_id_A_shuffle = self.y_id_A[rand_idx_ai].astype('int32')

        # shuffle target domain
        random_idx_B = np.arange(self.n_atac)
        np.random.shuffle(random_idx_B)
        self.data_B_shuffle = self.data_B[random_idx_B]
        self.emb_B_shuffle = self.emb_B[random_idx_B]
        self.meta_B_shuffle = self.meta_B.iloc[random_idx_B]
        if self.use_struct:
            self.struct_atac_emb = self.struct_atac_emb[random_idx_B]
        # self.y_B_shuffle = self.y_B[random_idx_B]
        # self.y_id_B_shuffle = self.y_id_B[random_idx_B].astype('int32')

    def get_nns(self, k=15):
        knn_ind = NN(self.emb_B_shuffle, query=self.emb_B_shuffle,
                        k=k+1, metric='manhattan', n_trees=10)[:, 1:]
        knn_ind = knn_ind.astype('int64')

        if self.type_label in self.meta_B_shuffle.columns:
            y_ = self.meta_B_shuffle[self.type_label].to_numpy()
            y_knn = y_[knn_ind.ravel()].reshape(knn_ind.shape)
            ratio = (y_.reshape(-1, 1) == y_knn).mean(axis=1).mean()
            print('==========================')
            print('knn correct ratio = {:.4f}'.format(ratio))
            print('==========================')

        self.knn_ind = knn_ind


# def save_ckpts(output_dir, model, step):
#     state = {
#         'encoder': model.encoder.state_dict(),
#         'head': model.head.state_dict(),
#     }
#     torch.save(state, os.path.join(output_dir, f"ckpt_{step}.pth"))


def cor(m):   # covariance matrix of embedding features
    m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def non_corr(x):
    l = torch.mean(torch.abs(torch.triu(cor(x), diagonal=1)))
    return l


def zero_center(x):  # control value magnitude
    l = torch.mean(torch.abs(x))
    return l


def max_var(x):
    l = max_moment1(x)
    return l

def max_moment1(feats):
    loss = 1 / torch.mean(
        torch.abs(feats - torch.mean(feats, dim=0)))
    return loss


def get_pos_ind(ind, knn_ind):
    choice_per_nn_ind = np.random.randint(
        low=0, high=knn_ind.shape[1], size=ind.shape[0])
    pos_ind = knn_ind[ind, choice_per_nn_ind]
    return pos_ind
