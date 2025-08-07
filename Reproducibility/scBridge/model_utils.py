import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity


class Net(nn.Module):
    def __init__(self, gene_num, type_num, ce_weight, args):
        super(Net, self).__init__()
        self.type_num = type_num
        self.ce_weight = ce_weight
        self.align_loss_epoch = args.align_loss_epoch
        self.encoder = nn.Sequential(
            nn.Linear(gene_num, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.InstanceNorm1d(64),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, type_num),
        )
        self.adj_decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def run(
        self,
        source_dataloader_train,
        source_dataloader_eval,
        target_dataloader_train,
        target_dataloader_eval,
        target_adj,
        args,
    ):
        optim = torch.optim.AdamW(self.parameters(), lr=args.learning_rate)
        wce_loss = nn.CrossEntropyLoss(weight=self.ce_weight)
        align_loss = AlignLoss(type_num=self.type_num, feature_dim=64, args=args)
        epochs = args.train_epoch
        target_iter = iter(target_dataloader_train)
        for epoch in range(epochs):
            wce_loss_epoch = align_loss_epoch = stc_loss_epoch = 0.0
            train_acc = train_tot = 0.0
            self.train()
            for (source_x, source_y) in source_dataloader_train:
                source_x = source_x.cuda()
                source_y = source_y.cuda()
                try:
                    # 原数据及其索引+表示该批次样本在原始数据集中的索引
                    (target_x, adj_index), target_index = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_dataloader_train)
                    (target_x, adj_index), target_index = next(target_iter)
                target_x = target_x.cuda()

                source_h = self.encoder(source_x)
                source_pred = self.classifier(source_h)
                target_h = self.encoder(target_x)

                loss_wce = wce_loss(source_pred, source_y)

                wce_loss_epoch += loss_wce.item()
                train_acc += (
                    torch.argmax(
                        source_pred,
                        dim=-1,
                    )
                    == source_y
                ).sum()
                train_tot += source_x.shape[0]

                loss_epoch = loss_wce

                if epoch >= self.align_loss_epoch:
                    loss_align = align_loss(
                        source_h,
                        source_y,
                        target_h,
                        preds[target_index],
                        prob_feature[target_index] * prob_logit[target_index],
                    )
                    loss_epoch += loss_align
                    align_loss_epoch += loss_align.item()

                if args.novel_type:
                    adj = target_adj[adj_index, :][:, adj_index]
                    cos_sim_x = torch.from_numpy(adj).float().cuda()
                    target_h = F.normalize(self.adj_decoder(target_h), dim=-1)
                    cos_sim_h = F.relu(target_h @ target_h.T)
                    stc_loss = (cos_sim_x - cos_sim_h) * (cos_sim_x - cos_sim_h)
                    stc_loss = torch.clamp(stc_loss - 0.01, min=0).mean()
                    loss_epoch += stc_loss
                    stc_loss_epoch += stc_loss.item()

                optim.zero_grad()
                loss_epoch.backward()
                optim.step()

            train_acc /= train_tot
            wce_loss_epoch /= len(source_dataloader_train)
            align_loss_epoch /= len(source_dataloader_train)
            stc_loss_epoch /= len(source_dataloader_train)

            # 对验证集进行处理
            feature_vec, type_vec, omic_vec, loss_vec = self.inference(
                source_dataloader_eval, target_dataloader_eval
            )
            similarity, preds = feature_prototype_similarity(
                feature_vec[omic_vec == 0],
                type_vec,
                feature_vec[omic_vec == 1],
            )
            if epoch == self.align_loss_epoch - 1:
                align_loss.init_prototypes(
                    feature_vec[omic_vec == 0],
                    type_vec,
                    feature_vec[omic_vec == 1],
                    preds,  ## 验证集 通过 训练集原型 余弦相似度的聚类结果
                )
            prob_feature = gmm(1 - similarity)  
            prob_logit = gmm(loss_vec)

            preds = torch.from_numpy(preds).long().cuda()
            prob_feature = torch.from_numpy(prob_feature).float().cuda()
            prob_logit = torch.from_numpy(prob_logit).float().cuda()

            if args.novel_type:
                print(
                    "Epoch [%d/%d] WCE Loss: %.4f, ALG Loss: %.4f, STC Loss: %.4f, Train ACC: %.4f"
                    % (
                        epoch,
                        epochs,
                        wce_loss_epoch,
                        align_loss_epoch,
                        stc_loss_epoch,
                        train_acc,
                    )
                )
            else:
                print(
                    "Epoch [%d/%d] WCE Loss: %.4f, ALG Loss: %.4f, Train ACC: %.4f"
                    % (epoch, epochs, wce_loss_epoch, align_loss_epoch, train_acc)
                )

            if train_acc > args.early_stop_acc: # 训练RNA分类精度大于0.99可提前中断
                print("Early Stop.")
                break
        return preds.cpu(), prob_feature.cpu(), prob_logit.cpu()

    def inference(self, source_dataloader, target_dataloader):  
        self.eval()
        feature_vec, type_vec, omic_vec, loss_vec = [], [], [], []
        for (x, y) in source_dataloader:
            x = x.cuda()
            with torch.no_grad():
                h = self.encoder(x)
                logit = self.classifier(h)
            feature_vec.extend(h.cpu().numpy())
            type_vec.extend(y.numpy())
            omic_vec.extend(np.zeros(x.shape[0]))
        ce_loss = nn.CrossEntropyLoss(reduction="none")
        for (x, _), _ in target_dataloader:
            x = x.cuda()
            with torch.no_grad():
                h = self.encoder(x)
                logit = self.classifier(h)
                pred = torch.argmax(logit, dim=-1)      
                loss = ce_loss(logit, pred)
            feature_vec.extend(h.cpu().numpy()) 
            omic_vec.extend(np.ones(x.shape[0]))        # 这里对 ATAC-seq 也要用交叉熵？
            loss_vec.extend(loss.cpu().numpy())
        feature_vec, type_vec, omic_vec, loss_vec = (
            np.array(feature_vec),
            np.array(type_vec),
            np.array(omic_vec),
            np.array(loss_vec),
        )
        return feature_vec, type_vec, omic_vec, loss_vec    
        # 对于验证集当中(RNA,ATAC数目不等于原数据RNA-seq,ATAC-seq)  验证集包括：RNA-seq+已分类的ATAC-seq；剩下未分类好的ATAC-seq
        # output : RNA+ATAC Embedding//测试集源域的label//组学的标识符 RNA:0 ATAC:1//测试集目标域的(未分类的ATAC-seq每个细胞)的交叉熵数值


def gmm(X):
    X = ((X - X.min()) / (X.max() - X.min())).reshape(-1, 1)  # 每个数据点都规范化为0到1之间
    gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4).fit(X)
    prob = gmm.predict_proba(X)[:, gmm.means_.argmin()]
    return prob


def feature_prototype_similarity(source_feature, source_label, target_feature):
    """
    input : RNA-embedding  //  源域的label  //    ATAC-embedding
        Embedding中：ATAC 对 RNA原型(RNA各类细胞embedding总和) 去做一个相似度计算
    output: 相似度矩阵中ATAC最高的相似度及其索引(与idx类最相似) 
    """
                                                                                    # paired数据和unpaired数据处理起来区别是啥
    type_num = source_label.max() + 1
    source_prototypes = np.zeros((type_num, source_feature.shape[1])).astype(float)
    for k in range(type_num):
        source_prototypes[k] = source_feature[source_label == k].sum(axis=0)
    similarity = cosine_similarity(target_feature, source_prototypes) # 验证计算 ATAC到RNA原型的相似度矩阵
    pred = np.argmax(similarity, axis=1)    # 通过相似度矩阵来判别ATAC归属哪类
    similarity = np.max(similarity, axis=1) # 返回对应类别的最大的相似度值
    return similarity, pred


class AlignLoss(nn.Module):
    def __init__(self, type_num, feature_dim, args):
        super(AlignLoss, self).__init__()
        self.type_num = type_num
        self.feature_dim = feature_dim
        self.source_prototypes = torch.zeros(self.type_num, self.feature_dim).cuda()
        self.target_prototypes = torch.zeros(self.type_num, self.feature_dim).cuda()
        self.momentum = args.prototype_momentum
        self.criterion = nn.MSELoss()

    def init_prototypes(
        self, source_feature, source_label, target_feature, target_prediction
    ):
        source_feature = torch.from_numpy(source_feature).cuda()
        source_label = torch.from_numpy(source_label).cuda()
        target_feature = torch.from_numpy(target_feature).cuda()
        target_prediction = torch.from_numpy(target_prediction).cuda()
        # source_prototypes,target_prototypes 分别是[type_num,embedding]存有各组学的embedding中心点
        for k in range(self.type_num):
            self.source_prototypes[k] = source_feature[source_label == k].mean(dim=0)
            target_index = target_prediction == k
            if target_index.sum() != 0:
                self.target_prototypes[k] = target_feature[target_index].mean(dim=0)

    def forward(
        self,
        source_feature,
        source_label,
        target_feature,
        target_prediction,
        target_reliability,  # from GMM output
    ):
        self.source_prototypes.detach_()
        self.target_prototypes.detach_()
        for k in range(self.type_num):
            source_index = source_label == k
            if source_index.sum() != 0:         # 将训练的embedding 按比例融入原型当中
                self.source_prototypes[k] = self.momentum * self.source_prototypes[
                    k
                ] + (1 - self.momentum) * source_feature[source_label == k].mean(dim=0)
            target_index = target_prediction == k
            if target_index.sum() != 0: # 如果类别k在目标域中有样本,则更新目标域原型
                if torch.abs(self.target_prototypes[k]).sum() > 1e-7:
                    self.target_prototypes[k] = self.momentum * self.target_prototypes[
                        k
                    ] + (1 - self.momentum) * (
                        target_reliability[target_index].unsqueeze(1)       # 用GMM打分出来的权重 进一步更新ATAC原型
                        * target_feature[target_index]
                    ).mean(
                        dim=0
                    )
                else:  # Not Initialized
                    self.target_prototypes[k] = (
                        target_reliability[target_index].unsqueeze(1)
                        * target_feature[target_index]
                    ).mean(dim=0)
        loss = self.criterion(
            F.normalize(self.source_prototypes, dim=-1),
            F.normalize(self.target_prototypes, dim=-1),
        )
        # In the absence of some prototypes
        if (torch.abs(self.target_prototypes).sum(dim=1) > 1e-7).sum() < self.type_num:
            loss *= 0
        return loss     # 计算RNA和ATAC原型的loss
