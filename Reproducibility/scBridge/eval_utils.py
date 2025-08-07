import os
import pandas as pd
import torch
import numpy as np
import scanpy as sc
import seaborn as sns
import scanpy.external as sce
from torch import nn
from model_utils import feature_prototype_similarity, gmm
from utilsForCompeting import evaluator

def infer_result(net, source_dataloader, target_dataloader, args):
    net.eval()
    feature_vec, type_vec, pred_vec, loss_vec = [], [], [], []
    for (x, y) in source_dataloader:
        x = x.cuda()
        with torch.no_grad():
            h = net.encoder(x)
        feature_vec.extend(h.cpu().numpy())
        type_vec.extend(y.numpy())
    ce_loss = nn.CrossEntropyLoss(reduction="none")
    for (x, _), _ in target_dataloader:
        x = x.cuda()
        with torch.no_grad():
            h = net.encoder(x)
            logit = net.classifier(h)
            pred = torch.argmax(logit, dim=-1)
            loss = ce_loss(logit, pred)
        feature_vec.extend(h.cpu().numpy())
        pred_vec.extend(pred.cpu().numpy())
        loss_vec.extend(loss.cpu().numpy())
    feature_vec, type_vec, pred_vec, loss_vec = (
        np.array(feature_vec),
        np.array(type_vec),
        np.array(pred_vec),
        np.array(loss_vec),
    )

    similarity, _ = feature_prototype_similarity(
        feature_vec[: len(source_dataloader.dataset)],
        type_vec,
        feature_vec[len(source_dataloader.dataset) :],
    )
    prob_feature = gmm(1 - similarity)
    prob_logit = gmm(loss_vec)
    reliability_vec = prob_feature * prob_logit

    if args.novel_type:
        prob_gmm = gmm(reliability_vec)
        novel_index = prob_gmm > 0.5
        pred_vec[novel_index] = -1

    return feature_vec, pred_vec, reliability_vec


def save_result(
    feature_vec,
    pred_vec,
    reliability_vec,
    label_map,
    type_num,
    source_adata,
    target_adata,
    args,
):
    adata = sc.AnnData(feature_vec)
    adata.obs["Domain"] = np.concatenate(
        (source_adata.obs["Domain"], target_adata.obs["Domain"]), axis=0
    )
    sc.tl.pca(adata)
    sce.pp.harmony_integrate(adata, "Domain", theta=0.0, verbose=False)
    feature_vec = adata.obsm["X_pca_harmony"]

    # source_adata.obsm["Embedding"] = feature_vec[: len(source_adata.obs["Domain"])]
    target_adata.obsm["Embedding"] = feature_vec[len(source_adata.obs["Domain"]) :]
    predictions = np.empty(len(target_adata.obs["Domain"]), dtype=np.dtype("U30"))
    for k in range(type_num):
        predictions[pred_vec == k] = label_map[k]
    if args.novel_type:
        predictions[pred_vec == -1] = "Novel (Most Unreliable)"
    target_adata.obs["pred_type"] = predictions
    target_adata.obs["Reliability"] = reliability_vec

    try:
        target_label_int = torch.from_numpy(
            (
                target_adata.obs["cell_type"]
                .rank(method="dense", ascending=True)
                .astype(int)
                - 1
            ).values
        )
        evaluation = True
    except:
        print("No Target Cell Type Annotations Provided, Skip Evaluation")
        evaluation = False
    if evaluation and not args.novel_type:
        print("=======Evaluation=======")
        closed_acc, f1_score = evaluator(pred_vec, target_label_int)
        print(f"scBridge Accuracy: {closed_acc:.4f}, F1_score: {f1_score:.4f}")
        

    # source_save_path = args.data_path + args.source_data[:-5] + "-integrated.h5ad"
    # target_save_path = args.data_path + args.target_data[:-5] + "-integrated.h5ad"
    # source_adata.write(source_save_path)
    # target_adata.write(target_save_path)
    # print(
    #     "Integration Results Saved to %s and %s" % (source_save_path, target_save_path)
    # )
