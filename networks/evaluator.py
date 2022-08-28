import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from scipy.optimize import linear_sum_assignment

bceloss = nn.BCELoss(reduction='none')
img2cross_entropy = nn.CrossEntropyLoss()
thre_list = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95]
img2mse = lambda x, y: torch.mean((x - y) ** 2)
nllloss = nn.NLLLoss(ignore_index=-1, reduction='none')
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
cross_entropy = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def ins_criterion(pred_ins, gt_labels, ins_num):
    # change label to one hot
    valid_gt_labels = torch.unique(gt_labels)
    gt_ins = torch.zeros(size=(gt_labels.shape[0], ins_num))

    valid_ins_num = len(valid_gt_labels)
    gt_ins[..., :valid_ins_num] = F.one_hot(gt_labels.long())[..., valid_gt_labels.long()]

    cost_ce, cost_siou, order_row, order_col = hungarian(pred_ins, gt_ins, valid_ins_num, ins_num)
    valid_ce = torch.mean(cost_ce[order_row, order_col[:valid_ins_num]])

    if not (len(order_col) == valid_ins_num):
        invalid_ce = torch.mean(pred_ins[:, order_col[valid_ins_num:]])
    else:
        invalid_ce = torch.tensor([0])
    valid_siou = torch.mean(cost_siou[order_row, order_col[:valid_ins_num]])

    ins_loss_sum = valid_ce + invalid_ce + valid_siou
    return ins_loss_sum, valid_ce, invalid_ce, valid_siou


# matching function
def hungarian(pred_ins, gt_ins, valid_ins_num, ins_num):
    @torch.no_grad()
    def reorder(cost_matrix, valid_ins_num):
        valid_scores = cost_matrix[:valid_ins_num]
        valid_scores = valid_scores.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(valid_scores)

        unmapped = ins_num - valid_ins_num
        if unmapped > 0:
            unmapped_ind = np.array(list(set(range(ins_num)) - set(col_ind)))
            col_ind = np.concatenate([col_ind, unmapped_ind])
        return row_ind, col_ind

    # preprocess prediction and ground truth
    pred_ins = pred_ins.permute([1, 0])
    gt_ins = gt_ins.permute([1, 0])
    pred_ins = pred_ins[None, :, :]
    gt_ins = gt_ins[:, None, :]

    cost_ce = torch.mean(-gt_ins * torch.log(pred_ins + 1e-8) - (1 - gt_ins) * torch.log(1 - pred_ins + 1e-8), dim=-1)

    # get soft iou score between prediction and ground truth, don't need do mean operation
    TP = torch.sum(pred_ins * gt_ins, dim=-1)
    FP = torch.sum(pred_ins, dim=-1) - TP
    FN = torch.sum(gt_ins, dim=-1) - TP
    cost_siou = TP / (TP + FP + FN + 1e-6)
    cost_siou = 1.0 - cost_siou

    # final score
    cost_matrix = cost_ce + cost_siou
    # get final indies order
    order_row, order_col = reorder(cost_matrix, valid_ins_num)

    return cost_ce, cost_siou, order_row, order_col


def calculate_ap(IoUs_Metrics, gt_number, confidence=None, function_select='integral'):
    def interpolate_11(prec, rec):
        ap = 0.
        for t in torch.arange(0., 1.1, 0.1):
            if torch.sum(rec >= t) == 0:
                p = 0
            else:
                p = torch.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def integral_method(prec, rec):
        """
            This method same as coco
        """
        mrec = torch.cat((torch.Tensor([0.]), rec, torch.Tensor([1.])))
        mprec = torch.cat((torch.Tensor([0.]), prec, torch.Tensor([0.])))
        for i in range(mprec.shape[0] - 1, 0, -1):
            mprec[i - 1] = torch.maximum(mprec[i - 1], mprec[i])
        index = torch.where(mrec[1:] != mrec[:-1])[0]
        ap = torch.sum((mrec[index + 1] - mrec[index]) * mprec[index + 1])
        return ap

    '''begin'''
    # make TP matrix
    if confidence is not None:
        column_max_index = torch.argsort(confidence, descending=True)
        column_max_value = IoUs_Metrics[column_max_index]
    else:
        column_max_value = torch.sort(IoUs_Metrics, descending=True)
        column_max_value = column_max_value[0]

    ap_list = []
    for thre in thre_list:
        tp_list = column_max_value > thre
        tp_list = tp_list.to(device=device)
        precisions = torch.cumsum(tp_list, dim=0) / (torch.arange(len(tp_list)) + 1)
        recalls = torch.cumsum(tp_list, dim=0).type(torch.float32) / gt_number

        # select calculate function
        if function_select == 'integral':
            ap_list.append(integral_method(precisions, recalls).item())
        elif function_select == 'interpolate':
            ap_list.append(interpolate_11(precisions, recalls).item())

    return ap_list


def ins_eval(pred_ins, gt_ins, gt_ins_num, ins_num, mask=None):
    if mask is None:
        pred_label = torch.argmax(pred_ins, dim=-1)
        valid_pred_labels = torch.unique(pred_label)
    else:
        pred_label = torch.argmax(pred_ins, dim=-1)

        pred_label[mask == 0] = ins_num  # unlabeled index for prediction set as -1
        valid_pred_labels = torch.unique(pred_label)[:-1]

    valid_pred_num = len(valid_pred_labels)
    # prepare confidence masks and confidence scores
    pred_conf_mask = np.max(pred_ins.numpy(), axis=-1)

    pred_conf_list = []
    valid_pred_labels = valid_pred_labels.numpy().tolist()
    for label in valid_pred_labels:
        index = torch.where(pred_label == label)
        ssm = pred_conf_mask[index[0], index[1]]
        pred_obj_conf = np.median(ssm)
        pred_conf_list.append(pred_obj_conf)
    pred_conf_scores = torch.from_numpy(np.array(pred_conf_list))

    # change predicted labels to each signal object masks not existed padding as zero
    pred_ins = torch.zeros_like(gt_ins)
    pred_ins[..., :valid_pred_num] = F.one_hot(pred_label)[..., valid_pred_labels]

    cost_ce, cost_iou, order_row, order_col = hungarian(pred_ins.reshape((-1, ins_num)),
                                                        gt_ins.reshape((-1, ins_num)),
                                                        gt_ins_num, ins_num)

    valid_inds = order_col[:gt_ins_num]
    ious_metrics = 1 - cost_iou[order_row, valid_inds]

    # prepare confidence values
    confidence = torch.zeros(size=[gt_ins_num])
    for i, valid_ind in enumerate(valid_inds):
        if valid_ind < valid_pred_num:
            confidence[i] = pred_conf_scores[valid_ind]
        else:
            confidence[i] = 0

    ap = calculate_ap(ious_metrics, gt_ins_num, confidence=confidence, function_select='integral')

    invalid_mask = valid_inds >= valid_pred_num
    valid_inds[invalid_mask] = 0
    valid_pred_labels = torch.from_numpy(np.array(valid_pred_labels))
    return_labels = valid_pred_labels[valid_inds].cpu().numpy()
    return_labels[invalid_mask] = -1

    return pred_label, ap, return_labels
