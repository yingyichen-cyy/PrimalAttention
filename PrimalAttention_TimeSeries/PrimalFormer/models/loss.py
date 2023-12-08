import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_module(config):

    task = config['task']

    if (task == "imputation") or (task == "transduction"):
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element

    if task == "classification":
        if config["model"] == "softmax":
            return NoFussCrossEntropyLoss(reduction='none')
        elif config["model"].startswith("primal"):
            return NoFussCE_KSVD(reduction='none')  # outputs loss for each batch sample

    if task == "regression":
        return nn.MSELoss(reduction='none')  # outputs loss for each batch sample

    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task))

class NoFussCE_KSVD(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, Lambda_list, inp, target, eta, score_list):
        loss_ce = F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
        loss_ksvd = 0
        for i in range(len(score_list)):
            # dimension, direction
            Lambda_diag = torch.diag_embed(Lambda_list[i])
            loss_escore = torch.mean((torch.einsum('...nd,...ds->...ns', score_list[i][0], Lambda_diag.unsqueeze(0).repeat(score_list[i][0].size(0),1,1,1))).norm(dim=-1, p=2)**2, dim=[1,2])/2
            loss_rscore = torch.mean((torch.einsum('...nd,...ds->...ns', score_list[i][1], Lambda_diag.unsqueeze(0).repeat(score_list[i][1].size(0),1,1,1))).norm(dim=-1, p=2)**2, dim=[1,2])/2
            loss_trace = torch.einsum('...ps,...pd->...sd', score_list[i][2], score_list[i][3].type_as(score_list[i][2])).mean(dim=0).trace()
            loss_ksvd = loss_ksvd + (loss_escore + loss_rscore - loss_trace) ** 2
        loss_ksvd = loss_ksvd / len(score_list)
        loss_total = loss_ce + eta * loss_ksvd
        return loss_total, loss_ce, loss_ksvd

def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)
