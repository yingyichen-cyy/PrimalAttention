import torch
import torch.nn as nn
import math
from model import Model

def pooling(inp, mode):
    if mode == "CLS":
        pooled = inp[:, 0, :]
    elif mode == "MEAN":
        pooled = inp.mean(dim = 1)
    else:
        raise Exception()
    return pooled

def append_cls(inp, mask, vocab_size):
    batch_size = inp.size(0)
    cls_id = ((vocab_size - 1) * torch.ones(batch_size, dtype = torch.long, device = inp.device)).long()
    cls_mask = torch.ones(batch_size, dtype = torch.float, device = mask.device)
    inp = torch.cat([cls_id[:, None], inp[:, :-1]], dim = -1)
    mask = torch.cat([cls_mask[:, None], mask[:, :-1]], dim = -1)
    return inp, mask

class SCHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_mode = config["pooling_mode"]
        self.mlpblock = nn.Sequential(
            nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["transformer_hidden_dim"], config["num_classes"])
        )

    def forward(self, inp):
        seq_score = self.mlpblock(pooling(inp, self.pooling_mode))
        return seq_score

class ModelForSC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]
        self.pooling_mode = config["pooling_mode"]
        self.vocab_size = config["vocab_size"]

        self.model = Model(config)

        self.seq_classifer = SCHead(config)
        self.attn_type = config["attn_type"]
        if self.attn_type.startswith("primal"):
            self.eta = config["eta"]
            self.trace_no_x = config["trace_no_x"]

    def forward(self, input_ids_0, mask_0, label):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):

            if self.pooling_mode == "CLS":
                input_ids_0, mask_0 = append_cls(input_ids_0, mask_0, self.vocab_size)

            if self.attn_type.startswith("primal"):
                token_out, score_list, Lambda_list = self.model(input_ids_0, mask_0)
            else:
                token_out = self.model(input_ids_0, mask_0)

            seq_scores = self.seq_classifer(token_out)

            if self.attn_type.startswith("primal"):
                loss_ce = torch.nn.CrossEntropyLoss(reduction = "none")(seq_scores, label)
                loss_ksvd = 0
                for i in range(len(score_list)):
                    # dimension, direction
                    Lambda_diag = torch.diag_embed(Lambda_list[i])
                    loss_escore = torch.mean((torch.einsum('...nd,...ds->...ns',score_list[i][0], Lambda_diag.unsqueeze(0).repeat(score_list[i][0].size(0),1,1,1))).norm(dim=-1, p=2)**2)/2
                    loss_rscore = torch.mean((torch.einsum('...nd,...ds->...ns',score_list[i][1], Lambda_diag.unsqueeze(0).repeat(score_list[i][1].size(0),1,1,1))).norm(dim=-1, p=2)**2)/2
                    if self.trace_no_x:
                        loss_trace = torch.einsum('...ps,...pd->...sd', score_list[i][2], score_list[i][3].type_as(score_list[i][2])).mean(dim=0).trace()
                    else:
                        loss_trace = torch.einsum('...ps,...pd->...sd', score_list[i][2], score_list[i][3].type_as(score_list[i][2])).mean(dim=(0,1)).trace()
                    loss_ksvd = loss_ksvd + (loss_escore + loss_rscore - loss_trace) ** 2
                loss_ksvd = loss_ksvd / len(score_list)
                seq_loss = loss_ce + self.eta * loss_ksvd
            else:
                seq_loss = torch.nn.CrossEntropyLoss(reduction = "none")(seq_scores, label)
            
            seq_accu = (seq_scores.argmax(dim = -1) == label).to(torch.float32)
            outputs = {}
            outputs["loss"] = seq_loss
            outputs["accu"] = seq_accu

            if self.attn_type.startswith("primal"):
                outputs["loss_ce"] = loss_ce
                outputs["loss_ksvd"] = loss_ksvd

        return outputs

class SCHeadDual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_mode = config["pooling_mode"]
        self.mlpblock = nn.Sequential(
            nn.Linear(config["transformer_dim"] * 4, config["transformer_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["transformer_hidden_dim"], config["num_classes"])
        )

    def forward(self, inp_0, inp_1):
        X_0 = pooling(inp_0, self.pooling_mode)
        X_1 = pooling(inp_1, self.pooling_mode)
        seq_score = self.mlpblock(torch.cat([X_0, X_1, X_0 * X_1, X_0 - X_1], dim = -1))
        return seq_score

class ModelForSCDual(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]
        self.pooling_mode = config["pooling_mode"]
        self.vocab_size = config["vocab_size"]
        
        self.model = Model(config)

        self.seq_classifer = SCHeadDual(config)
        self.attn_type = config["attn_type"]
        if self.attn_type.startswith("primal"):
            self.eta = config["eta"]
            self.trace_no_x = config["trace_no_x"]

    def forward(self, input_ids_0, input_ids_1, mask_0, mask_1, label):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):

            if self.pooling_mode == "CLS":
                input_ids_0, mask_0 = append_cls(input_ids_0, mask_0, self.vocab_size)
                input_ids_1, mask_1 = append_cls(input_ids_1, mask_1, self.vocab_size)

            if self.attn_type.startswith("primal"):
                token_out_0, score_list_0, Lambda_list_0 = self.model(input_ids_0, mask_0)
                token_out_1, score_list_1, Lambda_list_1 = self.model(input_ids_1, mask_1)
            else:
                token_out_0 = self.model(input_ids_0, mask_0)
                token_out_1 = self.model(input_ids_1, mask_1)

            seq_scores = self.seq_classifer(token_out_0, token_out_1)

            if self.attn_type.startswith("primal"):
                # extract KSVD related weights
                loss_ce = torch.nn.CrossEntropyLoss(reduction = "none")(seq_scores, label)
                loss_ksvd = 0
                for i in range(len(score_list_0)):
                    # dimension, direction
                    Lambda_diag_0 = torch.diag_embed(Lambda_list_0[i])
                    Lambda_diag_1 = torch.diag_embed(Lambda_list_1[i])
                    loss_escore_0 = torch.mean((torch.einsum('...nd,...ds->...ns',score_list_0[i][0], Lambda_diag_0.unsqueeze(0).repeat(score_list_0[i][0].size(0),1,1,1))).norm(dim=-1, p=2)**2)/2
                    loss_escore_1 = torch.mean((torch.einsum('...nd,...ds->...ns',score_list_1[i][0], Lambda_diag_1.unsqueeze(0).repeat(score_list_1[i][0].size(0),1,1,1))).norm(dim=-1, p=2)**2)/2
                    loss_rscore_0 = torch.mean((torch.einsum('...nd,...ds->...ns',score_list_0[i][1], Lambda_diag_0.unsqueeze(0).repeat(score_list_0[i][1].size(0),1,1,1))).norm(dim=-1, p=2)**2)/2
                    loss_rscore_1 = torch.mean((torch.einsum('...nd,...ds->...ns',score_list_1[i][1], Lambda_diag_1.unsqueeze(0).repeat(score_list_1[i][1].size(0),1,1,1))).norm(dim=-1, p=2)**2)/2
                    if self.trace_no_x:
                        loss_trace_0 = torch.einsum('...ps,...pd->...sd', score_list_0[i][2], score_list_0[i][3].type_as(score_list_0[i][2])).mean(dim=0).trace()
                        loss_trace_1 = torch.einsum('...ps,...pd->...sd', score_list_1[i][2], score_list_1[i][3].type_as(score_list_1[i][2])).mean(dim=0).trace()
                    else:
                        loss_trace_0 = torch.einsum('...ps,...pd->...sd', score_list_0[i][2], score_list_0[i][3].type_as(score_list_0[i][2])).mean(dim=(0,1)).trace()
                        loss_trace_1 = torch.einsum('...ps,...pd->...sd', score_list_1[i][2], score_list_1[i][3].type_as(score_list_1[i][2])).mean(dim=(0,1)).trace()
                    loss_ksvd = loss_ksvd + (0.5 * (loss_escore_0 + loss_escore_1) + 0.5 * (loss_rscore_0 + loss_rscore_1) - 0.5 * (loss_trace_0 + loss_trace_1)) ** 2
                loss_ksvd = loss_ksvd / len(score_list_0)
                seq_loss = loss_ce + self.eta * loss_ksvd
            else:            
                seq_loss = torch.nn.CrossEntropyLoss(reduction = "none")(seq_scores, label)
            
            seq_accu = (seq_scores.argmax(dim = -1) == label).to(torch.float32)
            outputs = {}
            outputs["loss"] = seq_loss
            outputs["accu"] = seq_accu

            if self.attn_type.startswith("primal"):
                outputs["loss_ce"] = loss_ce
                outputs["loss_ksvd"] = loss_ksvd

        return outputs