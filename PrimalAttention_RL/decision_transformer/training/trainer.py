import numpy as np
import torch
from tqdm import tqdm

import time
import os
import json

import utils
from log import Logger


class Trainer:

    def __init__(self, variant, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None, ref_min_score=None, ref_max_score=None):
        self.variant = variant
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.total_steps = 0
        self.ref_min_score = ref_min_score
        self.ref_max_score = ref_max_score

        self.start_time = time.time()
        self.eta = self.variant["eta"]

        # make directory
        ts = time.gmtime()
        ts = time.strftime("%m-%d-%H:%M", ts)
        exp_name = str(variant['env']) + '-' + str(variant['dataset']) + '-' + ts + '-bs'  \
            + str(variant['batch_size']) + '-s' + str(variant['seed'])
        if variant['exp_prefix'] is not None:
            exp_name = variant['exp_prefix'] + '-' + exp_name
        variant['work_dir'] = variant['work_dir'] + '/' + exp_name
        utils.make_dir(variant['work_dir'])

        variant['model_dir'] = os.path.join(variant['work_dir'], 'model')
        utils.make_dir(variant['model_dir'])
        variant['video_dir'] = os.path.join(variant['work_dir'], 'video')
        utils.make_dir(variant['video_dir'])
        self.video_dir = variant['video_dir']

        with open(os.path.join(variant['work_dir'], 'args.json'), 'w') as f:
            json.dump(variant, f, sort_keys=True, indent=4)

        utils.snapshot_src('.', os.path.join(variant['work_dir'], 'src'), '.gitignore')

        self.logger = Logger(variant['work_dir'], use_tb=True)

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in tqdm(range(num_steps), desc="train step"):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

            self.total_steps += 1
            self.logger.log('train_steps/train_loss', train_loss, self.total_steps)
            self.logger.log('train_steps/lr', utils.get_lr(self.optimizer), self.total_steps)
        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in tqdm(self.eval_fns, desc="eval_fn"):
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v
                self.logger.log(f'evaluation/{k}', v, iter_num)

            if self.ref_min_score is not None and self.ref_max_score is not None:
                for k, v in outputs.items():
                    if 'return_mean' in k:
                        nv = (v - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)
                        logs[f'evaluation_normalized/{k}'] = nv
                        self.logger.log(f'evaluation_normalized/{k}', nv, iter_num)

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        self.logger.log('training/train_loss_mean', np.mean(train_losses), iter_num)
        self.logger.log('training/train_loss_std', np.std(train_losses), iter_num)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]
            self.logger.log(k, self.diagnostics[k], iter_num)

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        self.save(self.variant['model_dir'], iter_num)
        self.logger.dump(iter_num)

        return logs

    def save(self, model_dir, step):
        torch.save(
            self.model.state_dict(), '%s/model_%s.pt' % (model_dir, step)
        )

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds, score_list, Lambda_list = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss_main = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:, 1:], action_target, reward_target[:, 1:],
        )
        # compute the regularizations
        loss_ksvd = 0
        for i in range(len(score_list)):
            # dimension, direction
            Lambda_diag = torch.diag_embed(Lambda_list[i])
            loss_escore = torch.mean((torch.einsum('...nd,...ds->...ns', score_list[i][0], Lambda_diag.unsqueeze(0).repeat(score_list[i][0].size(0),1,1,1))).norm(dim=-1, p=2)**2, dim=[1,2])/2
            loss_rscore = torch.mean((torch.einsum('...nd,...ds->...ns', score_list[i][1], Lambda_diag.unsqueeze(0).repeat(score_list[i][1].size(0),1,1,1))).norm(dim=-1, p=2)**2, dim=[1,2])/2
            loss_trace = torch.einsum('...ps,...pd->...sd', score_list[i][2], score_list[i][3].type_as(score_list[i][2])).mean(dim=0).trace()
            loss_ksvd = loss_ksvd + (loss_escore + loss_rscore - loss_trace) ** 2
        loss_ksvd = (loss_ksvd / len(score_list)).mean()
        loss = loss_main + self.eta * loss_ksvd
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def eval(self, model_path, video=False):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        for eval_fn in tqdm(self.eval_fns, desc="eval_fn"):
            outputs = eval_fn(self.model, video=video, video_path=self.video_dir)
            for k, v in outputs.items():
                if 'return_mean' in k:
                    nv = (v - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)
                    print(f'evaluation_normalized/{k}', nv)
                    self.logger.log(f'evaluation_normalized/{k}', nv, 0)

        self.logger.dump(0)
