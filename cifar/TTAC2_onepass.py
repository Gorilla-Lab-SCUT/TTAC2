import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *

# ----------------------------------
import copy
import math
import random
import numpy as np
import torch.backends.cudnn as cudnn

from offline import *
from utils.contrastive import *
import time
# ----------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default="./data")
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--batch_size_align', default=512, type=int)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--num_sample', default=1000000, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--iters', default=4, type=int)
parser.add_argument('--outf', default='.')
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--corruption', default='snow')
parser.add_argument('--resume', default=None, help='directory of pretrained model')
parser.add_argument('--ckpt', default=None, type=int)
parser.add_argument('--ssl', default='fixmatch', help='self-supervised task')
parser.add_argument('--temperature', default=0.5, type=float)
parser.add_argument('--align_ext', action='store_true')
parser.add_argument('--align_ssh', action='store_true')
parser.add_argument('--fix_ssh', action='store_true')
parser.add_argument('--with_ssl', action='store_true', default=False)
parser.add_argument('--with_contrastive', action='store_true', default=False)
parser.add_argument('--without_global', action='store_true', default=False)
parser.add_argument('--without_mixture', action='store_true', default=False)

parser.add_argument('--ssl_sample', default='weak+strong', choices=['weak', 'weak+strong'])
parser.add_argument('--align_sample', default='weak', choices=['weak', 'weak+strong', 'none'])

parser.add_argument('--filter', default="ours", choices=['ours', 'posterior', 'none'])
parser.add_argument('--model', default='resnet50', help='resnet50')
parser.add_argument('--seed', default=0, type=int)


args = parser.parse_args()

print(args)

my_makedir(args.outf)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

cudnn.benchmark = True

# -------------------------------

net, ext, head, ssh, classifier = build_resnet50(args)
_, teloader = prepare_test_data(args)

# -------------------------------

args.batch_size = min(args.batch_size, args.num_sample)
args.batch_size_align = min(args.batch_size_align, args.num_sample)

args_align = copy.deepcopy(args)
args_align.ssl = None
args_align.batch_size = args.batch_size_align

tr_dataset, tr_dataloader = prepare_train_data(args, args.num_sample)

# -------------------------------

print('Resuming from %s...' %(args.resume))

load_resnet50(net, head, ssh, classifier, args)

if torch.cuda.device_count() > 1:
    ext = torch.nn.DataParallel(ext)

# ----------- Test ------------

all_err_cls = []
all_err_ssh = []

print('Running...')

if args.fix_ssh:
    optimizer = optim.SGD(ext.parameters(), lr=args.lr, momentum=0.9)
else:
    optimizer = optim.SGD(ssh.parameters(), lr=args.lr, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    'min', factor=0.5, patience=10, cooldown=10,
    threshold=0.0001, threshold_mode='rel', min_lr=0.0001, verbose=True)

criterion = SupConLoss(temperature=args.temperature).cuda()

# -------------------------------

class_num = 10 if args.dataset == 'cifar10' else 100

# ----------- Offline Feature Summarization ------------
_, offlineloader = prepare_train_data(args_align)

ext_src_mu, ext_src_cov, ssh_src_mu, ssh_src_cov, mu_src_ext, cov_src_ext, mu_src_ssh, cov_src_ssh = offline(offlineloader, ext, classifier, head, class_num)
bias = cov_src_ext.max().item() / 30.
bias2 = cov_src_ssh.max().item() / 30.
template_ext_cov = torch.eye(2048).cuda() * bias
template_ssh_cov = torch.eye(128).cuda() * bias2

print('Error (%)\t\ttest')
err_cls = test(teloader, net)[0]
print(('Epoch %d:' %(0)).ljust(24) +
            '%.2f\t\t' %(err_cls*100))

# ----------- Improved Test-time Training ------------

ext_src_mu = torch.stack(ext_src_mu)
ext_src_cov = torch.stack(ext_src_cov) + template_ext_cov[None, :, :]

source_component_distribution = torch.distributions.MultivariateNormal(ext_src_mu, ext_src_cov)
target_compoent_distribution = torch.distributions.MultivariateNormal(ext_src_mu, ext_src_cov)

is_both_activated=False
sample_predict_ema_logit = torch.zeros(len(tr_dataloader.dataset), class_num, dtype=torch.float)
sample_predict_alpha = torch.ones(len(tr_dataloader.dataset), dtype=torch.float)
ema_alpha = 0.9

ema_n = torch.zeros(class_num).cuda()
ema_ext_mu = ext_src_mu.clone()
ema_ext_cov = ext_src_cov.clone()

ema_ext_total_mu = torch.zeros(2048).float()
ema_ext_total_cov = torch.zeros(2048, 2048).float()

ema_ext_total_mu_weak = torch.zeros(2048).float().cuda()
ema_ext_total_cov_weak = torch.zeros(2048, 2048).float().cuda()

ema_ext_total_mu_strong = torch.zeros(2048).float()
ema_ext_total_cov_strong = torch.zeros(2048, 2048).float()


ema_total_n = 0.
if class_num == 10:
    ema_length = 128
    mini_batch_length = 4096
else: 
    ema_length = 64
    mini_batch_length = 4096

if class_num == 10:
    loss_scale = 0.05
else:
    loss_scale = 0.5

mini_batch_indices = []

correct = []
    
for te_batch_idx, (te_inputs, te_labels) in enumerate(teloader):
    mini_batch_indices.extend(te_inputs[-1].tolist())
    mini_batch_indices = mini_batch_indices[-mini_batch_length:]
    print('mini_batch_length:', len(mini_batch_indices))
    try:
        del tr_dataset_subset
        del tr_dataloader
    except:
        pass
    tr_dataset_subset = data.Subset(tr_dataset, mini_batch_indices)
    tr_dataloader = data.DataLoader(tr_dataset_subset, batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.workers,
                                        worker_init_fn=seed_worker, pin_memory=True, drop_last=True)

    if args.fix_ssh:
        head.eval()
    else:
        head.train()
    ext.train()
    classifier.eval()

    for iter_id in range(min(args.iters, int(len(mini_batch_indices) / 256) + 1) + 1):
        if iter_id > 0:
            sample_predict_alpha = torch.where(sample_predict_alpha < 1, sample_predict_alpha + 0.2, torch.ones_like(sample_predict_alpha))

        for batch_idx, (inputs, labels) in enumerate(tr_dataloader):
            optimizer.zero_grad()
            bsz = inputs[0].shape[0]

            if args.with_contrastive:
                images = torch.cat([inputs[0], inputs[3]], dim=0)
                images = images.cuda(non_blocking=True)
                indexes = inputs[-1]
                backbone_features = ext(images)
                features = F.normalize(head(backbone_features), dim=1)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = criterion(features)
                loss.backward()
                del loss, images, backbone_features, features, f1, f2
                torch.cuda.empty_cache()
            
            loss = 0.

            if args.with_ssl:
                if args.ssl_sample == 'weak+strong':
                    images = torch.cat([inputs[0], inputs[1]], dim=0)
                    images = images.cuda(non_blocking=True)
                    feats = ext(images)
                    weak_feats, strong_feats = torch.split(feats, [bsz, bsz], dim=0)
                    logits = classifier(feats)
                    weak_logits, strong_logits = torch.split(logits, [bsz, bsz], dim=0)
                elif args.ssl_sample == 'weak':
                    inputs_0 = inputs[0].cuda(non_blocking=True)
                    weak_feats = ext(inputs_0)
                    weak_logits = classifier(weak_feats)
            elif iter_id > 0 and args.align_ext:
                if args.align_sample == 'weak+strong':
                    images = torch.cat([inputs[0], inputs[1]], dim=0)
                    images = images.cuda(non_blocking=True)
                    feats = ext(images)
                    weak_feats, strong_feats = torch.split(feats, [bsz, bsz], dim=0)
                elif args.align_sample == 'weak':
                    inputs_0 = inputs[0].cuda(non_blocking=True)
                    weak_feats = ext(inputs_0)
                    
            if args.with_ssl:
                pro, target = weak_logits.softmax(dim=-1).detach().max(dim=-1)
                mask = pro.ge(0.95).float()
                if args.ssl_sample == 'weak':
                    strong_logits = weak_logits
                loss += (F.cross_entropy(strong_logits, target, reduction="none") * mask).mean() * loss_scale * 20.
                
            if iter_id > 0 and args.align_ext:
                if args.align_sample == 'none':
                    inputs_2 = inputs[2].cuda(non_blocking=True)
                    weak_feats = ext(inputs_2)
                
                indexes = inputs[-1]

                with torch.no_grad():
                    ext.eval()
                    if args.align_sample == 'none':
                        inputs_ = inputs[2]
                    else:
                        inputs_ = inputs[0]
                    weak_softmax_logits = classifier(ext(inputs_.cuda(non_blocking=True))).softmax(dim=-1)
                    weak_pseudo_label = weak_softmax_logits.cpu()
                    old_logit = sample_predict_ema_logit[indexes, :]
                    max_val, max_pos = weak_pseudo_label.max(dim=1)
                    old_max_val = old_logit[torch.arange(max_pos.shape[0]), max_pos]
                    accept_mask = max_val > (old_max_val - 0.001)

                    sample_predict_alpha[indexes] = torch.where(accept_mask, sample_predict_alpha[indexes], torch.zeros_like(accept_mask).float())

                    sample_predict_ema_logit[indexes, :] = \
                        torch.where(sample_predict_ema_logit[indexes, :] == torch.zeros(class_num), \
                                    weak_pseudo_label, \
                                    (1 - ema_alpha) * sample_predict_ema_logit[indexes, :] + ema_alpha * weak_pseudo_label)
                    
                    pro, pseudo_label = sample_predict_ema_logit[indexes].max(dim=1)
                    ext.train()
                    del weak_softmax_logits

                if args.filter == 'ours':
                    pseudo_label_mask = (sample_predict_alpha[indexes] == 1) & (pro > 0.9)
                    if args.align_sample == 'weak+strong':
                        feat_ext2 = strong_feats[pseudo_label_mask]
                        pseudo_label2 = pseudo_label[pseudo_label_mask].cuda()
                    else:
                        feat_ext2 = weak_feats[pseudo_label_mask]
                        pseudo_label2 = pseudo_label[pseudo_label_mask].cuda()

                elif args.filter == 'none':
                    feat_ext2 = weak_feats
                    pseudo_label2 = pseudo_label.cuda()
                elif args.filter == 'posterior':
                    with torch.no_grad():
                        posterior = target_compoent_distribution.log_prob(weak_feats[:, None, :])
                        posterior_tmp = posterior.max(dim=1, keepdim=True)[0] - math.log((2 ** 127) / 10) # B, K
                        posterior -= posterior_tmp
                        posterior = posterior.exp()
                        posterior /= posterior.sum(dim=1, keepdim=True)
                        posterior = posterior.transpose(0, 1).detach()  # K, N
                else:
                    raise Exception("%s filter type has not yet been implemented." % args.filter)


                if args.align_ext:
                    if not args.without_mixture:
                        # Mixture Gaussian
                        if args.filter != 'posterior':
                            b, d = feat_ext2.shape
                            feat_ext2_categories = torch.zeros(class_num, b, d).cuda() # K, N, D
                            feat_ext2_categories.scatter_add_(dim=0, index=pseudo_label2[None, :, None].expand(-1, -1, d), src=feat_ext2[None, :, :])
                            num_categories = torch.zeros(class_num, b, dtype=torch.int).cuda() # K, N
                            num_categories.scatter_add_(dim=0, index=pseudo_label2[None, :], src=torch.ones_like(pseudo_label2[None, :], dtype=torch.int))

                            ema_n += num_categories.sum(dim=1) # K
                            alpha = torch.where(ema_n > ema_length, torch.ones(class_num, dtype=torch.float).cuda() / ema_length, 1. / (ema_n + 1e-10))

                            delta_pre = (feat_ext2_categories - ema_ext_mu[:, None, :]) * num_categories[:, :, None] # K, N, D
                            delta = alpha[:, None] * delta_pre.sum(dim=1) # K, D
                            new_component_mean = ema_ext_mu + delta
                            new_component_cov = ema_ext_cov \
                                                + alpha[:, None, None] * ((delta_pre.permute(0, 2, 1) @ delta_pre) - num_categories.sum(dim=1)[:, None, None] * ema_ext_cov) \
                                                - delta[:, :, None] @ delta[:, None, :]

                            with torch.no_grad():
                                ema_ext_mu = new_component_mean.detach()
                                ema_ext_cov = new_component_cov.detach()
                            
                            if (class_num == 10 or len(mini_batch_indices) >= 4096) and (iter_id > int(args.iters / 2) or args.filter == 'none'):
                                target_compoent_distribution.loc = new_component_mean
                                target_compoent_distribution.covariance_matrix = new_component_cov + template_ext_cov
                                target_compoent_distribution._unbroadcasted_scale_tril = torch.linalg.cholesky(new_component_cov + template_ext_cov)
                                loss += (torch.distributions.kl_divergence(source_component_distribution, target_compoent_distribution) \
                                        + torch.distributions.kl_divergence(target_compoent_distribution, source_component_distribution)).mean() * loss_scale
                        else:
                            feat_ext2_categories = weak_feats[None, :, :].expand(class_num, -1, -1) # K, N, D
                            num_categories = posterior # K, N
                            ema_n += num_categories.sum(dim=1) # K
                            alpha = torch.where(ema_n > ema_length, torch.ones(class_num, dtype=torch.float).cuda() / ema_length, 1. / (ema_n + 1e-10))
                            
                            delta_pre = (feat_ext2_categories - ema_ext_mu[:, None, :]) * num_categories[:, :, None] # K, N, D
                            delta = alpha[:, None] * delta_pre.sum(dim=1) # K, D
                            new_component_mean = ema_ext_mu + delta
                            new_component_cov = ema_ext_cov \
                                                + alpha[:, None, None] * ((delta_pre.permute(0, 2, 1) @ delta_pre) - num_categories.sum(dim=1)[:, None, None] * ema_ext_cov) \
                                                - delta[:, :, None] @ delta[:, None, :]

                            with torch.no_grad():
                                ema_ext_mu = new_component_mean.detach()
                                ema_ext_cov = new_component_cov.detach()
                            
                            if (class_num == 10 or len(mini_batch_indices) >= 4096) and iter_id > int(args.iters / 2):
                                target_compoent_distribution.loc = new_component_mean
                                target_compoent_distribution.covariance_matrix = new_component_cov + template_ext_cov
                                target_compoent_distribution._unbroadcasted_scale_tril = torch.linalg.cholesky(new_component_cov + template_ext_cov)
                                loss += (torch.distributions.kl_divergence(source_component_distribution, target_compoent_distribution) \
                                        + torch.distributions.kl_divergence(target_compoent_distribution, source_component_distribution)).mean() * loss_scale

                    if not args.without_global:
                        # Global Gaussian
                        b = weak_feats.shape[0]
                        ema_total_n += b
                        alpha = 1. / 1280 if ema_total_n > 1280 else 1. / ema_total_n
                        
                        delta_pre = (weak_feats - ema_ext_total_mu_weak)
                        delta = alpha * delta_pre.sum(dim=0)
                        tmp_mu_weak = ema_ext_total_mu_weak + delta
                        tmp_cov_weak = ema_ext_total_cov_weak + alpha * (delta_pre.t() @ delta_pre - b * ema_ext_total_cov_weak) - delta[:, None] @ delta[None, :]
                        with torch.no_grad():
                            ema_ext_total_mu_weak = tmp_mu_weak.detach()
                            ema_ext_total_cov_weak = tmp_cov_weak.detach()

                        source_domain = torch.distributions.MultivariateNormal(mu_src_ext, cov_src_ext + template_ext_cov)
                        target_domain = torch.distributions.MultivariateNormal(tmp_mu_weak, tmp_cov_weak + template_ext_cov)
                        loss += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) * loss_scale

            if iter_id > 0 :
                if type(loss) != float:
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            elif args.with_ssl:
                if args.ssl_sample == 'weak':
                    del weak_logits, weak_feats, inputs_0
                else:
                    del weak_logits, strong_logits, logits, weak_feats, strong_feats, feats, images, loss, pro, target, mask
            torch.cuda.empty_cache()

    #### Test ####
    net.eval()
    with torch.no_grad():
        outputs = net(te_inputs[0].cuda())
        _, predicted = outputs.max(1)

        correct.append(predicted.cpu().eq(te_labels))
    print('real time error:', 1 - torch.cat(correct).numpy().mean())
    net.train()

print(args.corruption, 'Test time training result:', 1 - torch.cat(correct).numpy().mean())