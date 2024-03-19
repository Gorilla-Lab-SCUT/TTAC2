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
from utils.find_prototypes import FindPrototypes

# ----------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default="./data")
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--batch_size_align', default=512, type=int)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--num_sample', default=1000000, type=int)
parser.add_argument('--bnepoch', default=2, type=int)
parser.add_argument('--nepoch', default=500, type=int)
parser.add_argument('--stopepoch', default=25, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--outf', default='.')
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--corruption', default='snow')
parser.add_argument('--resume', default=None, help='directory of pretrained model')
parser.add_argument('--ckpt', default=None, type=int)
parser.add_argument('--ssl', default='fixmatch', help='self-supervised task')
parser.add_argument('--align_ext', action='store_true')
parser.add_argument('--with_ssl', action='store_true', default=False)
parser.add_argument('--with_shot', action='store_true', default=False)
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

_, tr_dataloader = prepare_train_data(args, args.num_sample)

# -------------------------------

print('Resuming from %s...' %(args.resume))

load_resnet50(net, head, ssh, classifier, args)

if torch.cuda.device_count() > 1:
    ext = torch.nn.DataParallel(ext)

# ----------- Test ------------

all_err_cls = []

print('Running...')

optimizer = optim.SGD(ext.parameters(), lr=args.lr, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    'min', factor=0.5, patience=10, cooldown=10,
    threshold=0.0001, threshold_mode='rel', min_lr=0.0001, verbose=True)

# -------------------------------

class_num = 10 if args.dataset == 'cifar10' else 100

# ----------- Offline Feature Summarization ------------

print('Error (%)\t\ttest')
err_cls = test(teloader, net)[0]
print(('Epoch %d:' %(0)).ljust(24) +
            '%.2f\t\t' %(err_cls*100))

# ----------- Improved Test-time Training ------------

find_prototypes = FindPrototypes(classifier.fc, n_sample=1)
prototypes, _ = find_prototypes.inference()
del find_prototypes


with torch.no_grad():
    bias = torch.linalg.svdvals(prototypes.t().cov())[0] / 30.
    # components
    ext_src_mu = prototypes
    ext_src_mu_scale = torch.norm(ext_src_mu, dim=1, keepdim=True)
    ext_src_cov = torch.eye(2048).cuda()[None, :, :].expand(class_num, -1, -1) * bias
    # global
    mu_src_ext = torch.mean(prototypes, dim=0)
    mu_src_ext_scale = torch.norm(mu_src_ext, dim=0, keepdim=True)
    cov_src_ext = prototypes.t().cov()

    bias = cov_src_ext.max() / 30.
    template_ext_cov = torch.eye(2048).cuda() * bias


source_component_distribution = torch.distributions.MultivariateNormal(ext_src_mu, ext_src_cov)
target_compoent_distribution = torch.distributions.MultivariateNormal(ext_src_mu, ext_src_cov)

is_both_activated=False
sample_predict_ema_logit = torch.zeros(len(tr_dataloader.dataset), class_num, dtype=torch.float)
sample_predict_alpha = torch.ones(len(tr_dataloader.dataset), dtype=torch.float)
ema_alpha = 0.9

ema_n = torch.zeros(class_num).cuda()
ema_ext_mu = ext_src_mu.clone().cpu()
ema_ext_cov = ext_src_cov.clone().cpu()

ema_ext_total_mu_weak = torch.zeros(2048).float().cuda()
ema_ext_total_cov_weak = torch.zeros(2048, 2048).float().cuda()


ema_total_n = 0.
if class_num == 10:
    ema_length = 128
    epoch_bias = 10
else: 
    ema_length = 64
    epoch_bias = 10

if class_num == 10:
    loss_scale = 0.05
else:
    loss_scale = 0.5

for epoch in range(1, args.nepoch+1):
    tic = time.time()

    head.eval()
    ext.train()
    classifier.eval()

    sample_predict_alpha = torch.where(sample_predict_alpha < 1, sample_predict_alpha + 0.2, torch.ones_like(sample_predict_alpha))

    for batch_idx, (inputs, _) in enumerate(tr_dataloader):
        optimizer.zero_grad()

        bsz = inputs[0].shape[0]

        if args.with_ssl:
            loss = 0.
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
                strong_logits = weak_logits.clone()
            
            pro, target = weak_logits.softmax(dim=-1).detach().max(dim=-1)
            mask = pro.ge(0.95).float()
            loss += (F.cross_entropy(strong_logits, target, reduction="none") * mask).mean() * loss_scale * 20.
            loss.backward()
            del loss

        if is_both_activated and args.align_ext:
            loss = 0.
            if args.align_sample == 'weak+strong':
                images = torch.cat([inputs[0], inputs[1]], dim=0)
                images = images.cuda(non_blocking=True)
                feats = ext(images)
                weak_feats, strong_feats = torch.split(feats, [bsz, bsz], dim=0)
            elif args.align_sample == 'weak':
                inputs_0 = inputs[0].cuda(non_blocking=True)
                weak_feats = ext(inputs_0)
            elif args.align_sample == 'none':
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
                    feat_ext2 = torch.cat([weak_feats[pseudo_label_mask], strong_feats[pseudo_label_mask]], dim=0)
                    pseudo_label2 = torch.cat([pseudo_label[pseudo_label_mask], pseudo_label[pseudo_label_mask]], dim=0).cuda()
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
                    for label in pseudo_label2.unique():
                        feat_ext_per_category = feat_ext2[pseudo_label2 == label, :]

                        b = feat_ext_per_category.shape[0]
                        ema_n[label] += b
                        alpha = 1. / ema_length if ema_n[label] > ema_length else 1. / ema_n[label]

                        ema_ext_mu_that = ema_ext_mu[label, :].cuda()
                        ema_ext_cov_that = ema_ext_cov[label, :, :].cuda()
                        delta_pre = feat_ext_per_category - ema_ext_mu_that

                        delta = alpha * delta_pre.sum(dim=0)
                        tmp_mu = ema_ext_mu_that + delta

                        with torch.no_grad():
                            ema_ext_mu[label, :] = tmp_mu.detach().cpu()

                        if epoch > epoch_bias:
                            source_domain = torch.distributions.MultivariateNormal(ext_src_mu[label, :], ext_src_cov[label, :, :])
                            target_domain = torch.distributions.MultivariateNormal(tmp_mu / torch.norm(tmp_mu, dim=0, keepdim=True).detach() * ext_src_mu_scale[label, :], ext_src_cov[label, :, :])
                            loss += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) * loss_scale / class_num

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
                    scale = mu_src_ext_scale / torch.norm(tmp_mu_weak, dim=0, keepdim=True)
                    target_domain = torch.distributions.MultivariateNormal(tmp_mu_weak * scale.detach(), tmp_cov_weak * scale.pow(2).detach() + template_ext_cov)
                    loss += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) * loss_scale
                
                if type(loss) != float:
                    loss.backward()
                    del loss
                    
        if epoch > args.bnepoch:
            optimizer.step()
            optimizer.zero_grad()

    err_cls = test(teloader, net)[0]
    all_err_cls.append(err_cls)

    toc = time.time()
    
    # both components
    if not is_both_activated and epoch > args.bnepoch:
        is_both_activated = True

    # termination
    if epoch > (args.stopepoch + 1) and all_err_cls[-args.stopepoch] < min(all_err_cls[-args.stopepoch+1:]):
        print("{} Termination: {:.2f}".format(args.corruption, all_err_cls[-args.stopepoch]*100))
        break

    # save
    if epoch > args.bnepoch and len(all_err_cls) > 2 and all_err_cls[-1] < min(all_err_cls[:-1]):
        state = {'net': net.state_dict(), 'head': head.state_dict()}
        save_file = os.path.join(args.outf, args.corruption + '.pth')
        torch.save(state, save_file)
        print('Save model to', save_file)

    # lr decay
    scheduler.step(err_cls)

