import argparse
import os
import torch
import torch.optim as optim
import torch.utils.data as data
# ----------------------------------

import random
import numpy as np
import torch.backends.cudnn as cudnn

from utils.test_helpers import build_model, test
from utils.prepare_dataset import prepare_transforms, create_dataloader, ImageNetCorruption, ImageNet_
from utils.offline import offline
import torch.nn.functional as F
from utils.find_prototypes import FindPrototypes

from functools import partial
# ----------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default=None)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--adapt_batch_size', default=128, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--iters', default=2, type=int)
parser.add_argument('--corruption', default='snow')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--with_ssl', action='store_true', default=False)

args = parser.parse_args()

print(args)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

cudnn.benchmark = True

########### build and load model #################
net, ext, classifier = build_model()

# ########### create dataset and dataloader #################
train_transform, val_transform, val_corrupt_transform, fixmatch_transform = prepare_transforms()

source_dataset = ImageNet_(args.dataroot, 'val', transform=val_transform, is_carry_index=True)

target_dataset_adapt = ImageNetCorruption(args.dataroot, args.corruption, transform=fixmatch_transform, is_carry_index=True)
target_dataset_test = ImageNetCorruption(args.dataroot, args.corruption, transform=val_corrupt_transform, is_carry_index=True)

source_dataloader = create_dataloader(source_dataset, args, True, False)
target_dataloader_test = create_dataloader(target_dataset_test, args, True, False)

class_num = 1000

if os.path.exists('learned_prototypes.pth') is False:
    find_prototypes = FindPrototypes(classifier, n_sample=1)
    prototypes, _ = find_prototypes.inference(num_step=1000)
    del find_prototypes

    torch.save({'prototypes': prototypes}, 'learned_prototypes.pth')
else:
    tmp = torch.load('learned_prototypes.pth')
    prototypes = tmp['prototypes']

with torch.no_grad():
    bias = torch.linalg.svdvals(prototypes.t().cov())[0] / 30.
    # components
    ext_mean_categories = prototypes
    ext_src_mu_scale = torch.norm(ext_mean_categories, dim=1, keepdim=True)
    ext_src_cov = torch.eye(2048).cuda() * bias
    # global
    ext_mean = torch.mean(prototypes, dim=0)
    mu_src_ext_scale = torch.norm(ext_mean, dim=0, keepdim=True)
    cov_src_ext = prototypes.t().cov()

    bias = cov_src_ext.max().item() / 30.
    template_ext_cov = torch.eye(2048).cuda() * bias

########### create optimizer #################
optimizer = optim.SGD(ext.parameters(), lr=args.lr, momentum=0.9)

########### test before TTT #################
print('Error (%)\t\ttest')
err_cls = test(target_dataloader_test, net)[0]
print(('Epoch %d:' %(0)).ljust(24) +
            '%.2f\t\t' %(err_cls*100))
# ########### TTT #################

is_both_activated = False

sample_predict_ema_logit = torch.zeros(len(target_dataset_adapt), class_num, dtype=torch.float)
sample_alpha = torch.ones(len(target_dataset_adapt), dtype=torch.float)

ema_alpha = 0.9
ema_ext_mu = torch.zeros(class_num, 2048).cuda()
ema_ext_cov = torch.zeros(class_num, 2048).cuda()
ema_n = torch.zeros(class_num).cuda()
ema_ext_total_mu = torch.zeros(2048).cuda()
ema_ext_total_cov = torch.zeros(2048, 2048).cuda()
ema_total_n = 0.

class_ema_length = 64
loss_scale = 0.05
mini_batch_length = 4096

mini_batch_indices = []
correct = []

for te_batch_idx, (te_inputs, te_labels) in enumerate(target_dataloader_test):
    mini_batch_indices.extend(te_inputs[-1].tolist())
    mini_batch_indices = mini_batch_indices[-mini_batch_length:]
    print('mini_batch_length:', len(mini_batch_indices))
    try:
        del target_adapt_subset
        del target_dataloader_adapt
    except:
        pass

    target_adapt_subset = data.Subset(target_dataset_adapt, mini_batch_indices)
    args.batch_size = args.adapt_batch_size
    target_dataloader_adapt = create_dataloader(target_adapt_subset, args, True, True)

    net.train()
    for iter_id in range(min(args.iters, int(len(mini_batch_indices) / 256) + 1) + 1):
        if iter_id > 0:
            sample_alpha = torch.where(sample_alpha < 1, sample_alpha + 0.2, torch.ones_like(sample_alpha))
        
        for batch_idx, (inputs, labels) in enumerate(target_dataloader_adapt):
            optimizer.zero_grad()

            if args.with_ssl:
                bsz = inputs[0].shape[0]
                images = torch.cat([inputs[0], inputs[1]], dim=0)
                images = images.cuda(non_blocking=True)
                feats = ext(images)
                logits = classifier(feats)
                weak_logits, strong_logits = torch.split(logits, [bsz, bsz], dim=0)

                pro, target = weak_logits.softmax(dim=-1).detach().max(dim=-1)
                mask = pro.ge(0.95).float()
                loss = (F.cross_entropy(strong_logits, target, reduction="none") * mask).mean() * loss_scale * 20.
                loss.backward()
                del loss

            ####### feature alignment ###########
            loss = 0.
            input_0, indexes = inputs[0], inputs[-1]
            input_0 = input_0.cuda()

            feat_ext = ext(input_0)
            with torch.no_grad():
                net.eval()
                origin_images = input_0
                origin_image_index = indexes
                predict_logit = net(origin_images)
                softmax_logit = predict_logit.softmax(dim=1).cpu()

                old_logit = sample_predict_ema_logit[origin_image_index, :]
                max_val, max_pos = softmax_logit.max(dim=1)
                old_max_val = old_logit[torch.arange(max_pos.shape[0]), max_pos]
                accept_mask = max_val > (old_max_val - 0.01)

                sample_alpha[origin_image_index] = torch.where(accept_mask, sample_alpha[origin_image_index], torch.zeros_like(accept_mask).float())

                sample_predict_ema_logit[origin_image_index, :] = \
                    torch.where(sample_predict_ema_logit[origin_image_index, :] == torch.zeros(class_num), \
                                softmax_logit, \
                                (1 - ema_alpha) * sample_predict_ema_logit[origin_image_index, :] + ema_alpha * softmax_logit)
                
                pro, pseudo_label = sample_predict_ema_logit[origin_image_index].max(dim=1)

                net.train()
                del predict_logit

            pseudo_label_mask = (pro > 0.9) & (sample_alpha[origin_image_index] == 1)
            feat_ext2 = feat_ext[pseudo_label_mask]
            pseudo_label2 = pseudo_label[pseudo_label_mask].cuda()

            # Gaussian Mixture Distribution Alignment
            b, d = feat_ext2.shape
            feat_ext2_categories = torch.zeros(class_num, b, d).cuda() # K, N, D
            feat_ext2_categories.scatter_add_(dim=0, index=pseudo_label2[None, :, None].expand(-1, -1, d), src=feat_ext2[None, :, :])

            num_categories = torch.zeros(class_num, b, dtype=torch.int).cuda() # K, N
            num_categories.scatter_add_(dim=0, index=pseudo_label2[None, :], src=torch.ones_like(pseudo_label2[None, :], dtype=torch.int))

            ema_n += num_categories.sum(dim=1) # K
            alpha = torch.where(ema_n > class_ema_length, torch.ones(class_num, dtype=torch.float).cuda() / class_ema_length, 1. / (ema_n + 1e-10))

            delta_pre = (feat_ext2_categories - ema_ext_mu[:, None, :]) * num_categories[:, :, None] # K, N, D
            delta = alpha[:, None] * delta_pre.sum(dim=1) # K, D
            ext_mu_categories = ema_ext_mu + delta
            with torch.no_grad():
                ema_ext_mu = ext_mu_categories.detach()

            if iter_id > int(args.iters / 2):
                for label in pseudo_label2.unique():
                    if ema_n[label] > class_ema_length:
                        source_domain = torch.distributions.MultivariateNormal(ext_mean_categories[label, :], ext_src_cov)
                        target_domain = torch.distributions.MultivariateNormal(ext_mu_categories[label, :] / torch.norm(ext_mu_categories[label, :], dim=0, keepdim=True).detach() * ext_src_mu_scale[label, :], ext_src_cov)
                        loss += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) * loss_scale / class_num

            # Gaussian Distribution Alignment
            b = feat_ext.shape[0]
            ema_total_n += b
            alpha = 1. / 1280 if ema_total_n > 1280 else 1. / ema_total_n
            delta = alpha * (feat_ext - ema_ext_total_mu).sum(dim=0)
            tmp_mu = ema_ext_total_mu + delta
            tmp_cov = ema_ext_total_cov + alpha * ((feat_ext - ema_ext_total_mu).t() @ (feat_ext - ema_ext_total_mu) - b * ema_ext_total_cov) - delta[:, None] @ delta[None, :]

            with torch.no_grad():
                ema_ext_total_mu = tmp_mu.detach()
                ema_ext_total_cov = tmp_cov.detach()

            source_domain = torch.distributions.MultivariateNormal(ext_mean, cov_src_ext + template_ext_cov)
            cov_scale = (mu_src_ext_scale / torch.norm(tmp_mu, dim=0, keepdim=True)).pow(2)
            target_domain = torch.distributions.MultivariateNormal(tmp_mu / torch.norm(tmp_mu, dim=0, keepdim=True).detach() * mu_src_ext_scale, tmp_cov * cov_scale.detach() + template_ext_cov)
            loss += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) * loss_scale

            loss.backward()
            del loss

            if iter_id > 0:
                optimizer.step()
                optimizer.zero_grad()
    #### Test ####
    net.eval()
    with torch.no_grad():
        outputs = net(te_inputs[0].cuda())
        _, predicted = outputs.max(1)
        correct.append(predicted.cpu().eq(te_labels))

    print('BATCH: %d/%d' % (te_batch_idx + 1, len(target_dataloader_test)), 'instance error:', 1 - torch.cat(correct).numpy().mean())
    net.train()

print(args.corruption, 'Test time training result:', 1 - torch.cat(correct).numpy().mean())




