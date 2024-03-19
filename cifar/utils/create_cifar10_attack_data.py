from advertorch.attacks import LinfPGDAttack
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
sys.path.append('./')
from utils.test_helpers import *
import numpy as np

import os

DATAROOT = os.path.join(os.path.dirname(__file__), '../data')
BATCHSIZE = 256
CHECKPOINT = os.path.join(os.path.dirname(__file__), '../results/cifar10_joint_resnet50')

class Args(object):
    
    # ResNet
    dataset = "cifar10"
    resume = CHECKPOINT
    ckpt = None

    # PDG
    eps = 8. / 255.
    nb_iter=40
    eps_iter=0.01
    rand_init=True
    clip_min=0.0
    clip_max=1.0
    targeted=False

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
normalize = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=mean, std=std)
            ])
mean_ = torch.Tensor(mean).cuda()
std_ = torch.Tensor(std).cuda()
reverse_normalize = lambda data: data.permute(0, 2, 3, 1).mul(255).to(torch.uint8)


args = Args()
net, ext, head, ssh, classifier = build_resnet50(args)
load_resnet50(net, head, ssh, classifier, args)

teset = CIFAR10(root=DATAROOT, train=False, download=True, transform=normalize)
teloader = DataLoader(teset, BATCHSIZE, False)
net.eval()

adversary = LinfPGDAttack(
    net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.eps,
    nb_iter=args.nb_iter, eps_iter=args.eps_iter, rand_init=args.rand_init, clip_min=args.clip_min, clip_max=args.clip_max,
    targeted=args.targeted)

adv_dataset = []
for batch_idx, (data, target) in enumerate(teloader):
    adv_data = adversary.perturb(data.cuda(), target.cuda()).detach()
    adv_data = reverse_normalize(adv_data)
    adv_dataset.append(adv_data.cpu())
    print('process: %d / %d;\r' % (batch_idx + 1, len(teloader)), end='')

os.makedirs(os.path.join(DATAROOT, 'CIFAR-10-Attack'), exist_ok=True)
np.save(os.path.join(DATAROOT, 'CIFAR-10-Attack', 'pgd2.npy'), torch.concat(adv_dataset, dim=0).numpy())