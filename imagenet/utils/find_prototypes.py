import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class FindPrototypes(object):
    def __init__(self, classifier, n_sample=1):
        super().__init__()
        self.prototypes = Prototype(classifier.weight, n_sample)
        self.classifier = classifier
        self.optimizer = torch.optim.LBFGS(self.prototypes.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def closure(self):
        self.optimizer.zero_grad()
        logits = self.prototypes(self.classifier)
        loss = self.criterion(logits, self.prototypes.get_label()).reshape(self.prototypes.n_class, self.prototypes.n_sample).sum(dim=-1).mean()
        loss.backward()
        return loss
        
    def train_step(self):
        self.prototypes.train()
        self.optimizer.step(self.closure)
        return
    
    def inference(self, hook=None, num_step=10000, verbose=False):
        for iter in range(num_step):
            self.train_step()
            if verbose:
                print('inference step: %05d/%05d | ' % (iter+1, num_step), end='')
            if hook is not None:
                hook(prototype=self.prototypes.get_prototype())
            elif verbose:
                print('')
        del self.optimizer
        torch.cuda.empty_cache()
        return self.prototypes.get_prototype(), self.prototypes.get_prototype_diverse()

class Prototype(nn.Module):
    def __init__(self, classifier_weight, n_sample:int):
        super().__init__()
        self.n_class, self.dim = classifier_weight.shape
        self.n_sample = n_sample
        self.prototype = Parameter(torch.randn([self.n_class * self.n_sample, self.dim], device=classifier_weight.device))
        nn.init.kaiming_uniform_(self.prototype, a=math.sqrt(5))
        self.labels = torch.arange(self.n_class, dtype=torch.long, device=classifier_weight.device)[:, None].expand(-1, self.n_sample).reshape(-1)

    def forward(self, classifier):
        return classifier(torch.abs(self.prototype))

    def get_prototype(self):
        prototype = torch.abs(self.prototype).reshape(self.n_class, self.n_sample, self.dim)
        return prototype.mean(dim=1).detach()
    
    def get_prototype_diverse(self):
        prototype = torch.abs(self.prototype).reshape(self.n_class, self.n_sample, self.dim)
        covs = []
        for i in range(self.n_class):
            n_prototype = prototype[i, :, :] # n_sample, dim
            cov = torch.diagonal(n_prototype.t().cov())
            covs.append(cov)
        return torch.stack(covs).detach()

    def get_label(self):
        return self.labels