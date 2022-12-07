import numpy as np
import torch

import torch
import torch.utils.data.dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


def get_batch_jacobian_cov(net, x, target, device, split_data):
    x.requires_grad_(True)

    N = x.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data
        y = net(x[st:en])
        y.backward(torch.ones_like(y))

    jacob = x.grad.detach()
    x.requires_grad_(False)
    return jacob, target.detach()

def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return np.sum(np.log(v + k) + 1./(v + k))


def compute_jacob_cov(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    jacobs, labels = get_batch_jacobian_cov(net, inputs, targets, device, split_data=split_data)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    try:
        jc = eval_score(jacobs, labels)
    except Exception as e:
        print(e)
        jc = np.nan

    return jc













selector = ['hook_logdet']


def hooklogdet(K, labels=None):
    # C=1e-6
    C=1
    # print ("K.shape", K.shape)
    dimInput=K.shape[0]
    # print ("dimInput", dimInput)
    I = torch.eye(dimInput)
    I = I.numpy()
    s, ld = np.linalg.slogdet(K + C *I)
    print ("s", s)
    print ("ld", ld)
    return ld

def random_score(jacob, label=None):
    return np.random.normal()


_scores = {
        'hook_logdet': hooklogdet,
        'random': random_score
        }

def get_score_func(score_name):
    return _scores[score_name]




def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    print ("x.shape", x.shape)
    # y, out = net(x)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    # return jacob, target.detach(), y.detach(), out.detach()
    return jacob, target.detach(), y.detach()


def score_net (network, train_sample_array, train_label_array, device, batch_size):

    network.K = np.zeros((batch_size, batch_size))
    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1.-x) @ (1.-x.t())
            # print ("x", x)
            # print ("K", K)
            network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
        except:
            pass

        
    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

        
    # for name, module in network.named_modules():
    #     if 'ReLU' in str(type(module)):
    #         #hooks[name] = module.register_forward_hook(counting_hook)
    #         module.register_forward_hook(counting_forward_hook)
    #         module.register_backward_hook(counting_backward_hook)

    for name, module in network.named_modules():
        if 'ReLU' in str(type(module)):
            #hooks[name] = module.register_forward_hook(counting_hook)
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)




    # network = network.to(device)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)


    s = []
    for j in range(1):
        # data_iterator = iter(train_loader)

        # x = train_sample_array[0]
        # target = train_label_array[0]

        x = train_sample_array
        target = train_label_array


        # x, target = next(data_iterator)
        x2 = torch.clone(x)
        x2 = x2.to(device)

        target = target.to(device)
        # x, target = x.to(device), target.to(device)
        # jacobs, labels, y, out = get_batch_jacobian(network, x, target, device)
        jacobs, labels, y = get_batch_jacobian(network, x, target, device)


        # print ("x2", x2)
        # network(x2.to(device))
        # s.append(get_score_func('hook_logdet')(network.K, target))

        if 'hook_logdet' in selector:
            network(x2.to(device))
            s.append(get_score_func('hook_logdet')(network.K, target))
        else:
            s.append(get_score_func('random')(jacobs, labels))

    score = np.mean(s)

    return score




