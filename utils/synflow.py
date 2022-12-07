# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_layer_metric_array(net, metric, mode): 
    metric_array = []
    # for layer in net.modules():

    #     print ("layer: ", layer)
    for layer in net.modules():
        if mode=='channel' and hasattr(layer,'dont_ch_prune'):
            continue
        # if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #     metric_array.append(metric(layer))
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))
    

    
    return metric_array

def compute_synflow_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None):

    device = inputs.device

    #convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    #convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net)
    
    # Compute gradients with input of 1s 
    net.zero_grad()
    # net.double()
    input_dim = list(inputs[0,:].shape)
    # inputs = torch.ones([1] + input_dim).double().to(device)
    inputs = torch.ones([1] + input_dim).to(device)
    output = net.forward(inputs)
    torch.sum(output).backward() 

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)
            

    grads_abs = get_layer_metric_array(net, synflow, mode)

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs