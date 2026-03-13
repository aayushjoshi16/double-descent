import sys
sys.path.insert(0, '/root/double-descent/hessian-eff-dim')

import torch
import hess
from hess.nets import SimpleNet
import hess.loss_surfaces as loss_surfaces
import hess.utils as utils

def get_model(hidden_size=20, n_hidden=5):
    in_dim = 2
    out_dim = 1
    model = hess.nets.SimpleNet(in_dim, out_dim, n_hidden=n_hidden, hidden_size=hidden_size,
                         activation=torch.nn.ELU(), bias=True)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
    if use_cuda:
        model=model.cuda()
        
    return model

def train_model(model, train_x, train_y):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.BCEWithLogitsLoss()

    losses = []
    trainL = -1

    for step in range(2000):
        optimizer.zero_grad()
        outputs = model(train_x)

        loss=loss_func(outputs, train_y)
        trainL = loss.detach().item()
        # if step % 500 is 0:
        #     print("train loss = ", trainL)
        losses.append(trainL)
        loss.backward()
        optimizer.step()
    # print("train loss = ", trainL)
    return losses

import hess.utils as utils

def get_hessian(model, train_x, train_y):
    n_par = sum(torch.numel(p) for p in model.parameters())

    hessian = torch.zeros(n_par, n_par)
    for pp in range(n_par):
        base_vec = torch.zeros(n_par).unsqueeze(0)
        base_vec[0, pp] = 1.

        base_vec = utils.unflatten_like(base_vec, model.parameters())
        utils.eval_hess_vec_prod(base_vec, list(model.parameters()), model,
                                criterion=torch.nn.BCEWithLogitsLoss(),
                                inputs=train_x, targets=train_y, use_cuda=True)
        if pp == 0:
            output = utils.gradtensor_to_tensor(model, include_bn=True)
            hessian = torch.zeros(output.nelement(), output.nelement())
            hessian[:, pp] = output

        hessian[:, pp] = utils.gradtensor_to_tensor(model, include_bn=True).cpu()
        
    return hessian

def run_single_experiment(args):
    rep, hidden_size, model_depth, train_x, train_y, test_x, test_y, device = args
    
    model = get_model(n_hidden=model_depth, hidden_size=hidden_size).to(device)
    losses = train_model(model, train_x, train_y)
    hessian = get_hessian(model, train_x, train_y).detach()
    eigs = torch.linalg.eigvals(hessian).cpu().numpy()
    
    with torch.no_grad():
        test_loss = torch.nn.BCEWithLogitsLoss()(model(test_x), test_y)
        test_loss_val = test_loss.item()
        
    n_par = sum(p.numel() for p in model.parameters())
    
    del model, hessian
    torch.cuda.empty_cache()
    
    # print(f"rep = {rep}, n_hidden = {i}, test_loss = {test_loss_val}, n_par = {n_par}")
    return losses, eigs, test_loss_val, n_par
