"""
General-purpose class for interacting with Sagittarius model.
"""

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.join(sys.path[0], '../'))
from model.Sagittarius import Sagittarius


def check_for_early_termination(val_losses, min_epochs=250):
    """
    Parameters:
        val_losses (List[float]): validation losses for previous epochs
        min_epochs (int): the number of epochs we must train for before termination
    
    Returns:
        True iff the model has not (globally) improved validation loss for 250 epochs
    """
    if len(val_losses) < min_epochs:
        return False
    return np.argmin(val_losses) < len(val_losses) - min_epochs


class Sagittarius_Manager():
    """
    Manager interaction class for Sagittarius model.
    """
    def __init__(self, input_dim, num_classes, class_sizes, cvae_catdims,
                 cvae_hiddendims, cvae_ld, attn_heads, num_ref_points, temporal_dim,
                 tr_catdims, minT, maxT, device, transformer_dim=None, batch_size=8,
                 beta=1.0, train_transfer=False, num_cont=None,
                 rec_loss='mse'):
        self.train_losses = {'loss': []}
        self.val_losses = {'loss': []}
        """
        Parameters:
            input_dim (int): dimension of measurements
            num_classes (int): number of experimental variables
            class_sizes (List[int]): number of options per experimental variable;
                must have length `num_classes`
            cvae_catdims (List[int]): embedding dimension per experimental variable
                in Sagittarius's initial encoder; must have length `num_classes`
            cvae_hiddendims (List[int]): hidden dimensions to use for symmetric MLP
                encoder/decoder; len(`cvae_hiddendims`) indicates the number of
                hidden layers
            cvae_ld (int): latent dim to use for encoder/decoder's gaussian space
            attn_heads (int): number of heads H for multi-head attention
            num_ref_points (int): number of regularly-spaced reference points S+1 to
                use in transformer
            temporal_dim (int): embedding dimension V to use for temporal variable
                in Sagittarius's transformer
            tr_catdims (List[int]): embedding dimension to use for each experimental
                variable in Sagittarius's transformer
            minT (int): minimum time point to use as anchor of reference space
            maxT (int): maximum time point to use as anchor of reference space
            device (str): device to use for model
            transformer_dim (int): dimension to use for transformer latent space; if
                None, use `cvae_ld`
            batch_size (int): batch size to use for training/inference
            beta (float): KL divergence regularizing weight
            train_transfer (bool): True iff we include a generation objective during
                training
            num_cont (int): number of continuous variables; can be None if only 1
                "time" variable
            other_temporal_dims (List[int]): embedding dimension V_b to use for the
                bth temporal variable with b>0 in Sagittarius's transformer; must
                have length `num_cont` - 1; can be None if `num_cont` = 1
            other_minT (List[int]): minimum time point to use in anchor space for 
                the bth temporal variable with b>0 in Sagittarius's transformer;
                must have length `num_cont` - 1; can be None if `num_cont` = 1
            other_maxT (List[int]): maximum time point to use in anchor space for 
                the bth temporal variable with b>0 in Sagittarius's transformer;
                must have length `num_cont` - 1; can be None if `num_cont` = 1
            rec_loss (str): reconstruction loss type to use; either 'mse' for mean 
                squared error or bce for binary cross entropy
        """
        self.model = Sagittarius(
            input_dim, num_classes, class_sizes, latent_dim=cvae_ld,
            temporal_dim=temporal_dim, cat_dims=tr_catdims,
            cvae_hidden_dims=cvae_hiddendims, num_heads=attn_heads,
            num_ref_points=num_ref_points, tr_dim=transformer_dim,
            cvae_yemb_dims=cvae_catdims, minT=minT, maxT=maxT, device=device,
            num_cont=num_cont, rec_loss=rec_loss
        ).to(device)
        self.device = device
        self.batch_size = batch_size
        self.train_transfer = train_transfer
        self.beta = beta

        # placeholders for variables to compute
        self.full_expr = None
        self.full_mask = None
        self.repeated_classes = None
        self.single_classes = None
        self.full_times = None
        self.predicted_expr = None

    def get_latent_variables(self, expr, ts, ys, mask):
        mu_list = []
        logvar_list = []
        for bstart in range(0, len(expr), self.batch_size):
            bend = min(len(expr), bstart + self.batch_size)
            x = torch.tensor(expr[bstart:bend], dtype=torch.float32).to(self.device)
            y = [y[bstart:bend].long().to(self.device) for y in self.repeated_classes]
            xhat, mu, logvar = self.model(x, ts[bstart:bend].float().to(self.device), y, mask[bstart:bend].float().to(self.device))
            mu_list.append(mu)
            logvar_list.append(logvar)
        mu = torch.cat(mu_list, dim=0)
        logvar = torch.cat(logvar_list, dim=0)
        return mu, logvar


    def train_model(self, expr, ts, ys, mask, reload, mfile, val_mask=None, transfer_expr=None,
                    transfer_ts=None, transfer_ys=None, transfer_mask=None, num_epochs=9000, lr=1e-3):
        """
        Train the Sagittarius model given the dataset.
        
        Sets the model to eval() status afterwards.
        
        Parameters:
            expr (Tensor): input expression, shape N x T x M
            ts (Tensor): input time points, shape N x T
            ys (List[Tensor]): input experimental variables, shape N x T;
                must have length `num_classes`
            mask (Tensor): input mask, shape N x T, where `mask[i, t] = 1` indicates
                that `expr[i, t]` was measured AND is part of the training set
            reload (bool): True iff we should load a pre-trained model file from
                `mfile` and skip training
            mfile (str): model file to save to/load from
            val_mask (Tensor): mask to use for validation data; `val_mask[i, t] = 1`
                indicates that `expr[i, t]` was measured AND is part of the
                validation set
            transfer_expr (Tuple[Tensor]): 0th entry are source expression
                measurements to use to generate 1st entry target expression
                measurements; both have shape N x T x M. Must be non-None if
                `train_transfer`
            transfer_ts (Tuple[Tensor]): 0th entry are source time points to use to
                generate 1st entry target time points; both have shape N x T. Must
                be non-None if `train_transfer`
            transfer_ys (Tuple[List[Tensor]]): 0th entry are source experimental
                variables to use to generate 1st entry target experimental 
                variables; both have shape N x T. Must be non-None if 
                `train_transfer`
            transfer_mask (Tuple[Tensor]): 0th entry are source masks to use to
                generate 1st entry with target mask; both have shape N x T, with
                `transfer_mask[x][i, t]` = 1 indicates that the 
                `transfer_expr[x][i, t]` was measured. Must be non-None if
                `train_transfer`
            num_epochs (int): maximum number of training epochs to consider
            lr (float): learning rate to use
        """

        self.full_expr = expr.to(self.device)


        self.full_mask = mask.to(self.device)  # N x T
        self.full_times = ts.to(self.device)  # N x T
        self.single_classes = ys


        


        N, T, M = self.full_expr.shape
        self.repeated_classes = [
            torch.stack([y for _ in range(T)], dim=1) for y in self.single_classes]

        
        if '/' in mfile:  # nested directory
            mdir = '/'.join(mfile.split('/')[:-1]) + '/'
            if not os.path.exists(mdir):
                os.makedirs(mdir)
        else:
            mdir = ''

        if reload:
            self.model.load_state_dict(torch.load(mfile, map_location=self.device))
        else:  # train the model
            losses = {}
            val_losses = {}
            tr_opt = torch.optim.Adam(lr=lr, params=self.model.parameters())
            
            saved_model = False

            for ep in tqdm(range(num_epochs)):
                if val_mask is not None:  # using validation!
                    if ep > 0 and check_for_early_termination(val_losses['loss']):
                        print('Terminating after {} epochs.'.format(ep))
                        self.model.load_state_dict(torch.load(
                            mdir + 'tmp_model.pth', map_location=self.device))
                        torch.save(self.model.state_dict(), mfile)
                        saved_model = True
                        break
                ep_losses = {}
                ep_val_losses = {}
                for bstart in range(0, N, self.batch_size):
                    bend = min(N, bstart + self.batch_size)

                    x = torch.tensor(expr[bstart:bend], dtype=torch.float32).to(self.device) # Numpy don't have Float, unlike tensors
                    # x = expr[bstart:bend].float().to(self.device)

                    y = [y[bstart:bend].long().to(self.device) 
                         for y in self.repeated_classes]
                    xhat, mu, logvar = self.model(
                        x, ts[bstart:bend].float().to(self.device), y, 
                        mask[bstart:bend].float().to(self.device))

                    loss = self.model.loss_fn(
                        x, xhat, mu, logvar, beta=self.beta)
                    """
                    print("Keys in the loss dictionary:", loss.keys())
                    if 'kl_loss' in loss:
                        kl_loss = loss['kl_loss']
                        print(f"KL Loss: {kl_loss.item()}")

                    if 'reconstruction_loss' in loss:
                        reconstruction_loss = loss['reconstruction_loss']
                        print(f"Reconstruction Loss: {reconstruction_loss.item()}")
                    """


                    for k in loss:
                        if k not in ep_losses:
                            ep_losses[k] = []
                            val_losses[k] = []
                        ep_losses[k].append(loss[k])
                        
                    tr_opt.zero_grad()
                    loss['loss'].backward()
                    tr_opt.step()
                    
                    # potentially repeat for transfer generation task!
                    if self.train_transfer:
                        bendTr = min(len(transfer_expr[0]), 
                                     bstart + self.batch_size)
                        if bendTr <= bstart:  # nothing else to do here
                            continue
                        xgen, mu, logvar = self.model.generate(
                            transfer_expr[0][bstart:bendTr].float().to(self.device),
                            transfer_ts[0][bstart:bendTr].float().to(self.device),
                            transfer_ts[1][bstart:bendTr].float().to(self.device),
                            [torch.stack([y[bstart:bendTr] for _ in range(T)],
                                         dim=1).long().to(self.device)
                             for y in transfer_ys[0]],
                            [torch.stack([y[bstart:bendTr] for _ in range(T)], 
                                         dim=1).long().to(self.device)
                             for y in transfer_ys[1]],
                            transfer_mask[0][bstart:bendTr].float().to(self.device))

                        loss = self.model.loss_fn(
                            transfer_expr[1][bstart:bendTr].float().to(self.device),
                            xgen, mu, logvar, beta=self.beta)
                        for k in loss:
                            tr_k = 'transfer_{}'.format(k)
                            if tr_k not in ep_losses:
                                ep_losses[tr_k] = []
                            ep_losses[tr_k].append(loss[k])
                        tr_opt.zero_grad()
                        loss['loss'].backward()
                        tr_opt.step()

                    # now, run validation check (if applicable)
                    if val_mask is not None:
                        self.model.eval()
                        valbend = min(bend, len(val_mask))
                        if valbend <= bstart:
                            continue  # nothing else to do here!
                        xhat, mu, logvar = self.model(
                            x, ts[bstart:valbend].float().to(self.device), y,
                            val_mask[bstart:valbend].float().to(self.device))
                        loss = self.model.loss_fn(
                            x, xhat, mu, logvar)
                        for k in loss:
                            if k not in ep_val_losses:
                                ep_val_losses[k] = []
                            ep_val_losses[k].append(loss[k])
                        self.model.train()

                for k in ep_losses:
                    if k not in losses:
                        losses[k] = []
                        self.train_losses[k] = []
                    self.train_losses[k].append(sum(ep_losses[k]).item())
                    losses[k].append(sum(ep_losses[k]).item())
                if val_mask is not None:
                    for k in ep_val_losses:
                        if k not in val_losses:
                            val_losses[k] = []
                        val_losses[k].append(sum(ep_val_losses[k]).item())
                    if val_losses['loss'][-1] == min(val_losses['loss']):  # best epoch so far
                        torch.save(self.model.state_dict(), mdir + 'tmp_model.pth')

            if val_mask is None:
                # Done training -> save the model file and make loss plots!
                torch.save(self.model.state_dict(), mfile)
            elif not saved_model:  # had validation set but never terminated
                self.model.load_state_dict(torch.load(mdir + 'tmp_model.pth', map_location=self.device))
                torch.save(self.model.state_dict(), mfile)
        self.model.eval()

    def reconstruct(self, maxN=None):
        """
        Reconstruct dataset input.
        
        Parameters:
            maxN (int): latest index in dataset to reconstruct.
            
        Returns:
            K x M Tensor of reconstructed expression, where M is the expression dimension and K is
                the number of total measurements in the dataset.
        """
        recons = []
        N, T, M = self.full_expr.shape
        self.maxN = maxN if maxN is not None else N
        for bstart in range(0, self.maxN, self.batch_size):
            bend = min(bstart + self.batch_size, self.maxN)
            xhat = self.model(
                self.full_expr[bstart:bend].float().to(self.device),
                self.full_times[bstart:bend].float().to(self.device),
                [y[bstart:bend].long().to(self.device) for y in self.repeated_classes],
                self.full_mask[bstart:bend].float().to(self.device))[0]
            recons.append(xhat)
        self.predicted_expr = torch.cat(recons, dim=0)  # N x T x M
        return torch.masked_select(
            self.predicted_expr, self.full_mask[:self.maxN].view(self.maxN, T, 1).bool()
        ).view(-1, M)
    
    
    def generate(self, rev_mask):
        """
        Simulate new data points. Must have called `reconstruct()` first.
        
        Parameters:
            rev_mask (Tensor): mask of shape N x T, where `rev_mask[i, t] = 1` indicates that
                Sagittarius should simulate a measurement for time `self.full_times[i, t]` for the
                ith sequence
                
        Returns:
            simulated expression vector for first `maxN` sequences with shape K x M, where M is the
                expression dimension and K is the number of simulation points 
                (torch.count_nonzero(`rev_mask`[:maxN]))
        """
        N, T, M = self.full_expr.shape
        return torch.masked_select(
            self.predicted_expr, rev_mask[:self.maxN].view(self.maxN, T, 1).bool()
        ).view(-1, M)

    def categorical_reconstruction(self, recon_mask, sequence_idx, partner_idx):
        """
        Simulate one sequence from a partner sequence at measured time points.
        
        Parameters:
            recon_mask (Tensor): points to reconstruct, `recon_mask[i, t]` = 1 indicates that the ith
                partner index at time `self.full_times[i, t]` should be generated
            sequence_idx (int): index in dataset to use as target sequence
            partner_idx (int): index in dataset to use as source sequence
            
        Returns:
            simulated expression from `partner_idx` base to `sequence_idx` experimental and temporal
                variables; shape K x M where M is the expression dimension and K is 
                `torch.count_nonzero(recon_mask)`
        """
        N, T, M = self.full_expr.shape

        gens = self.model.generate(self.full_expr[partner_idx].unsqueeze(0).float(),
                                   self.full_times[partner_idx].unsqueeze(0).float(),
                                   self.full_times[sequence_idx].unsqueeze(0).float(),
                                   [y[partner_idx].unsqueeze(0).long() for y in self.repeated_classes],
                                   [y[sequence_idx].unsqueeze(0).long() for y in self.repeated_classes],
                                   self.full_mask[partner_idx].unsqueeze(0))
        return torch.masked_select(gens, recon_mask.view(1, T, 1).bool()).view(-1, M)

    def categorical_generation(self, rev_mask, sequence_idx, partner_idx):
        """
        Simulate one sequence from a partner sequence at unmeasured time points.
        
        Parameters:
            rev_mask (Tensor): points to generate, `recon_mask[i, t]` = 1 indicates that the ith
                partner index at time `self.full_times[i, t]` should be generated
            sequence_idx (int): index in dataset to use as target sequence
            partner_idx (int): index in dataset to use as source sequence
            
        Returns:
            simulated expression from `partner_idx` base to `sequence_idx` experimental and temporal
                variables; shape K x M where M is the expression dimension and K is 
                `torch.count_nonzero(recon_mask)`
        """
        N, T, M = self.full_expr.shape
        
        gens = self.model.generate(self.full_expr[partner_idx].unsqueeze(0).float(),
                                   self.full_times[partner_idx].unsqueeze(0).float(),
                                   self.full_times[sequence_idx].unsqueeze(0).float(),
                                   [y[partner_idx].unsqueeze(0).long() for y in self.repeated_classes],
                                   [y[sequence_idx].unsqueeze(0).long() for y in self.repeated_classes],
                                   self.full_mask[partner_idx].unsqueeze(0))
        return torch.masked_select(gens, rev_mask.view(1, T, 1).bool()).view(-1, M)


