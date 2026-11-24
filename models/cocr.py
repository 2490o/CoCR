"""Chain-of-Causal Reasoning (CoCR) modules for 3D intent grounding.

Implements, on top of the IntentNet / BUTD-DETR baseline:
  * ICP  - Intent Causal Parsing: T ordered reasoning steps decoded from the intent text,
           projected onto a closed functional vocabulary F (K nodes)  ->  P(f|Q)   (Eq.1-4)
  * ICG  - Intent Causal Grounding over detected candidate boxes:
           P(o|f) from category-name embeddings + train-split category->function prior,
           g_m = sum_j P(o_m|f_j) P(f_j|Q), dependency pruning                     (Eq.5-8)
  * CVFA - Causal-Visual Feature Alignment: N iterations of cross-attention from candidate
           features to function-aware representations, graph re-estimation from visual
           features and dependency-weighted consistency loss                       (Eq.9-12, 23-25)
  * QFS  - query-level function score: the same function-mediated score computed for the
           decoder queries (the boxes that are actually ranked at evaluation), so that
           g can be fused into the final ranking.

Deviations from the paper text (needed for it to train; see notes in the code):
  * cosine similarities are divided by a learnable temperature; exp(cos) with cos in [-1,1]
    gives an almost uniform distribution over 40 nodes.
  * the pruning weight is P(o|f) normalised by its max over candidates, so that the
    threshold theta is scale-free (raw P(o|f) is ~1/M and would always be < 0.2).
  * P(f|Q) enters the cross-attention as a log-prior bias instead of scaling the values,
    which keeps the attention numerically well-conditioned (cosine terms are scale-free anyway).
  * all residual updates are gated with zero-initialised gates so training starts from the baseline.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_softmax(logits, valid, dim):
    """valid: bool tensor broadcastable to logits, True = keep."""
    logits = logits.masked_fill(~valid, -1e4)
    return logits.softmax(dim)


class _Temperature(nn.Module):
    def __init__(self, init_tau=0.1, max_scale=100.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_tau)))
        self.max_scale = max_scale

    def forward(self):
        return self.log_scale.exp().clamp(max=self.max_scale)


class _FuncCrossAttention(nn.Module):
    """Candidates (queries) attend to functional nodes (keys/values) with a log P(f|Q) bias."""

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.drop = nn.Dropout(dropout)
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, x, func, log_pf):
        # x: (B, M, D)  func: (K, D) or (B, K, D)  log_pf: (B, K)
        B, M, D = x.shape
        if func.dim() == 2:
            func = func.unsqueeze(0).expand(B, -1, -1)
        K = func.shape[1]
        q = self.q(self.norm_q(x)).view(B, M, self.h, self.dh).transpose(1, 2)
        k = self.k(func).view(B, K, self.h, self.dh).transpose(1, 2)
        v = self.v(func).view(B, K, self.h, self.dh).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dh) + log_pf[:, None, None, :]
        attn = self.drop(logits.softmax(-1))
        out = (attn @ v).transpose(1, 2).reshape(B, M, D)
        x = x + torch.tanh(self.gate_attn) * self.o(out)
        x = x + torch.tanh(self.gate_ffn) * self.ffn(x)
        return x


class CoCR(nn.Module):
    def __init__(self, d_model=288, vocab_dir='data/cocr', num_steps=4, num_iters=2,
                 prune_theta=0.2, use_icp=True, use_icg=True, use_cvfa=True, use_prune=True,
                 detach_aux=False):
        super().__init__()
        # detach_aux: inputs from the shared network are detached wherever they only feed CoCR's
        # auxiliary scores/losses (text for ICP, refined features for the graph re-estimation and
        # consistency, decoder queries for QFS). Those losses repeat the main loss's target-selection
        # signal and otherwise shift shared capacity away from box localisation.
        self.detach_aux = detach_aux
        fe = torch.from_numpy(np.load(os.path.join(vocab_dir, 'func_text_emb.npy'))).float()
        prior = torch.from_numpy(np.load(os.path.join(vocab_dir, 'cat_func_prior.npy'))).float()
        self.K = fe.shape[0]
        self.T = num_steps if use_icp else 1
        self.N = num_iters if use_cvfa else 0
        self.theta = prune_theta
        self.use_icp, self.use_icg, self.use_cvfa, self.use_prune = use_icp, use_icg, use_cvfa, use_prune

        # functional requirement prototypes Embed(f_j)
        self.register_buffer('func_text_emb', fe)
        self.func_proj = nn.Sequential(nn.Linear(fe.shape[1], d_model), nn.LayerNorm(d_model))
        self.func_delta = nn.Parameter(torch.zeros(self.K, d_model))

        # ---- ICP ----
        if use_icp:
            self.step_queries = nn.Parameter(torch.randn(self.T, d_model) * 0.02)
            layer = nn.TransformerDecoderLayer(d_model, 8, 2 * d_model, 0.1, batch_first=True, norm_first=True)
            self.step_decoder = nn.TransformerDecoder(layer, 2)
            self.step_norm = nn.LayerNorm(d_model)
            self.register_buffer('step_causal_mask',
                                 torch.triu(torch.ones(self.T, self.T, dtype=torch.bool), diagonal=1))
        else:
            self.pool_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.step_key = nn.Linear(d_model, d_model)
        self.func_key = nn.Linear(d_model, d_model)
        self.tau_icp = _Temperature(0.1)

        # ---- ICG (category -> function compatibility) ----
        if use_icg:
            self.register_buffer('cat_func_logprior', torch.log(prior.clamp(min=1e-3)))
            self.cat_proj = nn.Sequential(nn.Linear(768, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            self.func_key_cat = nn.Linear(d_model, d_model)
            self.prior_weight = nn.Parameter(torch.tensor(1.0))
            self.tau_icg = _Temperature(0.1)

        # ---- CVFA ----
        if use_cvfa:
            self.cvfa_layers = nn.ModuleList([_FuncCrossAttention(d_model) for _ in range(self.N)])
            self.vis_key = nn.Linear(d_model, d_model)
            self.func_key_vis = nn.Linear(d_model, d_model)
            self.tau_vis = _Temperature(0.1)

        # ---- query-level function score (used for ranking fusion) ----
        self.qry_key = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.func_key_qry = nn.Linear(d_model, d_model)
        self.tau_qry = _Temperature(0.1)

    # ------------------------------------------------------------------
    def func_embed(self):
        return self.func_proj(self.func_text_emb) + self.func_delta  # (K, D)

    @staticmethod
    def _cos(a, b):
        return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).transpose(-1, -2)

    def parse(self, text_feats, text_padding_mask):
        """ICP. text_feats (B, L, D), text_padding_mask (B, L) True = pad. Returns P(f|Q) (B, K), aux."""
        func = self.func_embed()
        B = text_feats.shape[0]
        if self.detach_aux:
            text_feats = text_feats.detach()
        if self.use_icp:
            q = self.step_queries.unsqueeze(0).expand(B, -1, -1)
            h = self.step_decoder(q, text_feats, tgt_mask=self.step_causal_mask,
                                  memory_key_padding_mask=text_padding_mask)
            h = self.step_norm(h)  # (B, T, D)
        else:
            valid = (~text_padding_mask).float().unsqueeze(-1)
            h = (text_feats * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp(min=1)
            h = self.pool_proj(h)  # (B, 1, D)
        step_logits = self.tau_icp() * self._cos(self.step_key(h), self.func_key(func).unsqueeze(0))  # (B,T,K)
        p_step = step_logits.softmax(-1)
        p_func = p_step.mean(1)  # Eq.4
        return p_func, {'cocr_step_func': p_step}

    def _prune(self, p_of, valid):
        """p_of: (B, M, K) normalised over M. Returns pruned & renormalised P, pruning loss."""
        w = p_of / p_of.max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        weak = (w < self.theta) & valid
        loss = ((w ** 2) * weak.float()).sum() / valid.float().sum().clamp(min=1) / max(self.K, 1)
        keep = (~weak.detach()) & valid
        p = p_of * keep.float()
        p = p / p.sum(1, keepdim=True).clamp(min=1e-8)
        return p, loss

    def ground(self, detected_feats, det_cls_emb, det_class_ids, det_valid, p_func):
        """ICG + CVFA on detected candidates.

        detected_feats (B, M, D); det_cls_emb (B, M, 768); det_class_ids (B, M);
        det_valid (B, M) True = valid box (NOT the padding mask); p_func (B, K).
        Returns updated detected_feats and aux dict (log g over candidates, losses).
        """
        aux = {}
        func = self.func_embed()
        valid = det_valid.bool().unsqueeze(-1)  # (B, M, 1)
        log_pf = torch.log(p_func.clamp(min=1e-8))
        zero = detected_feats.sum() * 0.0
        prune_loss = zero

        # ICG: initial object-function dependencies from predicted category names
        if self.use_icg:
            cat = self.cat_proj(det_cls_emb.detach())
            s0 = self.tau_icg() * self._cos(cat, self.func_key_cat(func).unsqueeze(0))  # (B, M, K)
            s0 = s0 + self.prior_weight * self.cat_func_logprior[det_class_ids.long()]
            p0 = _masked_softmax(s0, valid, dim=1)  # P(o_m | f_j), normalised over candidates
            if self.use_prune:
                p0, l = self._prune(p0, valid.expand_as(p0))
                prune_loss = prune_loss + l
            g0 = (p0 * p_func.unsqueeze(1)).sum(-1)  # Eq.5/7  (B, M)
            aux['cocr_det_logp0'] = torch.log(g0.clamp(min=1e-8))

        # CVFA: iterative visual-functional refinement
        cons_loss = zero
        if self.use_cvfa:
            v = detected_feats
            for n in range(self.N):
                v = self.cvfa_layers[n](v, func, log_pf)  # Eq.23
                vk = self.vis_key(v.detach() if self.detach_aux else v)
                sv = self.tau_vis() * self._cos(vk, self.func_key_vis(func).unsqueeze(0))
                pn = _masked_softmax(sv, valid, dim=1)  # Eq.24
                if self.use_prune:
                    pn, l = self._prune(pn, valid.expand_as(pn))
                    prune_loss = prune_loss + l
            gn = (pn * p_func.unsqueeze(1)).sum(-1)  # Eq.25
            aux['cocr_det_logpN'] = torch.log(gn.clamp(min=1e-8))
            # dependency-weighted consistency (Eq.11-12), weights detached
            a = (pn * p_func.unsqueeze(1)) * valid.float()
            a = (a / a.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)).detach()
            cos = self._cos(vk, self.func_key_vis(func).unsqueeze(0))
            cons_loss = (a * (1 - cos)).sum(dim=(1, 2)).mean()
            detected_feats = v

        aux['cocr_prune_term'] = prune_loss
        aux['cocr_cons_term'] = cons_loss
        return detected_feats, aux

    def query_logp(self, query, p_func):
        """query (B, Q, D) -> log g_q over the Q decoder queries (B, Q)."""
        func = self.func_embed()
        if self.detach_aux:
            query = query.detach()
        s = self.tau_qry() * self._cos(self.qry_key(query), self.func_key_qry(func).unsqueeze(0))  # (B,Q,K)
        p = s.softmax(1)
        g = (p * p_func.unsqueeze(1)).sum(-1)
        return torch.log(g.clamp(min=1e-8))
