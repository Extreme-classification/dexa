import torch
import torch.nn.functional as F
import math


def construct_loss(params, pos_weight=1.0):
    """
    Return the loss
    Arguments:
    ----------
    params: NameSpace
        parameters of the model
        * mean: mean over all entries/terms (used with OVA setting)
        * sum: sum over all entries/terms (used with a shortlist)
               - the loss is then divided by batch_size resulting in
                 sum over labels and mean over data-points in a batch
    pos_weight: int or None, optional, default=None
        weight the loss terms where y_nl = 1
    """
    _reduction = 'mean'
    # pad index is for OVA training and not shortlist
    # pass mask for shortlist
    _pad_ind = None #if params.use_shortlist else params.label_padding_index
    if params.loss == 'bce':
        return BCEWithLogitsLoss(
            reduction=_reduction,
            pad_ind=_pad_ind,
            pos_weight=None)
    elif params.loss == 'triplet_margin_ohnm':
        if params.loss_num_positives > 1:
            return TripletMarginLossOHNMMulti(
                reduction=_reduction,
                num_negatives=params.loss_num_negatives,
                num_positives=params.loss_num_positives,
                margin=params.margin                
            )
        else:
            return TripletMarginLossOHNM(
                reduction=_reduction,
                apply_softmax=params.loss_agressive,
                tau=0.1,
                k=params.loss_num_negatives,
                margin=params.margin)
    elif params.loss == 'hinge_contrastive':
        return HingeContrastiveLoss(
            reduction=_reduction,
            pos_weight=pos_weight,
            margin=params.margin)
    elif params.loss == 'prob_contrastive':
        return ProbContrastiveLoss(
            reduction=_reduction,
            c=0.75,
            d=3.0,
            pos_weight=pos_weight,
            threshold=params.margin)
    elif params.loss == 'kprob_contrastive':
        return kProbContrastiveLoss(
            k=params.k,
            reduction='custom',
            c=0.9,
            d=1.5,
            apply_softmax=False,
            pos_weight=pos_weight)


class _Loss(torch.nn.Module):
    def __init__(self, reduction='mean', pad_ind=None):
        super(_Loss, self).__init__()
        self.reduction = reduction
        self.pad_ind = pad_ind

    def _reduce(self, loss):
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'custom':
            return loss.sum(dim=1).mean()
        else:
            return loss.sum()

    def _mask_at_pad(self, loss):
        """
        Mask the loss at padding index, i.e., make it zero
        """
        if self.pad_ind is not None:
            loss[:, self.pad_ind] = 0.0
        return loss

    def _mask(self, loss, mask=None):
        """
        Mask the loss at padding index, i.e., make it zero
        * Mask should be a boolean array with 1 where loss needs
        to be considered.
        * it'll make it zero where value is 0
        """
        if mask is not None:
            loss = loss.masked_fill(~mask, 0.0)
        return loss


def _convert_labels_for_svm(y):
    """
        Convert labels from {0, 1} to {-1, 1}
    """
    return 2.*y - 1.0


class HingeLoss(_Loss):
    r""" Hinge loss
    * it'll automatically convert target to +1/-1 as required by hinge loss

    Arguments:
    ----------
    margin: float, optional (default=1.0)
        the margin in hinge loss
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    pad_ind: int/int64 or None (default=None)
        ignore loss values at this index
        useful when some index has to be used as padding index
    """

    def __init__(self, margin=1.0, reduction='mean', pad_ind=None):
        super(HingeLoss, self).__init__(reduction, pad_ind)
        self.margin = margin

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            typically logits from the neural network
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
            * it'll automatically convert to +1/-1 as required by hinge loss
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is zero

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        loss = F.relu(self.margin - _convert_labels_for_svm(target)*input)
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)


class SquaredHingeLoss(_Loss):
    r""" Squared Hinge loss
    * it'll automatically convert target to +1/-1 as required by hinge loss

    Arguments:
    ----------
    margin: float, optional (default=1.0)
        the margin in squared hinge loss
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    pad_ind: int/int64 or None (default=None)
        ignore loss values at this index
        useful when some index has to be used as padding index
    """

    def __init__(self, margin=1.0, size_average=None, reduction='mean'):
        super(SquaredHingeLoss, self).__init__(size_average, reduction)
        self.margin = margin

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            typically logits from the neural network
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
            * it'll automatically convert to +1/-1 as required by hinge loss
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is zero

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        loss = F.relu(self.margin - _convert_labels_for_svm(target)*input)
        loss = loss**2
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)


class BCEWithLogitsLoss(_Loss):
    r""" BCE loss (expects logits; numercial stable)
    This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

    Arguments:
    ----------
    weight: torch.Tensor or None, optional (default=None))
        a manual rescaling weight given to the loss of each batch element.
        If given, has to be a Tensor of size batch_size
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    pos_weight: torch.Tensor or None, optional (default=None)
        a weight of positive examples.
        it must be a vector with length equal to the number of classes.
    pad_ind: int/int64 or None (default=None)
        ignore loss values at this index
        useful when some index has to be used as padding index
    """
    __constants__ = ['weight', 'pos_weight', 'reduction']

    def __init__(self, weight=None, reduction='mean',
                 pos_weight=None, pad_ind=None):
        super(BCEWithLogitsLoss, self).__init__(reduction, pad_ind)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            typically logits from the neural network
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is zero

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        loss = F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction='none')
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)


class HingeContrastiveLoss(_Loss):
    r""" Hinge contrastive loss (expects cosine similarity)

    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    margin: float, optional (default=0.8)
        margin in Hinge contrastive loss
    pos_weight: float, optional (default=1.0)
        weight of loss with positive target
    """

    def __init__(self, reduction='mean', margin=0.8, pos_weight=1.0):
        super(HingeContrastiveLoss, self).__init__(reduction=reduction)
        self.margin = margin
        self.pos_weight = pos_weight

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is zero

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        loss = torch.where(target > 0, (1-input) * self.pos_weight,
                           torch.max(
                               torch.zeros_like(input), input - self.margin))
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)


class TripletMarginLossOHNM(_Loss):
    r""" Triplet Margin Loss with Online Hard Negative Mining

    * Applies loss using the hardest negative in the mini-batch
    * Assumes diagonal entries are ground truth (for multi-class as of now)

    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    margin: float, optional (default=0.8)
        margin in triplet margin loss
    k: int, optional (default=2)
        compute loss only for top-k negatives in each row 
    apply_softmax: boolean, optional (default=2)
        promotes hard negatives using softmax
    """

    def __init__(self, reduction='mean', margin=0.8, k=3, apply_softmax=False, tau=0.1, num_violators=False):
        super(TripletMarginLossOHNM, self).__init__(reduction=reduction)
        self.margin = margin
        self.k = k
        self.tau = tau
        self.num_violators = num_violators
        self.apply_softmax = apply_softmax

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is False

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        sim_p = torch.diagonal(input).view(-1, 1)
        similarities = torch.where(target == 0, input, torch.full_like(input, -10))
        _, indices = torch.topk(similarities, largest=True, dim=1, k=self.k)
        sim_n = input.gather(1, indices)
        loss = torch.max(torch.zeros_like(sim_p), sim_n - sim_p + self.margin)
        if self.apply_softmax:
            sim_n[loss == 0] = -50
            prob = torch.softmax(sim_n/self.tau, dim=1)
            loss = loss * prob
        reduced_loss = self._reduce(loss)
        if self.num_violators:
            nnz = torch.sum((loss > 0), axis=1).float().mean()
            return reduced_loss, nnz
        else:
            return reduced_loss


class ProbContrastiveLoss(_Loss):
    r""" A probabilistic contrastive loss 
    * expects cosine similarity
    * or <w, x> b/w normalized vectors

    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    pos_weight: float, optional (default=1.0)
        weight of loss with positive target
    c: float, optional (default=0.75)
        c in PC loss (0, 1)
    d: float, optional (default=3.0)
        d in PC loss >= 1
    threshold: float, optional (default=0.05)
        clip loss values less than threshold for negatives 
    """

    def __init__(self, reduction='mean', pos_weight=1.0,
                 c=0.75, d=3.0, threshold=0.05):
        super(ProbContrastiveLoss, self).__init__(reduction=reduction)
        self.pos_weight = pos_weight
        self.d = d
        self.c = math.log(c)
        self.scale = 1/d
        self.threshold = threshold
        self.constant = c/math.exp(d)

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is False

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        # top_k negative entries
        loss = torch.where(
            target > 0, -self.c + (1-input)*self.d*self.pos_weight,
            -torch.log(1 - torch.exp(self.d*input)*self.constant))
        loss = loss * self.scale
        loss[torch.logical_and(target == 0, loss < self.threshold)] = 0.0
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)


class kProbContrastiveLoss(_Loss):
    r""" A probabilistic contrastive loss 
    *expects cosine similarity
    * or <w, x> b/w normalized vectors

    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    pos_weight: float, optional (default=1.0)
        weight of loss with positive target
    c: float, optional (default=0.75)
        c in PC loss (0, 1)
    d: float, optional (default=3.0)
        d in PC loss >= 1
    k: int, optional (default=2)
        compute loss only for top-k negatives in each row 
    apply_softmax: boolean, optional (default=2)
        promotes hard negatives using softmax
    """

    def __init__(self, reduction='mean', pos_weight=1.0, c=0.9,
                 d=1.5, k=2, apply_softmax=False):
        super(kProbContrastiveLoss, self).__init__(reduction=reduction)
        self.pos_weight = pos_weight
        self.d = d
        self.k = k
        self.c = math.log(c)
        self.scale = 1/d
        self.constant = c/math.exp(d)
        self.apply_softmax = apply_softmax

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is False

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        loss = torch.where(
            target > 0, -self.c + (1-input)*self.d*self.pos_weight,
            -torch.log(1 - torch.exp(self.d*input)*self.constant))
        neg_vals, neg_ind = torch.topk(loss-target*3, k=self.k)
        loss_neg = torch.zeros_like(target)
        if self.apply_softmax:
            neg_probs = torch.softmax(neg_vals, dim=1)
            loss_neg = loss_neg.scatter(1, neg_ind, neg_probs*neg_vals)
        else:
          loss_neg = loss_neg.scatter(1, neg_ind, neg_vals)
        loss = torch.where(
            target > 0, loss, loss_neg)        
        loss = self._mask(loss, mask)
        return self._reduce(loss)
    

class TripletMarginLossOHNMMulti(_Loss):
    r""" Triplet Margin Loss with Online Hard Negative Mining
    * Applies loss using the hardest negative in the mini-batch
    * Assumes diagonal entries are ground truth
    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    """

    def __init__(self, margin=1.0, eps=1.0e-6, reduction='mean',
                 num_positives=3, num_negatives=10,
                 num_violators=False, alpha=0.9):
        super(TripletMarginLossOHNMMulti, self).__init__(reduction)
        self.mx_lim = 100
        self.mn_lim = -100
        self.alpha = alpha
        self._eps = eps
        self._margin = margin
        self._reduction = reduction
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.num_violators = num_violators

    def forward(self, output, target, *args):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        B = target.size(0)
        if target.size(0) != target.size(1):
            MX_LIM = torch.full_like(output, self.mx_lim)
            sim_p = output.where(target == 1, MX_LIM)
            indices = sim_p.topk(largest=False, dim=1, k=self.num_positives)[1]
            sim_p = sim_p.gather(1, indices)
        else:
            sim_p = output.diagonal().view(B, 1)
        
        MN_LIM = torch.full_like(output, self.mn_lim)
        target = target.to(output.device)

        _, num_p = sim_p.size()
        sim_p = sim_p.view(B, num_p, 1)
        sim_m = MN_LIM.where(target == 1, output)
        indices = sim_m.topk(largest=True, dim=1, k=self.num_negatives)[1]
        sim_n = output.gather(1, indices)
        sim_n = sim_n.unsqueeze(1).repeat_interleave(num_p, dim=1)
        loss = F.relu(sim_n - sim_p + self._margin)
        prob = loss.clone()
        prob.masked_fill_(prob == 0, self.mn_lim)
        loss = F.softmax(prob, dim=-1)*loss
        if (self._reduction == "mean"):
            reduced_loss = loss.mean()
        else:
            reduced_loss = loss.sum()
        if self.num_violators:
            nnz = torch.sum((loss > 0), axis=1).float().mean()
            return reduced_loss, nnz
        else:
            return reduced_loss
