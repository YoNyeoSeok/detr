# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.pooling import AdaptiveAvgPool2d

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, accuracy_swig, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone_v, backbone_r, role_transformer, verb_transformer, num_classes, num_verb_embed, num_role_queries, gt_role_queries, verb_role_tgt_mask, use_verb_decoder, use_verb_fcn, num_mixture_proj, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_role_queries: number of role queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image.
            gt_role_queries: select gt role queris or not
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_verb_embed = num_verb_embed
        self.num_role_queries = num_role_queries
        self.gt_role_queries = gt_role_queries
        self.verb_role_tgt_mask = verb_role_tgt_mask
        self.role_transformer = role_transformer
        self.verb_transformer = verb_transformer
        hidden_dim = role_transformer.d_model
        self.nhead = role_transformer.nhead
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if num_verb_embed == 0:
            self.role_embed = nn.Embedding(num_role_queries, hidden_dim)
        else:
            self.verb_embed = nn.Embedding(num_verb_embed, hidden_dim // 2)
            self.role_embed = nn.Embedding(num_role_queries, hidden_dim // 2)
        self.input_proj_v = nn.Conv2d(backbone_v.num_channels, hidden_dim, kernel_size=1)
        self.input_proj_r = nn.Conv2d(backbone_r.num_channels, hidden_dim, kernel_size=1)
        if not use_verb_fcn:
            self.verb_query_for_verb_decoder = nn.Embedding(1, hidden_dim)
            self.verb_classifier = nn.Linear(hidden_dim, 504)
            if not use_verb_decoder:
                self.avg_pool = AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = AdaptiveAvgPool2d((7, 7))
            self.verb_classifier = nn.Sequential(
                nn.Flatten(-3),
                nn.Linear(hidden_dim * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 504),
            )
        self.mixture_proj = nn.ModuleList([nn.Linear(504, nm) for nm in num_mixture_proj])

        self.backbone_v = backbone_v
        self.backbone_r = backbone_r
        self.aux_loss = aux_loss
        self.use_verb_decoder = use_verb_decoder
        self.use_verb_fcn = use_verb_fcn

    def forward(self, samples: NestedTensor, targets):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
               - verbs: batched gt verbs of sample images [batch_size x 1]
               - roles: bathced roles according to gt verbs of sample iamges [batch_size x (max role:6)]

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_roles x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features_v, pos_v = self.backbone_v(samples)
        src_v, mask_v = features_v[-1].decompose()
        device = src_v.device
        assert mask_v is not None
        out = {}

        if self.use_verb_fcn:
            verb_hs = self.input_proj_v(src_v)
            verb_hs = self.avg_pool(verb_hs)[None, :, None]
        else:
            verb_hs, verb_memory = self.verb_transformer(
                self.input_proj_v(src_v), mask_v, self.verb_query_for_verb_decoder.weight, pos_v[-1], None, None)

            if not self.use_verb_decoder:
                verb_hs = self.avg_pool(verb_memory)[None, :, None, :, 0, 0]

        outputs_verb = self.verb_classifier(verb_hs)
        out.update({'pred_verb': outputs_verb[-1]})

        features_r, pos_r = self.backbone_r(samples)
        src_r, mask_r = features_r[-1].decompose()
        assert mask_r is not None

        if self.num_verb_embed == 0:
            query_embed = self.role_embed.weight
            # 190 x 512
        elif self.num_verb_embed == 1:
            verb_embed = self.verb_embed.weight.tile(len(self.role_embed.weight), 1)
            # 1 x 256  -> 190 x 256
            query_embed = torch.cat([self.role_embed.weight, verb_embed], axis=-1)
            # 190 x 256 cat 190 x 256 -> 190 x 512
        elif self.num_verb_embed == 504:
            # 504 x 256 -> batch_size x 256
            verb_embed = torch.stack([self.verb_embed.weight[t['verbs']] for t in targets])

            # 190 x batch_size x 256 cat 190 x batch_size x 256 -> 190 x batch_size x 512
            query_embed = torch.cat([
                # 190 x 256 -> 190 x batch_size x 256
                self.role_embed.weight[:, None].tile(1, len(targets), 1),
                # batch_size x 256 -> 190 x batch_size x 256
                verb_embed[None].tile(len(self.role_embed.weight), 1, 1)], axis=-1)

        if self.gt_role_queries:
            # self.verb_role_tgt_mask: 504 x 190 x 190
            # decoder_tgt_mask: batch_size x 190 x 190
            decoder_tgt_mask = torch.stack([self.verb_role_tgt_mask[t['verbs']] for t in targets]).to(device)
            # decoder_tgt_mask: batch_size*nhead x 190 x 190
            # TODO: 190 to num_roles
            decoder_tgt_mask = decoder_tgt_mask[:, None].tile(1, self.nhead, 1).contiguous().view(-1, 190, 190)

            # decoder_memory_mask: batch_size x 190 x 1
            decoder_memory_mask = torch.stack([
                # 190 x 1
                (~self.verb_role_tgt_mask[t['verbs']]).sum(dim=-1, keepdim=True) == 1
                for t in targets]).to(device)
            # decoder_memory_mask: batch_size x 190 x 1 -> batch_size*nhead x 190 x HW
            # TODO: 190 to num_roles, 49 to src_len
            decoder_memory_mask = decoder_memory_mask[:, None].tile(1, self.nhead, 1, 49).contiguous().view(-1, 190, 49)
        else:
            # self.verb_role_tgt_mask: 190 x 190 or None
            decoder_tgt_mask = self.verb_role_tgt_mask
            decoder_memory_mask = None

        mixture_weight = [F.softmax(mixture_proj(outputs_verb[-1]), dim=-1)
                          for mixture_proj in self.mixture_proj]
        hs = self.role_transformer(
            self.input_proj_r(src_r), mask_r, query_embed, pos_r[-1], decoder_tgt_mask, decoder_memory_mask, decoder_mixture_weight=mixture_weight)[0]
        outputs_class = self.class_embed(hs)
        out.update({'pred_logits': outputs_class[-1]})

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):

        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SWiGCriterion(nn.Module):
    """ This class computes the loss for DETR with SWiG dataset.
    """

    def __init__(self, num_classes, gt_role_queries, weight_dict):
        """ Create the criterion.
        """
        super().__init__()
        self.num_classes = num_classes
        self.gt_role_queries = gt_role_queries
        self.weight_dict = weight_dict
        self.loss_function = LabelSmoothing(0.2)
        self.loss_function_for_verb = LabelSmoothing(0.2)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        assert 'pred_logits' in outputs
        pred_logits = outputs['pred_logits']
        device = pred_logits.device

        batch_noun_loss = []
        batch_noun_acc = []
        for p, t in zip(pred_logits, targets):
            roles = t['roles']
            num_roles = len(roles)
            role_pred = p[roles]
            role_targ = t['labels'][:num_roles]
            role_targ = role_targ.long()
            batch_noun_acc += accuracy_swig(role_pred, role_targ)

            role_noun_loss = []
            for n in range(3):
                role_noun_loss.append(self.loss_function(role_pred, role_targ[:, n]))
            batch_noun_loss.append(sum(role_noun_loss))
        noun_loss = torch.stack(batch_noun_loss).mean()
        noun_acc = torch.stack(batch_noun_acc)

        verb_pred_logits = outputs['pred_verb'].squeeze(dim=1)
        gt_verbs = torch.stack([t['verbs'] for t in targets])
        verb_loss = self.loss_function_for_verb(verb_pred_logits, gt_verbs)
        verb_acc = accuracy(verb_pred_logits, gt_verbs, topk=(1, 5))

        batch_noun_acc_topk = []
        for verbs in verb_pred_logits.topk(5)[1].transpose(0, 1):
            batch_noun_acc = []
            for v, p, t in zip(verbs, pred_logits, targets):
                if v == t['verbs']:
                    roles = t['roles']
                    num_roles = len(roles)
                    role_targ = t['labels'][:num_roles]
                    role_targ = role_targ.long().cuda()
                    role_pred = p[roles]
                    batch_noun_acc += accuracy_swig(role_pred, role_targ)
                else:
                    batch_noun_acc += [torch.tensor(0., device=device)]
            batch_noun_acc_topk.append(torch.stack(batch_noun_acc))
        noun_acc_topk = torch.stack(batch_noun_acc_topk)

        stat = {'loss_vce': verb_loss, 'loss_nce': noun_loss,
                'noun_acc_top1': noun_acc_topk[0].mean(), 'noun_acc_all_top1': (noun_acc_topk[0] == 1).float().mean(),
                'noun_acc_top5': noun_acc_topk.sum(0).mean(), 'noun_acc_all_top1': (noun_acc_topk.sum(0) == 1).float().mean(),
                'verb_acc_top1': verb_acc[0], 'verb_acc_top5': verb_acc[1],
                'noun_acc_gt': noun_acc.mean(), 'noun_acc_all_gt': (noun_acc == 1).float().mean(),
                'class_error': torch.tensor(0.).to(device)}
        stat.update({'mean_acc': torch.stack([v for k, v in stat.items() if 'acc' in k]).mean()})

        return stat


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    elif args.dataset_file == "swig" or args.dataset_file == "imsitu":
        num_classes = args.num_classes
        if args.gt_role_queries:
            verb_role_tgt_mask = torch.tensor(~args.vr_adj_mat)
        else:
            if args.use_role_adj_attn_mask:
                verb_role_tgt_mask = torch.tensor(~args.vr_adj_mat.any(0))
            else:
                verb_role_tgt_mask = None

    device = torch.device(args.device)

    backbone_v = build_backbone(args)
    backbone_r = build_backbone(args)

    role_transformer = build_transformer(args)
    verb_transformer = build_transformer(args)

    model = DETR(
        backbone_v,
        backbone_r,
        role_transformer,
        verb_transformer,
        num_classes=num_classes,
        num_verb_embed=args.num_verb_embed,
        num_role_queries=args.num_role_queries,
        gt_role_queries=args.gt_role_queries,
        verb_role_tgt_mask=verb_role_tgt_mask,
        use_verb_decoder=args.use_verb_decoder,
        use_verb_fcn=args.use_verb_fcn,
        num_mixture_proj=args.num_mixture_proj,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    weight_dict = {'loss_nce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_vce'] = args.loss_ratio
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    if args.dataset_file != "swig" and args.dataset_file != "imsitu":
        matcher = build_matcher(args)
        criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses)
        criterion.to(device)
    else:
        criterion = SWiGCriterion(num_classes, gt_role_queries=args.gt_role_queries, weight_dict=weight_dict)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
