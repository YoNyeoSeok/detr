# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn

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

    def __init__(self, backbone, transformer, num_classes, num_verb_embed, num_role_queries, select_verb_role_queries, vidx_ridx, role_adj_mat):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_role_queries: number of role queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image.
            select_verb_role_queries: select gt role queris or not
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_verb_embed = num_verb_embed
        self.num_role_queries = num_role_queries
        self.select_verb_role_queries = select_verb_role_queries
        self.vidx_ridx = vidx_ridx
        self.role_adj_mat = role_adj_mat
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        if num_verb_embed == 0:
            self.role_embed = nn.Embedding(num_role_queries, hidden_dim)
        else:
            self.verb_embed = nn.Embedding(num_verb_embed, hidden_dim // 2)
            self.role_embed = nn.Embedding(num_role_queries, hidden_dim // 2)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.verb_classifier = nn.Linear(hidden_dim, 504)

    def select_verb_role_query_embed(self, batch_verb):
        if not self.select_verb_role_queries and self.num_verb_embed == 0:
            query_embed = self.role_embed.weight.unsqueeze(1).repeat(1, len(batch_verb), 1)
            attn_mask = self.role_adj_mat
        else:
            query_embed = []
            attn_mask = []
            for verb in batch_verb:
                roles = self.vidx_ridx[verb]
                if self.select_verb_role_queries:
                    selected_role_query_embed = self.role_embed.weight[roles]
                    attn_mask += [torch.ones((len(roles), len(roles)))]
                else:
                    selected_role_query_embed = self.role_embed.weight
                    attn_mask += [self.role_adj_mat]
                if self.num_verb_embed == 1:
                    selected_verb_query_embed = self.verb_embed.weight[0]
                elif self.num_verb_embed == 504:
                    selected_verb_query_embed = self.verb_embed.weight[verb]
                selected_verb_query_embed = selected_verb_query_embed.tile(selected_role_query_embed.shape[0], 1)

                selected_query_embed = torch.cat([
                    selected_role_query_embed, selected_verb_query_embed], axis=1)
                query_embed.append(selected_query_embed)

        return query_embed, attn_mask

    def forward_verb(self, src, mask, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)

        memory = self.transformer.forward_encoder(src, mask, pos_embed)
        mem = memory.permute(1, 2, 0).view(bs, c, h, w)

        # verb predict
        outputs_verb = self.avg_pool(mem).squeeze(dim=2).squeeze(dim=2)
        outputs_verb = self.verb_classifier(outputs_verb)

        return outputs_verb, memory

    def forward_role_noun(self, verbs, memory, mask, pos_embed):
        query_embed, attn_mask = self.select_verb_role_query_embed(verbs)

        if not self.select_verb_role_queries and self.num_verb_embed == 0:
            tgt = torch.zeros_like(query_embed)
            hs = self.transformer.forward_decoder(tgt, memory, attn_mask.to(tgt.device), mask, query_embed, pos_embed)
        else:
            hs = []
            for i, (sliced_mask, sliced_query_embed) in enumerate(zip(mask, query_embed)):
                tgt = torch.zeros_like(sliced_query_embed)
                # sliced_hs: num_layers, 190 or len(slized_query_embed), 1, hidden_dim
                sliced_hs = self.transformer.forward_decoder(
                    tgt[:, None], memory[:, i:i + 1], attn_mask[i].to(tgt.device), sliced_mask[None], sliced_query_embed[:, None], pos_embed[:, i:i + 1])
                # padded_hs: num_layers, 190 or 6, 1, hidden_dim
                if not self.select_verb_role_queries:
                    padded_hs = sliced_hs
                else:
                    padded_hs = F.pad(sliced_hs, (0, 0, 0, 0, 0, 6 - len(sliced_query_embed)), mode='constant', value=0)
                hs.append(padded_hs)
            hs = torch.cat(hs, axis=2)

        outputs_class = self.class_embed(hs.transpose(1, 2))

        return outputs_class

    def forward(self, samples: NestedTensor, verbs: Union[int, torch.Tensor]):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            and a verbs, which is one of:
               - verbs (int): Use top k verb pred
               - verbs (Tensor): a tensor of shape [batch_size x K], predict for K verbs

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
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        pos_embed = pos[-1]
        assert mask is not None

        # preprocess
        mask = mask.flatten(1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # encoder + verb classification
        src = self.input_proj(src)
        outputs_verb, memory = self.forward_verb(src, mask, pos_embed)
        # decoder + role noun classification
        if isinstance(verbs, int):
            verbs = outputs_verb.topk(verbs)[1]

        outputs_class_per_verb = []
        for batch_verb in verbs.transpose(0, 1):
            outputs_class = self.forward_role_noun(batch_verb, memory, mask, pos_embed)
            outputs_class_per_verb.append((batch_verb, outputs_class[-1]))

        out = {'pred_logits': outputs_class_per_verb, 'pred_verb': outputs_verb}
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

    def __init__(self, num_classes, select_verb_role_queries, weight_dict):
        """ Create the criterion.
        """
        super().__init__()
        self.num_classes = num_classes
        self.select_verb_role_queries = select_verb_role_queries
        self.weight_dict = weight_dict
        self.loss_function = LabelSmoothing(0.2)
        self.loss_function_for_verb = LabelSmoothing(0.2)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             retuction: 
        """
        assert 'pred_logits' in outputs
        assert 'pred_verb' in outputs
        pred_logits_per_verb = outputs['pred_logits']
        pred_verb = outputs['pred_verb']
        device = pred_verb.device

        gt_verbs = torch.stack([t['verbs'] for t in targets])
        verb_loss = self.loss_function_for_verb(pred_verb, gt_verbs)
        verb_acc = accuracy(pred_verb, gt_verbs, topk=(1, 5))

        batch_noun_loss = []
        batch_noun_acc_per_verb = []
        for verbs, pred_logits in pred_logits_per_verb:
            batch_noun_acc = []
            for v, p, t in zip(verbs, pred_logits, targets):
                if v == t['verbs']:
                    roles = t['roles']
                    num_roles = len(roles)
                    role_targ = t['labels'][:num_roles]
                    role_targ = role_targ.long().cuda()
                    role_pred = p[:num_roles] if self.select_verb_role_queries else p[roles]
                    batch_noun_acc += accuracy_swig(role_pred, role_targ)

                    role_noun_loss = []
                    for n in range(3):
                        role_noun_loss.append(self.loss_function(role_pred, role_targ[:, n]))
                    batch_noun_loss.append(sum(role_noun_loss))
                else:
                    batch_noun_acc += [torch.tensor(0., device=device)]
            batch_noun_acc_per_verb.append(torch.stack(batch_noun_acc))
        if batch_noun_loss:
            noun_loss = torch.stack(batch_noun_loss).mean()
        else:
            noun_loss = torch.tensor(0., requires_grad=True)
        noun_acc = torch.stack(batch_noun_acc_per_verb)

        return {'loss_vce': verb_loss, 'loss_nce': noun_loss,
                'verb_acc_top1': verb_acc[0], 'verb_acc_top5': verb_acc[1],
                'noun_acc_top1': noun_acc[0].mean(), 'noun_acc_all_top1': noun_acc[0].bool().float().mean(),
                'noun_acc_top5': noun_acc.sum(0).mean(), 'noun_acc_all_top1': noun_acc.sum(0).bool().float().mean(),
                'class_error': torch.tensor(0.).to(device)}


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
        vidx_ridx = args.vidx_ridx
        role_adj_mat = torch.tensor(args.role_adj_mat) if args.use_role_adj_attn_mask else None
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_verb_embed=args.num_verb_embed,
        num_role_queries=args.num_role_queries,
        select_verb_role_queries=args.select_verb_role_queries,
        vidx_ridx=vidx_ridx,
        role_adj_mat=role_adj_mat,
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
        criterion = SWiGCriterion(
            num_classes, select_verb_role_queries=args.select_verb_role_queries, weight_dict=weight_dict)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
