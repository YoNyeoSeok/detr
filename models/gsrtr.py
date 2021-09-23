# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized,
                       accuracy_swig, accuracy_swig_bbox)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class GSRTR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.noun_classifier1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                              nn.ReLU(),
                                              nn.Dropout(0.3),
                                              nn.Linear(hidden_dim*2, 9927+1))
        self.bbox_predictor1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(hidden_dim*2, hidden_dim*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(hidden_dim*2, 4))
        # classifiers & predictors (for grounded noun prediction)
        self.noun_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(hidden_dim*2, 9927+1))
        self.bbox_predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                            nn.ReLU(),
                                            nn.Dropout(0.2),
                                            nn.Linear(hidden_dim*2, hidden_dim*2),
                                            nn.ReLU(),
                                            nn.Dropout(0.2),
                                            nn.Linear(hidden_dim*2, 4))
        self.bbox_conf_predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                                 nn.ReLU(), 
                                                 nn.Dropout(0.2),
                                                 nn.Linear(hidden_dim*2, 1))

    def forward(self, samples: NestedTensor, gt_verb=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
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
        assert mask is not None
        # hs, hs2, _, _ = self.transformer(self.input_proj(src), mask, self.query_embed.weight, torch.zeros(6, 256).to("cuda:0"), pos[-1], torch.zeros(1, 256).to("cuda:0"))
        hs, verb_pred, num_roles, hs2, _, _ = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1], gt_verb)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        noun_pred1 = self.noun_classifier1(hs)
        bbox_pred1 = self.bbox_predictor1(hs).sigmoid()
        noun_pred = self.noun_classifier(hs2)
        # noun_pred = F.pad(noun_pred, (0, 0, 0, MAX_NUM_ROLES - num_roles), mode='constant', value=0)[-1].view(1, MAX_NUM_ROLES, self.num_noun_classes)
        bbox_pred = self.bbox_predictor(hs2).sigmoid()
        # bbox_pred = F.pad(bbox_pred, (0, 0, 0, MAX_NUM_ROLES - num_roles), mode='constant', value=0)[-1].view(1, MAX_NUM_ROLES, 4)
        bbox_conf_pred = self.bbox_conf_predictor(hs2)
        # bbox_conf_pred = F.pad(bbox_conf_pred, (0, 0, 0, MAX_NUM_ROLES - num_roles), mode='constant', value=0)[-1].view(1, MAX_NUM_ROLES, 1)

        out = {'pred_logits': outputs_class[-1].sum()*0 + noun_pred1[-1],
               'pred_boxes': outputs_coord[-1].sum()*0 + bbox_pred1[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class.sum()*0 + noun_pred1, outputs_coord.sum()*0 + bbox_pred1)
        out['pred_verb'] = verb_pred
        out["num_roles"] = num_roles
        out['pred_noun'] = noun_pred[-1]
        out['pred_bbox'] = bbox_pred[-1]
        out['pred_bbox_conf'] = bbox_conf_pred[-1]
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
        if target_classes_o.dim() == 1:
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
        elif target_classes_o.dim() == 2:
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device).unsqueeze(2).repeat(1, 1, 3)
        else:
            assert False
        target_classes[idx] = target_classes_o.long()

        if target_classes_o.dim() == 1:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        elif target_classes_o.dim() == 2:
            loss_ce = torch.stack([
                F.cross_entropy(src_logits.transpose(1, 2), t.squeeze(-1), self.empty_weight)
                for t in target_classes.split(1, dim=-1)]).mean()
        else:
            assert False
        losses = {'loss_ce': loss_ce}

        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     if target_classes_o.dim() == 1:
        #         losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        #     elif target_classes_o.dim() == 2:
        #         losses['class_error'] = 100 - accuracy_swig(src_logits[idx], target_classes_o)[1]
        #     else:
        #         assert False
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
        # SWiG bounding box padding with -1
        exist_boxes = target_boxes[:, 0] != -1
        assert exist_boxes.sum() != 0
        
        src_boxes = src_boxes[exist_boxes]
        target_boxes = target_boxes[exist_boxes]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox1'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou1'] = loss_giou.sum() / num_boxes
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
        # SWiG bounding padding with -1
        num_boxes = sum([sum(v["boxes"][:, 0]!=-1).cpu().item() for v in targets])
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
    """ 
    This class computes the loss for GSRTR with SWiG dataset.
    """
    def __init__(self, weight_dict, SWiG_json_train=None, SWiG_json_eval=None, idx_to_role=None):
        """ 
        Create the criterion.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.loss_function = LabelSmoothing(0.2)
        self.loss_function_verb = LabelSmoothing(0.3)
        self.SWiG_json_train = SWiG_json_train
        self.SWiG_json_eval = SWiG_json_eval
        self.idx_to_role = idx_to_role


    def forward(self, outputs, targets, eval=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             eval: boolean, used in evlauation
        """
        NUM_ANNOTATORS = 3

        # gt verb (value & value-all) acc and calculate noun loss
        assert 'pred_noun' in outputs
        pred_noun = outputs['pred_noun']
        device = pred_noun.device
        batch_size = pred_noun.shape[0]
        batch_noun_loss, batch_noun_acc, batch_noun_correct = [], [], []
        for i in range(batch_size):
            p, t = pred_noun[i], targets[i]
            roles = t['roles']
            num_roles = len(roles)
            role_pred = p[:num_roles]
            role_targ = t['labels'][:num_roles]
            role_targ = role_targ.long()
            acc_res = accuracy_swig(role_pred, role_targ)
            batch_noun_acc += acc_res[1]
            batch_noun_correct += acc_res[0]
            role_noun_loss = []
            for n in range(NUM_ANNOTATORS):
                role_noun_loss.append(self.loss_function(role_pred, role_targ[:, n]))
            batch_noun_loss.append(sum(role_noun_loss))
        noun_loss = torch.stack(batch_noun_loss).mean()
        noun_acc = torch.stack(batch_noun_acc)
        noun_correct = torch.stack(batch_noun_correct)

        # top-1 & top 5 verb acc and calculate verb loss 
        assert 'pred_verb' in outputs
        verb_pred_logits = outputs['pred_verb'].squeeze(1)
        gt_verbs = torch.stack([t['verbs'] for t in targets])
        verb_acc_topk = accuracy(verb_pred_logits, gt_verbs, topk=(1, 5))
        verb_loss = self.loss_function_verb(verb_pred_logits, gt_verbs)
        
        # top-1 & top 5 (value & value-all) acc
        batch_noun_acc_topk, batch_noun_correct_topk = [], []
        for verbs in verb_pred_logits.topk(5)[1].transpose(0, 1):
            batch_noun_acc = []
            batch_noun_correct = []
            for i in range(batch_size):
                v, p, t = verbs[i], pred_noun[i], targets[i]
                if v == t['verbs']:
                    roles = t['roles']
                    num_roles = len(roles)
                    role_pred = p[:num_roles]
                    role_targ = t['labels'][:num_roles]
                    role_targ = role_targ.long()
                    acc_res = accuracy_swig(role_pred, role_targ)
                    batch_noun_acc += acc_res[1]
                    batch_noun_correct += acc_res[0]
                else:
                    batch_noun_acc += [torch.tensor(0., device=device)]
                    batch_noun_correct += [torch.tensor([0, 0, 0, 0, 0, 0], device=device)]
            batch_noun_acc_topk.append(torch.stack(batch_noun_acc))
            batch_noun_correct_topk.append(torch.stack(batch_noun_correct))
        noun_acc_topk = torch.stack(batch_noun_acc_topk)
        noun_correct_topk = torch.stack(batch_noun_correct_topk) # topk x batch x max roles 

        # bbox prediction
        assert 'pred_bbox' in outputs
        assert 'pred_bbox_conf' in outputs
        pred_bbox = outputs['pred_bbox']
        pred_bbox_conf = outputs['pred_bbox_conf'].squeeze(2)
        batch_bbox_acc, batch_bbox_acc_top1, batch_bbox_acc_top5 = [], [], []
        batch_bbox_loss, batch_giou_loss, batch_bbox_conf_loss = [], [], []
        for i in range(batch_size):
            num_roles = len(t['roles'])
            pb, pbc, t = pred_bbox[i][:num_roles], pred_bbox_conf[i][:num_roles], targets[i]
            mw, mh, target_bboxes = t['max_width'], t['max_height'], t['boxes'][:num_roles]
            cloned_pb, cloned_target_bboxes = pb.clone(), target_bboxes.clone()
            bbox_exist = target_bboxes[:, 0] != -1
            num_bbox = bbox_exist.sum().item()

            # bbox conf loss
            loss_bbox_conf = F.binary_cross_entropy_with_logits(pbc, 
                                                                bbox_exist.float(), reduction='mean')
            batch_bbox_conf_loss.append(loss_bbox_conf)

            # bbox reg loss and giou loss
            if num_bbox > 0: 
                loss_bbox = F.l1_loss(pb[bbox_exist], target_bboxes[bbox_exist], reduction='none')
                loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(box_ops.swig_box_cxcywh_to_xyxy(pb[bbox_exist], mw, mh, device=device), 
                                                                       box_ops.swig_box_cxcywh_to_xyxy(target_bboxes[bbox_exist], mw, mh, device=device, gt=True)))
                batch_bbox_loss.append(loss_bbox.sum() / num_bbox)
                batch_giou_loss.append(loss_giou.sum() / num_bbox)

            # top1 correct noun & top5 correct nouns
            noun_correct_top1 = noun_correct_topk[0]
            noun_correct_top5 = noun_correct_topk.sum(dim=0)

            # convert coordinates
            pb_xyxy = box_ops.swig_box_cxcywh_to_xyxy(cloned_pb, mw, mh, device=device)
            gt_bbox_xyxy = box_ops.swig_box_cxcywh_to_xyxy(cloned_target_bboxes, mw, mh, device=device, gt=True)
            
            # accuracies
            if not eval:
                batch_bbox_acc += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles, 
                                                     noun_correct[i], bbox_exist, t, self.SWiG_json_train, 
                                                     self.idx_to_role)
                batch_bbox_acc_top1 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                                                          noun_correct_top1[i], bbox_exist, t, self.SWiG_json_train, 
                                                          self.idx_to_role)
                batch_bbox_acc_top5 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles, 
                                                          noun_correct_top5[i], bbox_exist, t, self.SWiG_json_train, 
                                                          self.idx_to_role)
            else:
                batch_bbox_acc += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles, 
                                                     noun_correct[i], bbox_exist, t, self.SWiG_json_eval, 
                                                     self.idx_to_role, eval)
                batch_bbox_acc_top1 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles, 
                                                          noun_correct_top1[i], bbox_exist, t, self.SWiG_json_eval, 
                                                          self.idx_to_role, eval) 
                batch_bbox_acc_top5 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles, 
                                                          noun_correct_top5[i], bbox_exist, t, self.SWiG_json_eval, 
                                                          self.idx_to_role, eval) 

        if len(batch_bbox_loss) > 0:
            bbox_loss = torch.stack(batch_bbox_loss).mean()
            giou_loss = torch.stack(batch_giou_loss).mean()
        else:
            bbox_loss = torch.tensor(0., device=device)
            giou_loss = torch.tensor(0., device=device)

        bbox_conf_loss = torch.stack(batch_bbox_conf_loss).mean()
        bbox_acc = torch.stack(batch_bbox_acc)
        bbox_acc_top1 = torch.stack(batch_bbox_acc_top1)
        bbox_acc_top5 = torch.stack(batch_bbox_acc_top5)

        out = {}
        # losses 
        out['loss_vce'] = verb_loss
        out['loss_nce'] = noun_loss
        out['loss_bbox'] = bbox_loss
        out['loss_giou'] = giou_loss
        out['loss_bbox_conf'] = bbox_conf_loss

        # Note that all metrics should be calculated per verb and averaged across verbs.
        ## In the dev and test split of SWiG dataset, there are 50 images for each verb (same number of images per verb).
        ### So, our implementation is correct to calculate metrics for the dev and test split of SWiG dataset. In SR task, many researchers calculate metrics in this way.
        ### We calculate metrics in this way for simple implementation in distributed data parallel setting.

        # accuracies (for verb and noun)
        out['noun_acc_top1'] = noun_acc_topk[0].mean()
        out['noun_acc_all_top1'] = (noun_acc_topk[0] == 100).float().mean()*100
        out['noun_acc_top5'] = noun_acc_topk.sum(dim=0).mean()
        out['noun_acc_all_top5'] = (noun_acc_topk.sum(dim=0) == 100).float().mean()*100
        out['verb_acc_top1'] = verb_acc_topk[0]
        out['verb_acc_top5'] = verb_acc_topk[1]
        out['noun_acc_gt'] = noun_acc.mean()
        out['noun_acc_all_gt'] = (noun_acc == 100).float().mean()*100
        out['mean_acc'] = torch.stack([v for k, v in out.items() if 'noun_acc' in k or 'verb_acc' in k]).mean()
        # accuracies (for bbox)
        out['bbox_acc_gt'] = bbox_acc.mean()
        out['bbox_acc_all_gt'] = (bbox_acc == 100).float().mean()*100
        out['bbox_acc_top1'] = bbox_acc_top1.mean()
        out['bbox_acc_all_top1'] = (bbox_acc_top1 == 100).float().mean()*100
        out['bbox_acc_top5'] = bbox_acc_top5.mean()
        out['bbox_acc_all_top5'] = (bbox_acc_top5 == 100).float().mean()*100

        return out


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    if args.dataset_file == "coco":
        num_classes = 91
    elif args.dataset_file == "swig":
        num_classes = 91
        pass
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = GSRTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox1': args.bbox_loss1_coef, 'loss_giou1': args.giou_loss1_coef,
                   'loss_nce': args.noun_loss_coef, 'loss_vce': args.verb_loss_coef, 
                   'loss_bbox':args.bbox_loss_coef, 'loss_giou':args.giou_loss_coef,
                   'loss_bbox_conf':args.bbox_conf_loss_coef}
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
    criterion = SetCriterion(9927, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    criterion2 = SWiGCriterion(weight_dict=weight_dict, 
                            SWiG_json_train=args.SWiG_json_train, 
                            SWiG_json_eval=args.SWiG_json_dev, 
                            idx_to_role=args.idx_to_role)

    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, criterion2, postprocessors
