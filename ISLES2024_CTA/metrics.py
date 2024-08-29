import warnings

import numpy as np
import torch
import torch.nn.functional as F
from panoptica import (
    ConnectedComponentsInstanceApproximator,
    InputType,
    NaiveThresholdMatching,
    Panoptica_Evaluator,
)


class Dice(object):
    def __init__(
        self,
        nb_labels,
        weights=None,
        input_type="prob",
        dice_type="soft",
        approx_hard_max=True,
        vox_weights=None,
        crop_indices=None,
        area_reg=0.1,
    ):
        self.nb_labels = nb_labels
        self.weights = torch.tensor(weights) if weights is not None else None
        self.vox_weights = torch.tensor(vox_weights) if vox_weights is not None else None
        self.input_type = input_type
        self.dice_type = dice_type
        self.approx_hard_max = approx_hard_max
        self.area_reg = area_reg
        self.crop_indices = crop_indices

        if self.crop_indices is not None and vox_weights is not None:
            self.vox_weights = self.batch_gather(self.vox_weights, self.crop_indices)

    def dice(self, y_true, y_pred):

        if self.crop_indices is not None:
            y_true = self.batch_gather(y_true, self.crop_indices)
            y_pred = self.batch_gather(y_pred, self.crop_indices)

        # if self.input_type == "prob":
        #     y_true = y_true / torch.sum(y_true, dim=0, keepdim=True)
        #     y_true = torch.clamp(y_true, min=torch.finfo(y_true.dtype).eps, max=1)

        #     y_pred = y_pred / torch.sum(y_pred, dim=0, keepdim=True)
        #     y_pred = torch.clamp(y_pred, min=torch.finfo(y_pred.dtype).eps, max=1)

        if self.dice_type == "hard":
            if self.input_type == "prob":
                if self.approx_hard_max:
                    y_pred_op = self._hard_max(y_pred, axis=-1)
                    y_true_op = self._hard_max(y_true, axis=-1)
                else:
                    y_pred_op = self._label_to_one_hot(
                        torch.argmax(y_pred, dim=-1), self.nb_labels
                    )
                    y_true_op = self._label_to_one_hot(
                        torch.argmax(y_true, dim=-1), self.nb_labels
                    )
            else:
                assert self.input_type == "max_label"
                y_pred_op = self._label_to_one_hot(y_pred, self.nb_labels)
                y_true_op = self._label_to_one_hot(y_true, self.nb_labels)
        else:
            assert self.input_type == "prob", "cannot do soft dice with max_label input"
            y_pred_op = y_pred
            y_true_op = y_true

        if y_pred_op.shape[0] == 2:
            y_pred_op = torch.split(y_pred_op, 1, 0)[1]
            y_true_op = torch.split(y_true_op, 1, 0)[1]
        else:
            y_pred_op = torch.split(y_pred_op, 1, 1)[1]
            y_true_op = torch.split(y_true_op, 1, 1)[1]

        sum_dim = (1, 2)
        top = 2 * torch.sum(y_true_op * y_pred_op, dim=sum_dim)
        bottom = torch.sum(y_true_op**2, dim=sum_dim) + torch.sum(y_pred_op**2, dim=sum_dim)

        bottom = torch.maximum(bottom, torch.tensor(self.area_reg, dtype=bottom.dtype))
        return top / bottom

    def mean_dice(self, y_true, y_pred):
        dice_metric = self.dice(y_true, y_pred)

        if self.weights is not None:
            dice_metric *= self.weights
        if self.vox_weights is not None:
            dice_metric *= self.vox_weights

        mean_dice_metric = torch.mean(dice_metric)
        assert torch.isfinite(mean_dice_metric).all(), "metric not finite"
        return mean_dice_metric

    def loss(self, y_true, y_pred):
        dice_metric = self.dice(y_true, y_pred)
        dice_loss = 1 - dice_metric

        if self.weights is not None:
            dice_loss *= self.weights

        mean_dice_loss = torch.mean(dice_loss)
        assert torch.isfinite(mean_dice_loss).all(), "Loss not finite"
        return mean_dice_loss

    def _label_to_one_hot(self, tens, nb_labels):
        y = tens.view(tens.size(0), -1)
        return F.one_hot(y, nb_labels).float()

    def _hard_max(self, tens, axis):
        tensmax = torch.max(tens, dim=axis, keepdim=True).values
        eps_hot = torch.maximum(
            tens - tensmax + torch.finfo(tens.dtype).eps, torch.tensor(0.0, dtype=tens.dtype)
        )
        one_hot = eps_hot / torch.finfo(tens.dtype).eps
        return one_hot

    def batch_gather(self, reference, indices):
        batch_size = reference.size(0)
        indices = torch.stack([torch.arange(batch_size), indices], dim=1)
        return reference[indices[:, 0], indices[:, 1]]


def compute_absolute_volume_difference(im1, im2, voxel_size):
    """
    Computes the absolute volume difference between two masks.

    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    voxel_size : scalar, float (ml)
        If not float, it will be converted.

    Returns
    -------
    abs_vol_diff : float, measured in ml.
        Absolute volume difference as a float.
        Maximum similarity = 0
        No similarity = inf


    Notes
    -----
    The order of inputs is irrelevant. The result will be identical if `im1` and `im2` are switched.
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    voxel_size = voxel_size.astype(float)

    if im1.shape != im2.shape:
        warnings.warn(
            "Shape mismatch: ground_truth and prediction have difference shapes."
            " The absolute volume difference is computed with mismatching shape masks"
        )

    ground_truth_volume = np.sum(im1) * voxel_size
    prediction_volume = np.sum(im2) * voxel_size
    abs_vol_diff = np.abs(ground_truth_volume - prediction_volume)

    return abs_vol_diff


def compute_dice_f1_instance_difference(ground_truth, prediction, empty_value=1.0):
    """
    Computes the lesion-wise F1-score, instance count difference, and Dice score between two masks.

    Parameters
    ----------
    ground_truth : array-like, int
        Any array of arbitrary size. If not int, it will be converted.
    prediction: array-like, bool
        Any other array of identical size as 'ground_truth'. If not int, it will be converted.
    empty_value : scalar, float.

    Returns
    -------
    f1_score : float
        Instance-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty F1-Score = empty_value

    -------
    dice_score : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        If both images are empty (sum equal to zero) = empty_value
    -------
    instance_count_difference : int
        Absolute instance count difference as integer.
        Maximum similarity = 0
        No similarity = --> inf

    """

    ground_truth = np.asarray(ground_truth).astype(int)
    prediction = np.asarray(prediction).astype(int)

    evaluator = Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(),
    )

    result, _ = evaluator.evaluate(prediction, ground_truth, verbose=False)["ungrouped"]

    instance_count_difference = abs(
        result.num_ref_instances - result.num_pred_instances
    )  # compute lesion count difference

    if result.num_ref_instances == 0 and result.num_pred_instances == 0:
        f1_score = empty_value
        dice_score = empty_value
    else:
        f1_score = result.rq  # get f1-score
        dice_score = result.global_bin_dsc

    return f1_score, instance_count_difference, dice_score
