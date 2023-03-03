import torch

from torchmetrics import Metric
from torchvision.ops import masks_to_boxes
from cc_torch import connected_components_labeling

class MissedWeeds(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("missed_weeds", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds, target):
        B = preds.shape[0]

        crop_to_weed = preds[:, 1, ::] * target[:, 2, ::]
        weed_to_crop = preds[:, 2, ::] * target[:, 1, ::]
        
        residual = target[:, 2, ::] - preds[:, 2, ::] 

        conn_pred_weeds = torch.cat([connected_components_labeling(preds[i, :, 2, ::].type(torch.uint8)) for i in range(B)])
        conn_pred_crop = torch.cat([connected_components_labeling(preds[:, 1, ::].type(torch.uint8)) for i in range(B)])
        conn_pred_background = torch.cat([connected_components_labeling(preds[:, 0, ::].type(torch.uint8)) for i in range(B)])

        conn_target_weeds = torch.cat([connected_components_labeling((target==2).type(torch.uint8)) for i in range(B)])
        conn_target_crop = torch.cat([connected_components_labeling((target==1).type(torch.uint8)) for i in range(B)])
        conn_target_background = torch.cat([connected_components_labeling((target==0).type(torch.uint8)) for i in range(B)])

        values, counts = torch.unique(conn_pred_weeds, return_counts=True, dim=0)
        conn_pred_weeds[residual > 0] = 0
        reduced_values, reduced_counts = torch.unique(conn_pred_weeds, return_counts=True, dim=0)

    def compute(self):
        return self.missed_weeds 