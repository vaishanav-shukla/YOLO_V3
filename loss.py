# ignoring the iou threshold greater than 0.5
# thats why we initialized it to -1 so we can ignore it while calculating the loss


"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference is use of CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn

from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        # these value can be change to prioritise a particular loss more or less than other
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        # here we dont want to penalise the anchors which had more than 0.5 iou_threshold
        # thats why we have used 0 and 1 only
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        # For the anchors in all cells that do not have an object assigned to them
        # i.e. all indices that are set to one in noobj we want to incur loss only for their object score.
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        # here we simply dont penalise the way we did in above case
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        # above we have chosen anchors for a particular scale
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        #  we have transformed it to log transform for better gradient stability (consider off set formula )
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        # we will simply use a mean squared error loss in the positions where there actually are objects.
        # All predictions where there is no corresponding target bounding box will be ignored
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        # below we have used log on target instead of exponantiating the prediction which is just inverse
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        # We will only incur loss for the class predictions where there actually is an object.
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #print("__________________________________")
        #print(self.lambda_box * box_loss)
        #print(self.lambda_obj * object_loss)
        #print(self.lambda_noobj * no_object_loss)
        #print(self.lambda_class * class_loss)
        #print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )