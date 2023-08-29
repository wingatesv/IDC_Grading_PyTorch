import torch
from torch import nn
import torchvision
import torchvision.models as models


class IDC_Grading_Model(nn.Module):
  def __init__(self, feature_extractor, class_weights = None):
    super().__init__()
    # initialise loss function with class_weights, no weights are given in mocel testing
    self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # get Imagenet weights from different backbone
    if feature_extractor == 'efficientnet_b0':
      weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    elif feature_extractor == 'efficientnet_v2_s':
      weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    elif feature_extractor == 'resnet50':
      weights = models.ResNet50_Weights.IMAGENET1K_V1
    elif feature_extractor == 'mobilenet_v2':
      weights = models.MobileNet_V2_Weights.IMAGENET1K_V1

    # initialise backbone
    self.feature_extractor = models.__dict__[feature_extractor](weights=weights)

    #  set backbone training to False
    for param in self.feature_extractor.parameters():
      param.requires_grad = False
    
    #  resnet50 only has fc instead classifier layer
    if feature_extractor == 'resnet50':
      num_ftrs = self.feature_extractor.fc.in_features
      self.feature_extractor.fc = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(num_ftrs, 256),
                                nn.ReLU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(256, 4))
     
    else:
      num_ftrs = self.feature_extractor.classifier[1].in_features
      self.feature_extractor.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(num_ftrs, 256),
                                nn.ReLU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(256, 4))
          

  #  forward pass
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.feature_extractor(x)



def train_step(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0

    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # get loss and acc
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    train_loss /= num_batches
    correct /= size
    print(f"Train acc: {(100*correct):>0.1f}%, Train loss: {train_loss:>8f} \n")

def val_step(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss, correct = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # model prediction
            pred = model(X)
            # get val loss and acc
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Val acc: {(100*correct):>0.1f}%, Val loss: {test_loss:>8f} \n")

    return correct


def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_true)) * 100
  return acc

def test(y_true, dataloader, model, device):
  # 1. Make predictions with trained model
  y_preds = []
  model.eval()
  with torch.inference_mode():
    for X, y in dataloader:
      # Send data and targets to target device
      X, y = X.to(device), y.to(device)
      # Do the forward pass
      y_logit = model(X)
      # Turn predictions from logits -> prediction probabilities -> predictions labels
      y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
      # Put predictions on CPU for evaluation
      y_preds.append(y_pred.cpu())
  # Concatenate list of predictions into a tensor
  y_pred_tensor = torch.cat(y_preds)
  acc = accuracy_fn(y_true, y_pred_tensor)
  return acc
