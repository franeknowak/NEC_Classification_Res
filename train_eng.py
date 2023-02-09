import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import wandb
from sklearn.metrics import precision_score, f1_score, recall_score


def train_step( model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    """
    Trains a PyTorch model for a singe epoch

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:
    
    (0.1112, 0.8743)
    """
 
    # Put model in train mode
    model.train() 

    # Allocate space for loss and accuracy values
    train_loss, train_acc = 0, 0 

    # Loop through the dataloader data batches
    # Enumerate through the dataloader to get individual batches, from there take batch number, X - image tensor and y - label
    for batch, (X, y) in enumerate(dataloader):
        
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Caclulate current loss and then add accumulate it to later average it across entire epoch
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Zero the optimizer
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Get the predicted class and add accuracies for each entire epoch
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        
        # Log the training accuracy and loss each 480 batch
        if batch % 10 == 0:
            batch_acc = (y_pred_class == y).sum().item()/len(y_pred)
            wandb.log({"Train_10batch_acc": batch_acc})
            wandb.log({"Train_10batch_loss": loss.item()})
    
    # Divide by number of batches to get loss and accuracy for the epoch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def test_step(  model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                device: torch.device) -> Tuple[float, float]:
    """
    Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:
    
    (0.0223, 0.8985)
  """
    # Put model in eval mode
    model.eval()

    # Allocate space for test_loss and test_accuracy
    test_loss, test_acc = 0, 0

    y_pred = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
          
            X, y = X.to(device), y.to(device)

            #if y.numel() == 1:
            #    y = y.unsqueeze(dim = 0)
                
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            y_pred = torch.cat((y_pred, test_pred_labels), dim=0)
            y_true = torch.cat((y_true, y), dim=0)

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = ((y_pred == y_true).sum().item()/len(y_pred))
    y_true, y_pred = y_true.to('cpu'), y_pred.to('cpu')

    test_precision = precision_score(y_true, y_pred, labels=[0,1,2], average='macro')   
    test_recall = recall_score(y_true, y_pred, labels=[0,1,2], average='macro')
    test_f1 = f1_score(y_true, y_pred, labels=[0,1,2], average='macro')


    return test_loss, test_acc, test_precision, test_recall, test_f1


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
  """
  # Create empty results dictionary
  results = {"Epoch Train Loss": [],
      "Epoch Train Accuracy": [],
      "Epoch Test Loss": [],
      "Epoch Test Accuracy": [],
      "Epoch Test Precision": [],
      "Epoch Test Recall": [],
      "Epoch Test F1": []
    }
  
  # Loop through training and testing steps for a number of epochs
  for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc, test_precision, test_recall, test_f1 = test_step(model=model,
                                                                            dataloader=test_dataloader,
                                                                            loss_fn=loss_fn,
                                                                            device=device)
      
        # Update user on progress
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"test_precision: {test_precision:.4f} | "
            f"test_recall: {test_recall:.4f} | "
            f"test_f1: {test_f1:.4f} | "
        )

        # Update results dictionary
        results["Epoch Train Loss"].append(train_loss)
        results["Epoch Train Accuracy"].append(train_acc)
        results["Epoch Test Loss"].append(test_loss)
        results["Epoch Test Accuracy"].append(test_acc)
        results["Epoch Test Precision"].append(test_precision)
        results["Epoch Test Recall"].append(test_recall)
        results["Epoch Test F1"].append(test_f1)
        
        
        wandb.log({"Epoch Train Loss": train_loss})
        wandb.log({"Epoch Test Loss": test_loss})
        wandb.log({"Epoch Train Accuracy": train_acc})
        wandb.log({"Epoch Test Accuracy": test_acc})
        wandb.log({"Epoch Test Precision": test_precision})
        wandb.log({"Epoch Test Recall": test_recall})
        wandb.log({"Epoch Test F1": test_f1})


    # Return the filled results at the end of the epochs
  return results
