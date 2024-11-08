import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import torch.optim as optim
from loguru import logger
from utils import build_transforms, get_optimizer, get_scheduler, get_criterion
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def setup_logger(config: DictConfig):
    logger.remove()
    logger.add(
        os.path.join(".hydra", "run.log"),
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        backtrace=True,
        diagnose=True,
        colorize=True,
        level="DEBUG",
    )
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        backtrace=True,
        diagnose=True,
        colorize=True,
        level="DEBUG",
    )
    logger.info("------Configuration Details:")
    logger.info(OmegaConf.to_yaml(config))

@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    weight_dir = os.path.join(os.getcwd(),'weights')
    os.makedirs(weight_dir, exist_ok=True)
    best_model_path = os.path.join(weight_dir, 'best_model.pth') 

    # setup logger
    setup_logger(cfg)
    
    best_accuracy = -1
    num_epochs = cfg.num_epochs
    batch_size = cfg.batch_size
    num_classes = cfg.model.num_classes
    best_epoch = -1

    if cfg.model.pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None
    
    # Load the pre-trained model
    model = torch.hub.load('pytorch/vision:v0.10.0', cfg.model.architecture, pretrained=weights)
    
    for param in model.parameters():
        param.requires_grad = True
    
    # Modify the last fully connected layer for a custom number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Build data transformations from config
    try:
        train_transforms = build_transforms(cfg.data.transforms.train)
        val_transforms = build_transforms(cfg.data.transforms.val)
    except ValueError as e:
        logger.error(f"Error in transform configuration: {e}")
        exit(1)
    
    train_dataset = datasets.ImageFolder(root=cfg.data.train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=cfg.data.val_dir, transform=val_transforms)
    print("Training Classes:", train_dataset.classes)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    
    criterion = get_criterion(cfg.criterion.name, cfg.criterion.params)
    optimizer = get_optimizer(cfg.optimizer.name, model.parameters(), cfg.optimizer.params)
    scheduler = get_scheduler(cfg.scheduler.name, optimizer, cfg.scheduler.params)
    
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=".")
    
    # Training and validation loop
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        running_train_loss = 0.0
        train_correct, train_total = 0, 0
        train_all_labels, train_all_predictions = [], []
        train_normal_correct, train_abnormal_correct, train_benign_correct  = 0, 0, 0
        train_normal_total, train_abnormal_total, train_benign_total = 0, 0, 0
    
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
    
            # Zero the gradients
            optimizer.zero_grad()
    
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
    
            # Accumulate training metrics
            running_train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            # labels are in alphabetical order
            train_normal_count = (labels == 2).sum().item()
            train_abnormal_count = (labels == 0).sum().item()
            train_benign_count = (labels == 1).sum().item()

            # Store all labels and predictions for metric calculation
            train_all_labels.extend(labels.cpu().numpy())
            train_all_predictions.extend(preds.cpu().numpy())

            train_normal_total += train_normal_count
            train_abnormal_total += train_abnormal_count
            train_benign_total += train_benign_count
            train_normal_correct += ((preds == labels) & (labels == 2)).sum().item()
            train_abnormal_correct += ((preds == labels) & (labels == 0)).sum().item()
            train_benign_correct += ((preds == labels) & (labels == 1)).sum().item()

            train_accuracy = 100 * train_correct / train_total
    
        # Calculate average training loss and accuracy
        train_epoch_loss = running_train_loss / len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        train_normal_accuracy = train_normal_correct / train_normal_total if train_normal_total > 0 else 0
        train_abnormal_accuracy = train_abnormal_correct / train_abnormal_total if train_abnormal_total > 0 else 0
        train_benign_accuracy = train_benign_correct / train_benign_total if train_benign_total > 0 else 0

    
        # Validation phase
        model.eval()
        val_running_loss = 0
        val_correct, val_total = 0, 0
        val_all_labels, val_all_predictions = [], []
        val_normal_correct, val_abnormal_correct, val_benign_correct  = 0, 0, 0
        val_normal_total, val_abnormal_total, val_benign_total = 0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
    
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
    
                # Accumulate validation metrics
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                # labels are in alphabetical order
                val_normal_count = (labels == 2).sum().item()
                val_abnormal_count = (labels == 0).sum().item()
                val_benign_count = (labels == 1).sum().item()

                # Store all labels and predictions for metric calculation
                val_all_labels.extend(labels.cpu().numpy())
                val_all_predictions.extend(preds.cpu().numpy())

                val_normal_total += val_normal_count
                val_abnormal_total += val_abnormal_count
                val_benign_total += val_benign_count
                val_normal_correct += ((preds == labels) & (labels == 2)).sum().item()
                val_abnormal_correct += ((preds == labels) & (labels == 0)).sum().item()
                val_benign_correct += ((preds == labels) & (labels == 1)).sum().item()

                val_accuracy = val_correct / val_total

        # After the loop, calculate metrics
        val_accuracy = (val_correct / val_total) * 100
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_normal_accuracy = val_normal_correct / val_normal_total if val_normal_total > 0 else 0
        val_abnormal_accuracy = val_abnormal_correct / val_abnormal_total if val_abnormal_total > 0 else 0
        val_benign_accuracy = val_benign_correct / val_benign_total if val_benign_total > 0 else 0

        # Update the learning rate
        scheduler.step(val_epoch_loss)
    
        # Log metrics to TensorBoard
        writer.add_scalar('Train/Loss', train_epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
        writer.add_scalar('Train/Normal_Accuracy', train_normal_accuracy, epoch)
        writer.add_scalar('Train/Abnormal_Accuracy', train_abnormal_accuracy, epoch)
        writer.add_scalar('Train/Benign_Accuracy', train_benign_accuracy, epoch)
        writer.add_scalar('Validation/Loss', val_epoch_loss, epoch)
        writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
        writer.add_scalar('Validation/Normal_Accuracy', val_normal_accuracy, epoch)
        writer.add_scalar('Validation/Abnormal_Accuracy', val_abnormal_accuracy, epoch)
        writer.add_scalar('Validation/Benign_Accuracy', val_benign_accuracy, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

    
        # Save the model if validation accuracy is the best we've seen so far
        if val_accuracy > best_accuracy:
            best_epoch = epoch
            best_accuracy = val_accuracy
            best_model_filename = f"best_model_epoch_{best_epoch}.pth"
            best_model_path = os.path.join(weight_dir, best_model_filename)
            torch.save(model.state_dict(), best_model_path)  # Save model state to best.pt
            logger.info(f"New best model saved with accuracy: {best_accuracy:.4f} at {best_model_path}")
    
        # Print epoch statistics
        logger.info(f"Epoch [{epoch}/{num_epochs}], "
                     f"Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                     f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                     f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
    # Close the TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()
