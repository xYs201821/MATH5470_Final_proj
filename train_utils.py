from tqdm import tqdm
import torch
import time
import os
import torch.nn.functional as F
import pandas as pd
from utils import convert_to_python_types
class recorder():
    def __init__(self, patience=2, tolerence=0, ckpt_path='checkpoint.pt'):
        self.patience = patience
        self.tolerence = tolerence
        self.ckpt_path = ckpt_path
        self.best_loss = float('inf')
        self.counter = 0
        self.stop_training = False
    
    def __call__(self, val_loss, epoch):
        if self.best_loss + self.tolerence < val_loss:
            self.counter += 1
            print(f"Early_Stopping Counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                print("[INFO]Early_Stopping is triggered!")
                self.stop_training = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0 
        return self.stop_training
    
def trainer(model, train_loader, val_loader, criterion, optimizer, num_epochs,
             device='cuda' if torch.cuda.is_available() else 'cpu',
               patience=2, tolerence=0, ckpt_path='./ckpt', name='baseline'):
    
    def epoch_train():
        TP = TN = FP = FN = train_loss = 0.
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.shape[0]
            predictions = F.softmax(outputs, dim=1)
            _, pred = torch.max(predictions, dim=1)
            TP += ((pred == labels) & (labels == 1)).sum()
            TN += ((pred == labels) & (labels == 0)).sum()
            FP += ((pred != labels) & (labels == 0)).sum()
            FN += ((pred != labels) & (labels == 1)).sum()
        TOT = TP + TN + FP + FN
        accuracy = (TP + TN) / TOT
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
            # if (i+1) % 100 == 0:
            #     print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
            #     running_loss = .0 
        train_loss /= len(train_loader.dataset)
        metrics = {
            'loss': train_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        }
        print(f"Epoch [{epoch+1}/{num_epochs}], Training_Loss: {train_loss:.4f}")
        return convert_to_python_types(metrics)
    
    def epoch_eval():
        TP = TN = FP = FN = val_loss = 0.
        with torch.no_grad():
            for i, (images, labels) in tqdm(enumerate(val_loader)):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.shape[0]        
                predictions = F.softmax(outputs, dim=1)
                _, pred = torch.max(predictions, dim=1)
                TP += ((pred == labels) & (labels == 1)).sum()
                TN += ((pred == labels) & (labels == 0)).sum()
                FP += ((pred != labels) & (labels == 0)).sum()
                FN += ((pred != labels) & (labels == 1)).sum()
        TOT = TP + TN + FP + FN
        accuracy = (TP + TN) / TOT
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
        metrics = {
            'loss': val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        }
        return convert_to_python_types(metrics)
    start = time.time()
    train_metrics = []
    val_metrics = []
    record = recorder(patience=patience, tolerence=tolerence, ckpt_path=ckpt_path)
    print(f"[INFO]Training on {device}!")
    for epoch in range(num_epochs):
        model.train()
        train_metrics.append(epoch_train())
        model.eval()
        val_metrics.append(epoch_eval())
        early_stopping = record(val_metrics[-1]['loss'], epoch)
        if early_stopping:
            print(f"[INFO]Early stopping at epoch {epoch + 1}")
            print(f"[INFO]Best epoch: {record.best_epoch + 1}")
            print(f"[INFO]Best validation loss: {record.best_loss:.4f}")
            break
        save_path = os.path.join(ckpt_path, f'{name}_{epoch + 1}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Epoch [{epoch +1}/{num_epochs}], Checkpoint saved at {save_path}")
    model.load_state_dict(torch.load(os.path.join(ckpt_path, f'{name}_{record.best_epoch + 1}.pth')))
    time_elapsed = time.time() - start
    train_df = pd.DataFrame(train_metrics).add_suffix('_train')
    val_df = pd.DataFrame(val_metrics).add_suffix('_val')
    if len(train_df) == len(val_df):
        metrics_df = pd.concat([train_df, val_df], axis=1)
        metrics_df['iteration'] = range(len(metrics_df))
    print(f'[INFO]Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model, metrics_df