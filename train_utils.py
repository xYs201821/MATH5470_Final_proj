from tqdm import tqdm
import torch
import time
import os
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
        model.train()
        running_loss = .0
        train_loss = .0
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.shape[0]
            running_loss += loss.item() * images.shape[0]
            # if (i+1) % 100 == 0:
            #     print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
            #     running_loss = .0 
        train_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training_Loss: {train_loss:.4f}")
        return train_loss
    
    def epoch_eval():
        model.eval()
        val_loss = .0
        with torch.no_grad():
            for i, (images, labels) in tqdm(enumerate(val_loader)):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.shape[0]
        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
        return val_loss
    start = time.time()
    train_losses = []
    val_losses = []
    record = recorder(patience=patience, tolerence=tolerence, ckpt_path=ckpt_path)
    print(f"[INFO]Training on {device}!")
    for epoch in range(num_epochs):
        train_losses.append(epoch_train())
        val_losses.append(epoch_eval())
        early_stopping = record(val_losses[-1], epoch)
        if early_stopping:
            print(f"[INFO]Early stopping at epoch {epoch + 1}")
            print(f"[INFO]Best epoch: {record.best_epoch + 1}")
            print(f"[INFO]Best validation loss: {record.best_loss:.4f}")
            break
        save_path = os.path.join(ckpt_path, f'{name}_{epoch + 1}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Epoch [{epoch +1}/{num_epochs}], Checkpoint saved at {save_path}")
    model.load_state_dict(torch.load(os.path.join(ckpt_path, f'{name}_{record.best_epoch + 1}.pth'),
                                     weights_only=True))
    time_elapsed = time.time() - start
    print(f'[INFO]Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model, train_losses, val_losses, epoch