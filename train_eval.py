from tqdm import tqdm
from functions import *


# Model Training and Evalaution

def train_and_eval(model, criterion, optimizer, NUM_EPOCHS, device, train_dataloader, val_dataloader):

    train_loss = []
    val_loss = []
    cMAP_epoch = []

    for i in range(NUM_EPOCHS):
        
        # Train
        model.train()
        train_running_loss = 0.0
        
        for inputs, targets in tqdm(train_dataloader):

            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            targets = targets.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.cpu()
            targets = targets.cpu()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()
            
            
        # Validation
        model.eval()
        val_running_loss = 0.0
        cMAP_val = []
        
        for inputs, targets in val_dataloader:

            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            targets = targets.float()
            outputs = model(inputs)
            outputs = outputs.detach().cpu()
            targets = targets.cpu()
            loss = criterion(outputs, targets)
            
            val_running_loss += loss.item()

            for batch_idx in range(len(outputs)):
                cMAP_batch = cMAP(outputs[batch_idx], targets[batch_idx])
                cMAP_val.append(cMAP_batch.item())

                
        train_running_loss /= len(train_dataloader)
        val_running_loss /= len(val_dataloader)
        avg_cMAP = sum(cMAP_val) / len(cMAP_val)
        
        train_loss.append(train_running_loss)
        val_loss.append(val_running_loss)
        cMAP_epoch.append(avg_cMAP)
            
        print(f"Epoch: {i+1}/{NUM_EPOCHS} | Training Loss: {train_running_loss:.5f} | Validation Loss: {val_running_loss:.5f} | cMAP: {avg_cMAP:.5f}")

    return train_loss, val_loss, cMAP_epoch