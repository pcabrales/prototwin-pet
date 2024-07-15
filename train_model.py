import csv
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from utils import plot_losses
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, epochs=10, patience=20, model_dir='.', timing_dir = '.',
          save_plot_dir='.', output_transform=None, losses_dir='.', deviations=False, accumulation_steps=1):
    
    start_time = time.time()  # Timing the training time
    # Initializing the optimizer for the model parameters
    optim = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=2e-6) 

    l2_loss = nn.MSELoss()
    # l1_loss = nn.L1Loss()
    beta = 2e-2
    threshold = 0.1
    best_val_loss = np.inf
    wait = 0  # for early stopping
    training_losses = []
    val_losses = [] 
    alpha_deviation = 0.005 # Loss between predicted and obtained deviations
    with open(losses_dir[:-4] + "-running.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])
    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0

        # Training loop
        batch = 0
        for batch_input, batch_target, deviations_target in tqdm(train_loader):
            loss = 0
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            if deviations:
                batch_output, deviations_output = model(batch_input, return_bottleneck=True)
                loss +=  alpha_deviation * l2_loss(deviations_output, deviations_target.to(device))
            else:
                batch_output = model(batch_input)  # generating images
            if output_transform is not None:
                batch_output = output_transform.inverse(batch_output)
                batch_target = output_transform.inverse(batch_target)
            if epoch * batch_target.shape[0] + batch > 150 and beta < 5:
                beta = beta * 1.03
            loss += l2_loss(batch_output, batch_target)
            loss.backward()  # backprop
            
            if batch % accumulation_steps == 0:
                optim.step()
                optim.zero_grad()  # resetting gradients
                
            train_loss += loss.item()
            batch += 1
            with open(timing_dir, "w") as file:
                file.write(f'epoch {epoch} batch {batch}\n')
                
        # Validation loop
        with torch.no_grad():
            for batch_input, batch_target, deviations_target in tqdm(val_loader):
                loss = 0
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                if deviations:
                    batch_output, deviations_output = model(batch_input, return_bottleneck=True)
                    loss +=  alpha_deviation * l2_loss(deviations_output, deviations_target.to(device))
                    print(deviations_output[0].detach().cpu().numpy(), deviations_target[0].detach().cpu().numpy(), loss.item())
                else:  
                    batch_output = model(batch_input)  # generating images
                if output_transform is not None:
                    batch_output = output_transform.inverse(batch_output)
                    batch_target = output_transform.inverse(batch_target)
                    
                loss += l2_loss(batch_output, batch_target) 
                val_loss += loss.item()
        
        scheduler.step() 
        
        # Calculate average losses (to make it independent of batch size)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch} training loss: {avg_train_loss}')
        print(f'Epoch {epoch} validation loss: {avg_val_loss}')

        # Log the losses for plotting
        training_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        plot_losses(training_losses, val_losses, save_plot_dir[:-4] + "-running.jpg")
        # Save losses
        with open(losses_dir[:-4] + "-running.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, avg_train_loss, avg_val_loss])

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            torch.save(model, model_dir)
        else:
            wait += 1
            if wait >= patience:
                print(f"Stopping early at epoch {epoch}")
                epoch = epoch - patience
                break
        
        
    # End and save timing, plots and definitive losses
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Training time: {elapsed_time} seconds')
    # Save to file
    with open(timing_dir, "w") as file:
        file.write(f'Training time: {elapsed_time} seconds. Best epoch: {epoch}')
        
    os.rename(save_plot_dir[:-4] + "-running.jpg", save_plot_dir)
    if os.path.exists(save_plot_dir[:-4] + "-running.eps"):
        os.rename(save_plot_dir[:-4] + "-running.eps", save_plot_dir[:-4] + ".eps")
    os.rename(losses_dir[:-4] + "-running.csv", losses_dir)
        
    return model
