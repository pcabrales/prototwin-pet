import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import numpy as np
import time
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import csv
from utils import gamma_index, plot_losses
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, epochs=10, patience=20, model_dir='.', timing_dir = '.',
          save_plot_dir='.', output_transform=None, losses_dir='.', deviations=False, accumulation_steps=1):
    
    start_time = time.time()  # Timing the training time
    # Initializing the optimizer for the model parameters
    optim = torch.optim.AdamW(model.parameters(), lr=0.0001)  ###0.001 before
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=2e-6) 

    l2_loss = nn.MSELoss()
    # l1_loss = nn.L1Loss()
    beta = 2e-2
    threshold = 0.1
    alpha = 0 / 1000  # ratio between the losses
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
            loss += l2_loss(batch_output, batch_target) # (1-alpha) * l2_loss(batch_output, batch_target) + alpha * (1 - gamma_index(batch_output, batch_target, beta=beta, threshold=threshold))
            # print((1-alpha) * l2_loss(batch_output, batch_target).item(), alpha * (1 - gamma_index(batch_output, batch_target, beta=beta, threshold=threshold)).item())
            
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
                    
                loss += l2_loss(batch_output, batch_target) # (1-alpha) * l2_loss(batch_output, batch_target) + alpha * (1 - gamma_index(batch_output, batch_target, beta=beta, threshold=threshold))
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


def train_adversarial(model, discrimator_model, train_loader, val_loader, epochs=10, patience = 20, model_dir='.', timing_dir = '.',
          save_plot_dir='.', output_transform=None, losses_dir='.'):
    start_time = time.time()  # Timing the training time
    
    # Initializing the optimizer for the model parameters
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    discriminator_optim = torch.optim.AdamW(discrimator_model.parameters(), lr=0.0005)
    
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    discriminator_loss = nn.BCELoss()
    
    beta = 2e-2
    threshold = 0.2
    alpha = 0.0 #50 / 1000  # ratio between the losses
    lambda_discriminator = 0.02
    tolerance = 0.03  # Tolerance per unit for gamma index
    
    best_val_loss = np.inf
    wait = 0  # for early stopping
    
    training_losses = []
    val_losses = []
     
    training_discriminator_losses = []
    val_discriminator_losses = [] 
    
    with open(losses_dir[:-4] + "-running.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Discriminator Training Loss', 'Discriminator Validation Loss'])
        
    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_discriminator_loss = 0.0
        val_discriminator_loss = 0.0

        # Training loop
        batch = 0
        for batch_input, batch_target, _ in tqdm(train_loader):
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            batch_output = model(batch_input)  # generating images
            if output_transform is not None:
                batch_output = output_transform.inverse(batch_output)
                batch_target = output_transform.inverse(batch_target)
            
            # Train discriminator
            discriminator_optim.zero_grad()
            # Train discriminator with real images
            real_label = torch.ones(batch_target.size(0), 1).to(device)
            output_real = discrimator_model(batch_target)
            loss_discriminator_real = discriminator_loss(output_real, real_label)
            # Train with discriminator with generated images
            fake_label = torch.zeros(batch_output.size(0), 1).to(device)
            output_fake = discrimator_model(batch_output.detach())
            loss_discriminator_fake = discriminator_loss(output_fake, fake_label)      
            # Combine losses and update
            loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) / 2
            loss_discriminator.backward()
            discriminator_optim.step()
            train_discriminator_loss += loss_discriminator.item()
            
            # Train generator
            optim.zero_grad()  # resetting gradients
            discriminator_output = discrimator_model(batch_output)
            loss_fooling_discriminator = discriminator_loss(discriminator_output, real_label)
            
            if epoch * batch_target.shape[0] + batch > 150 and beta < 5:
                beta = beta * 1.03
                
            loss = (1-alpha) * l1_loss(batch_output, batch_target) \
                + alpha * (1 - gamma_index(batch_output, batch_target, beta=beta, threshold=threshold)) \
                + lambda_discriminator * loss_fooling_discriminator       
            
            ###
            print("l1 loss: ", l1_loss(batch_output, batch_target).item())
            print("loss fooling discriminator: ", lambda_discriminator * loss_fooling_discriminator.item())    
            ### 
             
            loss.backward()  # backprop
            optim.step()
            train_loss += loss.item()
            
            batch += 1
            with open(timing_dir, "w") as file:
                file.write(f'epoch {epoch} batch {batch}\n')
                
        # Validation loop
        with torch.no_grad():
            for batch_input, batch_target, _ in tqdm(val_loader):
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                batch_output = model(batch_input)
                if output_transform is not None:
                    batch_output = output_transform.inverse(batch_output)
                    batch_target = output_transform.inverse(batch_target)
                
                # Val discriminator
                # Train discriminator with real images
                real_label = torch.ones(batch_target.size(0), 1).to(device)
                output_real = discrimator_model(batch_target)
                loss_discriminator_real = discriminator_loss(output_real, real_label)
                # Train with discriminator with generated images
                fake_label = torch.zeros(batch_output.size(0), 1).to(device)
                output_fake = discrimator_model(batch_output.detach())
                loss_discriminator_fake = discriminator_loss(output_fake, fake_label)      
                # Combine losses and update
                loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) / 2
                val_discriminator_loss += loss_discriminator.item()
                
                # Val generator
                optim.zero_grad()  # resetting gradients
                discriminator_output = discrimator_model(batch_output)
                loss_fooling_discriminator = discriminator_loss(discriminator_output, real_label)
                loss = (1-alpha) * l2_loss(batch_output, batch_target) \
                    + alpha * (1 - gamma_index(batch_output, batch_target, tolerance=tolerance, beta=beta, threshold=threshold)) \
                    + lambda_discriminator * loss_fooling_discriminator
                val_loss += loss.item()

        # Calculate average losses (to make it independent of batch size)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_discriminator_loss = train_discriminator_loss / len(train_loader)
        avg_val_discriminator_loss = val_discriminator_loss / len(val_loader)
        

        print(f'Epoch {epoch} training loss: {avg_train_loss}')
        print(f'Epoch {epoch} validation loss: {avg_val_loss}')
        print(f'Epoch {epoch} discriminator training loss: {avg_train_discriminator_loss}')
        print(f'Epoch {epoch} discriminator validation loss: {avg_val_discriminator_loss}')
        
        # Log the losses for plotting
        training_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        training_discriminator_losses.append(avg_train_discriminator_loss)
        val_discriminator_losses.append(avg_val_discriminator_loss)
            
        # Save losses
        plot_losses(training_losses, val_losses, save_plot_dir[:-4] + "-running.jpg")
        with open(losses_dir, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, val_loss, train_discriminator_loss, val_discriminator_loss])

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
        
    # End and save timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Training time: {elapsed_time} seconds')
    # Save to file
    with open(timing_dir, "w") as file:
        file.write(f'Training time: {elapsed_time} seconds')
    
    data = {'Epoch': range(len(training_losses)),'Generator Training': training_losses,'Generator Validation': val_losses,
            'Discriminator Training': training_discriminator_losses,'Discriminator Validation': val_discriminator_losses}
    df = pd.DataFrame(data)
    df = pd.melt(df, id_vars=['Epoch'], value_vars=df.columns[1:], var_name='Type', value_name='Loss')
    sns.set()
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    sns.lineplot(ax=axes[0], x='Epoch', y='Loss', hue='Type', 
             data=df[df['Type'].str.contains('Generator')], palette="viridis", linewidth=2.5)
    axes[0].set_title('Generator Loss')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].set_xlabel('Epochs')
    
    sns.lineplot(ax=axes[0], x='Epoch', y='Loss', hue='Type', 
             data=df[df['Type'].str.contains('Discriminator')], palette="viridis", linewidth=2.5)
    axes[0].set_title('Discriminator Loss')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].set_xlabel('Epochs')
    plt.savefig(save_plot_dir[:-4] + "-running.jpg", dpi=300, bbox_inches='tight')
        
    os.rename(save_plot_dir[:-4] + "-running.jpg", save_plot_dir)
    os.rename(losses_dir[:-4] + "-running.csv", losses_dir)
    return model


def train_n2n(model, train_loader, val_loader, epochs=10, patience = 5, model_dir='.', timing_dir = '.',
          save_plot_dir='.', output_transform=None, losses_dir='.'):
    
    start_time = time.time()  # Timing the training time
    # Initializing the optimizer for the model parameters
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)#, weight_decay=0.001) ###
    l2_loss = nn.MSELoss()
    best_val_loss = np.inf
    wait = 0  # for early stopping
    training_losses = []
    val_losses = []
    with open(losses_dir[:-4] + "-running.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])
    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        batch = 0
        # Training loop
        for batch_lowres, _, _  in tqdm(train_loader): 
            batch_train_loss = 0.0
            optim.zero_grad()
            for i in range(len(batch_lowres)):
                output_i = model(batch_lowres[i].to(device))
                for j in range(len(batch_lowres)):
                    if i != j:
                        batch_train_loss += l2_loss(output_i, batch_lowres[j].to(device))

            batch += 1
            with open(timing_dir, "w") as file:
                file.write(f'epoch {epoch} batch {batch}\n') 
            batch_train_loss.backward()  # backprop
            optim.step()
            train_loss += batch_train_loss.item() 
                           
        # Validation loop
        with torch.no_grad():
            for batch_lowres, batch_highres, _ in tqdm(val_loader):
                batch_val_loss = 0.0
                batch_highres = batch_highres.to(device)
                batch_output = 0.
                for i in range(len(batch_lowres)):
                    batch_output += model(batch_lowres[i].to(device)) / len(batch_lowres)  # combining outputs  
                       
                if output_transform is not None:
                    batch_output = output_transform.inverse(batch_output)
                    batch_highres = output_transform.inverse(batch_highres) 
                batch_val_loss += l2_loss(batch_output, batch_highres)
            
                val_loss += batch_val_loss.item()

        # Calculate average losses (to make it independent of batch size)
        n_comb = len(batch_lowres) * (len(batch_lowres) - 1)  # combinations between lowres dirs
        avg_train_loss = train_loss / n_comb/ len(train_loader)
        avg_val_loss = val_loss / n_comb / len(val_loader)

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
                break
            
        
    # End and save timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Training time: {elapsed_time} seconds')
    # Save to file
    with open(timing_dir, "w") as file:
        file.write(f'Training time: {elapsed_time} seconds. Best epoch: {epoch}')
        
    os.rename(save_plot_dir[:-4] + "-running.jpg", save_plot_dir)
    os.rename(losses_dir[:-4] + "-running.csv", losses_dir)

    return model


def train_pd(model, train_loader, val_loader, epochs=10, num_steps=1, patience = 20, model_dir='.', timing_dir = '.',
            save_plot_dir='.', output_transform=None, losses_dir='.'):
    # num_steps defines the number of times that the denoising model is applied on the final image
    
    start_time = time.time()  # Timing the training time
    # Initializing the optimizer for the model parameters
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    l2_loss = nn.MSELoss()
    threshold = 0.1
    best_val_loss = np.inf
    wait = 0  # for early stopping
    training_losses = []
    val_losses = [] 
    with open(losses_dir[:-4] + "-running.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])
    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        # Training loop
        batch = 0
        for batch_input, batch_target, t in tqdm(train_loader):
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            t = t.to(device)
            optim.zero_grad()  # resetting gradients
            predicted_noise = model(batch_input, t)  # generating images
            batch_output = batch_input - predicted_noise  # simply substracting, might require more advanced scaling in the future
            if output_transform is not None:
                batch_output = output_transform.inverse(batch_output)
                batch_target = output_transform.inverse(batch_target)
            loss = l2_loss(batch_output, batch_target)
            loss.backward()  # backprop
            optim.step()
            train_loss += loss.item()
            batch += 1
            with open(timing_dir, "w") as file:
                file.write(f'epoch {epoch} batch {batch}\n')
                
        # Validation loop
        with torch.no_grad():
            for batch_input, batch_target, t in tqdm(val_loader):
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                t = t.to(device)
                batch_output = batch_input.clone()
                for i in range(num_steps):     
                    batch_noise = model(batch_output, t)  # generating images
                    batch_output = batch_output - batch_noise.clone()
                    t -= 1
                if output_transform is not None:
                    batch_output = output_transform.inverse(batch_output)
                    batch_target = output_transform.inverse(batch_target)
                    
                loss = l2_loss(batch_output, batch_target)
                val_loss += loss.item()

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
    os.rename(losses_dir[:-4] + "-running.csv", losses_dir)
        
    return model
