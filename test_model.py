
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import pymed_gamma, psnr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.serif'] = 'Times New Roman'

def test(trained_model, test_loader, device, results_dir='.', output_transform=None, save_plot_dir=None, plot_type="gamma", deviations=False, mm_per_voxel=(1.9531, 1.9531, 1.5)):
    # plot_type can be "gamma" or "deviations", will plot the histogram of these values in the test set
    
    # Test loop (after the training is complete)
    time_list = []
    l2_loss_list = []
    l2_loss = nn.MSELoss()
    mre_loss_list = []  # Mean relative error loss
    gamma_pymed_list_1 = []  # For 1% tolerance
    gamma_value_pymed_list_1 = []
    gamma_pymed_list_3 = []  # For 3% tolerance
    gamma_value_pymed_list_3 = []
    psnr_list = []
    threshold = 0.1  # Minimum relative dose considered for gamma index
    random_subset = 1000  # Number of random voxels to calculate gamma index
    
    if deviations:
        x_deviation_error_list = []  
        y_deviation_error_list = []  
        psi_deviation_error_list = []  
        # Comparing with actual deviation size
        x_deviation_list = []  
        y_deviation_list = []  
        psi_deviation_list = []  
        
    with torch.no_grad():
        for batch_input, batch_target, deviations_target in tqdm(test_loader):
            batch_input = batch_input.to(device)
            start_time = time.time()
            if deviations:
                
                batch_output, deviations_output = trained_model(batch_input, return_bottleneck=True)
                deviations_output = deviations_output.detach().cpu()
                print(deviations_output, deviations_target)
                x_deviation_error_list.append(torch.abs(deviations_output[:, 0] - deviations_target[:, 0]))
                y_deviation_error_list.append(torch.abs(deviations_output[:, 1] - deviations_target[:, 1]))
                psi_deviation_error_list.append(torch.abs(deviations_output[:, 2] - deviations_target[:, 2]))
                x_deviation_list.append(torch.abs(deviations_target[:, 0]))
                y_deviation_list.append(torch.abs(deviations_target[:, 1]))
                psi_deviation_list.append(torch.abs(deviations_target[:, 2]))
                
            else:
                batch_output = trained_model(batch_input)  # generating images
            time_list.append((time.time() - start_time) * 1000)
            if output_transform is not None:
                batch_output = output_transform.inverse(batch_output)
                batch_target = output_transform.inverse(batch_target)
            batch_output = batch_output.detach().cpu()
            torch.cuda.empty_cache()
            
            l2_loss_list.append(l2_loss(batch_output, batch_target))
            # With 1% tolerance
            tolerance = 0.01  # Tolerance per unit for gamma index
            distance_mm_threshold = 1  # Distance in mm for gamma index
            pymed_gamma_index, gamma_value = pymed_gamma(batch_output, batch_target, mm_per_voxel=mm_per_voxel, dose_percent_threshold=tolerance*100, 
                                           distance_mm_threshold=distance_mm_threshold, threshold=threshold, random_subset=random_subset)
            gamma_pymed_list_1.append(pymed_gamma_index)
            gamma_value_pymed_list_1.append(gamma_value)
            # With 3% tolerance
            tolerance = 0.03  # Tolerance per unit for gamma index
            distance_mm_threshold = 3  # Distance in mm for gamma index
            pymed_gamma_index, gamma_value = pymed_gamma(batch_output, batch_target, mm_per_voxel=mm_per_voxel, dose_percent_threshold=tolerance*100,
                                           distance_mm_threshold=distance_mm_threshold, threshold=threshold, random_subset=random_subset)
            gamma_pymed_list_3.append(pymed_gamma_index)
            gamma_value_pymed_list_3.append(gamma_value)
            
            psnr_list.append(psnr(batch_output, batch_target))
            
            # MRE
            for image_output, image_target in zip(batch_output, batch_target):
                max_val = image_target.max()
                mre_loss_list.append(torch.mean(torch.abs(image_output[image_target > 0.01 * max_val] - image_target[image_target > 0.01 * max_val]) / max_val * 100))
            
    l2_loss_list = torch.tensor(l2_loss_list)
    mre_loss_list = torch.tensor(mre_loss_list)
    # gamma_list = torch.tensor(gamma_list)
    gamma_pymed_list_1 = torch.tensor(gamma_pymed_list_1)
    gamma_value_pymed_list_1 = torch.cat(gamma_value_pymed_list_1)
    gamma_pymed_list_3 = torch.tensor(gamma_pymed_list_3)
    gamma_value_pymed_list_3 = torch.cat(gamma_value_pymed_list_3)
    fraction_below_90 = torch.sum(gamma_pymed_list_3 < 0.9).item() / len(gamma_pymed_list_3)
    psnr_list = torch.cat(psnr_list)
    
    text_results = f"L2 Loss (Gy²): {torch.mean(l2_loss_list)} +- {torch.std(l2_loss_list)}\n" \
           f"Mean Relative Error (%): {torch.mean(mre_loss_list)} +- {torch.std(mre_loss_list)}\n" \
           f"Peak Signal-to-Noise Ratio: {torch.mean(psnr_list)} +- {torch.std(psnr_list)}\n" \
           f"Pymed gamma value (1mm, 1%): {torch.mean(gamma_value_pymed_list_1)} +- {torch.std(gamma_value_pymed_list_1)}\n" \
           f"Pymed gamma index (1mm, 1%): {torch.mean(gamma_pymed_list_1)} +- {torch.std(gamma_pymed_list_1)}\n" \
           f"Pymed gamma value (3mm, 3%): {torch.mean(gamma_value_pymed_list_3)} +- {torch.std(gamma_value_pymed_list_3)}\n" \
           f"Pymed gamma index (3mm, 3%): {torch.mean(gamma_pymed_list_3)} +- {torch.std(gamma_pymed_list_3)}\n" \
           f"Fraction of gamma values below 0.9: {fraction_below_90}\n\n" \
           f"Time per loading (ms): {np.mean(np.array(time_list))} +- {np.std(np.array(time_list))}"
    
    if deviations:
        x_deviation_error_list = torch.cat(x_deviation_error_list)
        y_deviation_error_list = torch.cat(y_deviation_error_list)
        psi_deviation_error_list = torch.cat(psi_deviation_error_list)
        x_deviation_list = torch.cat(x_deviation_list)
        y_deviation_list = torch.cat(y_deviation_list)
        psi_deviation_list = torch.cat(psi_deviation_list)
        text_results += f"\n\n x deviation error (mm), maximum deviation is +- 5mm: {torch.mean(x_deviation_error_list)} +- {torch.std(x_deviation_error_list)}\n" \
                        f"y deviation error (mm), maximum deviation is +- 5mm: {torch.mean(y_deviation_error_list)} +- {torch.std(y_deviation_error_list)}\n" \
                        f"Mean deviation error psi (degrees), maximum deviation is +- 5º: {torch.mean(psi_deviation_error_list)} +- {torch.std(psi_deviation_error_list)}\n"
   
    print(text_results)

    # Save to file
    with open(results_dir, "w") as file:
        file.write(text_results)

    if save_plot_dir is not None:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        if plot_type == "gamma":
            # Plot the histogram of gamma indices
            bin_width = 0.005
            font_size = 22
            plt.figure(figsize=(10, 6))# Plot first list
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            sns.histplot(gamma_pymed_list_3.cpu().numpy().flatten(), color="red", alpha=0.9, label='AI-generated vs Target Simulated Dose SOBPs',
                binwidth=bin_width, hatch='/', edgecolor='black'
            )
            plt.title('Gamma Index Histogram (tolerance=0.03, distance=1mm)', fontsize=font_size)
            plt.ylabel('Counts', fontsize=font_size)
            plt.xlabel('Gamma Index', fontsize=font_size)
            plt.legend(fontsize=font_size - 2)
            plt.tight_layout()
            if save_plot_dir is not None:   
                plt.savefig(save_plot_dir)
                plt.savefig(save_plot_dir[:-3] + "eps", format='eps')   
        elif plot_type == "deviations" and deviations:
            all_deviations_error = torch.cat([x_deviation_error_list, y_deviation_error_list, psi_deviation_error_list])
            all_deviations = torch.cat([x_deviation_list, y_deviation_list, psi_deviation_list])
            # Plot the histogram of gamma indices
            bin_width = 0.6
            font_size = 28
            plt.figure(figsize=(10, 6))# Plot first list
            plt.xticks([0.00, 0.05, 0.10, 0.15, 0.20], fontsize=font_size)
            plt.yticks(fontsize=font_size)
            # sns.histplot(all_deviations.cpu().numpy().flatten(), color="red", alpha=0.6, label='|Actual Positioning Deviations|',
            #     binwidth=bin_width, edgecolor='black'
            # )
            bin_width = 0.05
            sns.histplot(all_deviations_error.cpu().numpy().flatten(), color="blue", alpha=0.7, label='|Estimated - Actual Set-Up Deviations|',
                binwidth=bin_width, edgecolor='black'
            )
            plt.ylabel('Counts', fontsize=font_size)
            plt.xlabel('Set-up Deviations (º or mm)', fontsize=font_size)
            plt.legend(fontsize=font_size - 2)
            plt.tight_layout()
            if save_plot_dir is not None:   
                plt.savefig(save_plot_dir)
                plt.savefig(save_plot_dir[:-3] + "eps", format='eps')   
            
    return None
