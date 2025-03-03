import os
import matplotlib.pyplot as plt
import torch
import numpy as np


def get_mask_histogram(saliency_values_dict, save_path, y_scaling=True, dataset_mimic=False):

    for method, masks in saliency_values_dict.items():
        if len(masks) > 0: #Exclude empty lists
            plt.figure(figsize=(8, 6))
            saliency_values = torch.cat(masks, dim=0)
            number_samples = saliency_values.shape[0]
            values = saliency_values.flatten().cpu().detach().numpy()

            if y_scaling:
                plt.hist(values, bins=min(1_000, number_samples),
                         alpha=0.8,  color='#56B4E9', weights=np.ones_like(values) / 10e2)
                plt.ylabel('Frequency in $[10^{2}]$', fontsize=16)
            else:
                plt.hist(values, bins=min(1_000, number_samples), alpha=12, color='#56B4E9')
                plt.ylabel('Frequency', fontsize=12)

            plt.xlabel('Absolute gradient value', fontsize=16)

            plt.xlim(0, 1)  # Adjust the range as needed
            if dataset_mimic:
                plt.ylim(0, 10_000)

            # Adjust tick label font sizes
            plt.xticks(fontsize=16)  # Smaller fontsize for x-axis ticks
            plt.yticks(fontsize=16)

            plt.grid(True)
            plt.savefig(os.path.join(save_path,
                                     f"Visualization distribution gradient distribution "
                                     f"{method} samples {number_samples}.pdf"), dpi=300,
                        format="pdf")
            plt.close()