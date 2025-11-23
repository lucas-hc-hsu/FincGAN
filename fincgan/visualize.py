#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from fincgan.logger import get_logger

def auto_plot_figure_3(result_dir = 'results/', save_fig=True, save_dir='./figures/'):
    fig = plt.figure(figsize=(24, 16))
    MARKER_SIZE = 16

    logger = get_logger('auto_plot_figure_3')
    logger.info(f"Searching text files in {result_dir}")

    for file in glob.glob(result_dir + "*.txt", recursive = True):
        print(file)
    logger.info("Search end, Plotting the result...")
    for subplot_idx, target_metric in enumerate(["AUC-PRC(mean)", "AUC-ROC(mean)", "Precision(mean)", "Recall(mean)"]):
        ax = plt.subplot(2, 2, subplot_idx+1)

        plt.xticks(np.arange(0, 1.3, 0.1))
        x_range = np.arange(0.2, 1.3, 0.1)
        x_range = np.insert(x_range, 0, 0)

        gan_range = np.arange(0.1, 1.3, 0.1)
        gan_range = np.insert(gan_range, 0, 0)

        color_map = ['k', 'c', 'r', 'm', 'b', 'g']
        maker_map = ['*-', 's-', 'v-', 'x-', 'o-', '*-']
        for file, color, marker in zip(glob.glob(result_dir + "*.txt", recursive=True), color_map, maker_map):
    #         print(file)
            file_name = file.split(".")[0]
    #         print(file_name)
            setting = file_name.split("_")[-1]
    #         print(setting)

            ''' training result pre-processing '''
            loop_df = pd.read_csv(file)
            loop_result = []

            for ratio in np.arange(0.1, 1.3, 0.1):
                ratio = np.round_(ratio, 1)
                temp_df = loop_df[loop_df['ratio'] == ratio]
                loop_result.append(temp_df.mean(numeric_only=True).tolist()[-6:])

            setting_df = pd.DataFrame(loop_result)
            setting_df.columns = ["AUC-PRC(mean)", "AUC-ROC(mean)", "F-score(mean)", "Precision(mean)", "Recall(mean)", "Acc(mean)"]
            setting_df.to_csv(str(setting)+".csv")
            setting_df.index = setting_df.index + 1
            setting_df.loc[0] = [ 0.4051, 0.8457, 0.4018, 0.5083, 0.3392, 0.9201] # scores of original Amazon musical instrument graph
            setting_df = setting_df.sort_index()

            if setting == 'gan':
                ax.plot(gan_range, setting_df[target_metric], marker, c=color, markersize = MARKER_SIZE, label="FincGAN")
            else:
                ax.plot(gan_range, setting_df[target_metric], marker, c=color, markersize = MARKER_SIZE, label=setting)

        ax.legend(fontsize=12)
        ax.set_title(target_metric, fontsize=16)

    # Ensure save_dir ends with /
    if not save_dir.endswith('/'):
        save_dir += '/'

    fig_path = save_dir + "figure_3.png"
    if save_fig:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.info(f"directory {save_dir} created.")
        plt.savefig(fig_path)
        logger.info(f"Figure saved to {fig_path}")
    logger.info("Program ends.")
    return fig_path

''' Deprecated Implementation Preserved Below '''

# gan = pd.read_csv("gan.csv", index_col=0)
# gan.index = gan.index + 1
# gan.loc[0] = [ 0.4051, 0.8457, 0.4018, 0.5083, 0.3392, 0.9201]
# gan = gan.sort_index()


# noise = pd.read_csv("noise.csv", header=None)
# noise.columns = ["method", "ratio", "AUC-ROC(mean)", "AUC-ROC(std)", "AUC-PRC(mean)", "AUC-PRC(std)", "F-score(mean)", "F-score(std)",
#              "Precision(mean)", "Precision(std)", "Recall(mean)", "Recall(std)", "Acc(mean)", "Acc(std)"]
# noise.index = noise.index + 1
# noise.loc[0] = ["origin", 0, 0.8457, 0, 0.4051, 0, 0.4018, 0, 0.5083, 0, 0.3392, 0, 0.9201, 0]
# noise = noise.sort_index()


# graphsmote = pd.read_csv("graphsmote.csv", header=None)
# graphsmote.columns = ["method", "ratio", "AUC-ROC(mean)", "AUC-ROC(std)", "AUC-PRC(mean)", "AUC-PRC(std)", "F-score(mean)", "F-score(std)",
#              "Precision(mean)", "Precision(std)", "Recall(mean)", "Recall(std)", "Acc(mean)", "Acc(std)"]
# graphsmote.index = graphsmote.index + 1
# graphsmote.loc[0] = ["origin", 0, 0.8457, 0, 0.4051, 0, 0.4018, 0, 0.5083, 0, 0.3392, 0, 0.9201, 0]
# graphsmote = graphsmote.sort_index()


# reweight = pd.read_csv("reweight.csv", header=None)
# reweight.columns = ["method", "ratio", "AUC-ROC(mean)", "AUC-ROC(std)", "AUC-PRC(mean)", "AUC-PRC(std)", "F-score(mean)", "F-score(std)",
#              "Precision(mean)", "Precision(std)", "Recall(mean)", "Recall(std)", "Acc(mean)", "Acc(std)"]
# reweight.index = reweight.index + 1
# reweight.loc[0] = ["origin", 0, 0.8457, 0, 0.4051, 0, 0.4018, 0, 0.5083, 0, 0.3392, 0, 0.9201, 0]
# reweight = reweight.sort_index()


# smote = pd.read_csv("smote.csv", header=None)
# smote.columns = ["method", "ratio", "AUC-ROC(mean)", "AUC-ROC(std)", "AUC-PRC(mean)", "AUC-PRC(std)", "F-score(mean)", "F-score(std)",
#              "Precision(mean)", "Precision(std)", "Recall(mean)", "Recall(std)", "Acc(mean)", "Acc(std)"]
# smote.index = smote.index + 1
# smote.loc[0] = ["origin", 0, 0.8457, 0, 0.4051, 0, 0.4018, 0, 0.5083, 0, 0.3392, 0, 0.9201, 0]
# smote = smote.sort_index()


# oversampling = pd.read_csv("oversampling.csv", header=None)
# oversampling.columns = ["method", "ratio", "AUC-ROC(mean)", "AUC-ROC(std)", "AUC-PRC(mean)", "AUC-PRC(std)", "F-score(mean)", "F-score(std)",
#              "Precision(mean)", "Precision(std)", "Recall(mean)", "Recall(std)", "Acc(mean)", "Acc(std)"]
# oversampling.index = oversampling.index + 1
# oversampling.loc[0] = ["origin", 0, 0.8457, 0, 0.4051, 0, 0.4018, 0, 0.5083, 0, 0.3392, 0, 0.9201, 0]
# oversampling = oversampling.sort_index()


# def plot_figure_3(save_fig=True):
#     fig = plt.figure(figsize=(24, 16))
#     MARKER_SIZE = 16


#     for subplot_idx, target_metric in enumerate(["AUC-PRC(mean)", "AUC-ROC(mean)", "Precision(mean)", "Recall(mean)"]):
#         ax = plt.subplot(2, 2, subplot_idx+1)

#         plt.xticks(np.arange(0, 1.3, 0.1))
#         x_range = np.arange(0.2, 1.3, 0.1)
#         x_range = np.insert(x_range, 0, 0)

#         gan_range = np.arange(0.1, 1.3, 0.1)
#         gan_range = np.insert(gan_range, 0, 0)
#         ax.plot(gan_range, gan[target_metric], '*-', c='k', markersize = MARKER_SIZE, label="FincGAN")
#         ax.plot(x_range, graphsmote[target_metric], 's-', c='c', markersize = MARKER_SIZE, label="GraphSmote")
#         ax.plot(x_range, smote[target_metric], 'v-', c='r', markersize = MARKER_SIZE, label="Smote")
#         ax.plot(x_range, noise[target_metric], 'x-', c='m', markersize = MARKER_SIZE, label="Noise")
#         ax.plot(x_range, oversampling[target_metric], 'o-', c='b', markersize = MARKER_SIZE, label="Oversampling")
#         ax.plot(x_range, reweight[target_metric], '*-', c='g', markersize = MARKER_SIZE, label="Reweight")
#         ax.legend(fontsize=12)
#         ax.set_title(target_metric, fontsize=16)
# #     plt.show()
#     fig_path = "./figures/figure_3.png"
#     if save_fig:
#         if not os.path.exists("./figures/"):
#             os.makedirs("./figures/")
#             logger.info(fdirectory ./figures/ created.)

#     return fig_path


''' code to plot metrices seperately '''

#     gan_range = np.arange(0.1, 1.3, 0.1)
#     gan_range = np.insert(gan_range, 0, 0)
#     plt.plot(gan_range, gan[target_metric], '*-', c='k', markersize = MARKER_SIZE, label="FincGAN")
#     plt.plot(x_range, graphsmote[target_metric], 's-', c='c', markersize = MARKER_SIZE, label="GraphSmote")
#     plt.plot(x_range, smote[target_metric], 'v-', c='r', markersize = MARKER_SIZE, label="Smote")
#     plt.plot(x_range, noise[target_metric], 'x-', c='m', markersize = MARKER_SIZE, label="Noise")
#     plt.plot(x_range, oversampling[target_metric], 'o-', c='b', markersize = MARKER_SIZE, label="Oversampling")
#     plt.plot(x_range, reweight[target_metric], '*-', c='g', markersize = MARKER_SIZE, label="Reweight")
#     plt.legend()
#     plt.title(target_metric, fontsize=16)
#     plt.show()

