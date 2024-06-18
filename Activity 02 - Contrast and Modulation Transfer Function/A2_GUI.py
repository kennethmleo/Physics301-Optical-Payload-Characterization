import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import savgol_filter
import pandas as pd
import cv2

# Function to visualize image in GUI
def visualize_image(path):
    image = plt.imread(path)
    plt.imshow(image, cmap='gray')
    plt.yticks([])
    plt.xticks([])
    plt.title("Image of Interest")
    plt.show()

# Function to visualize ROIs in image
def visualize_image_ROI(path):
    image = plt.imread(path)
    roi_bounds = pd.read_csv(path.replace('01.bmp', 'ROI_bounds.csv'))

    colors = ['red', 'blue', 'yellow', 'green', 'purple']
    plt.imshow(image, cmap='gray')
    for i in roi_bounds.index:
        plt.gca().add_patch(patches.Rectangle((roi_bounds.iloc[i][0], roi_bounds.iloc[i][1]),
                                              roi_bounds.iloc[i][2], roi_bounds.iloc[i][3],
                                              linewidth=1, edgecolor=colors[i], facecolor='none'))
    plt.xticks([])
    plt.yticks([])
    plt.title("Image with pre-selected ROIs")
    plt.show()

# Function to process MTF
def MTF_process(ROI, output, filtered):
    ESF = ROI.mean(axis=0)
    if filtered == 'yes':
        ESF = savgol_filter(ESF, 5, 1)
    LSF = np.abs(np.diff(ESF))

    MTF = np.abs(np.fft.fft(LSF))
    MTF = MTF[:] / np.max(MTF)
    MTF = MTF[:len(MTF) // 2]

    if output == 'ESF':
        return np.array(ESF)
    elif output == 'LSF':
        return np.array(LSF)
    elif output == 'MTF':
        return np.array(MTF)
    
def MTF_process_individual(ROI, output, filtered):
    ESF_vals = []
    LSF_vals = []
    MTF_vals = []
    for i in range(len(ROI)):
        ESF = ROI[i]
        if filtered == 'yes':
            ESF = savgol_filter(ESF, 5, 1)
        LSF = np.abs(np.diff(ESF))
        
        MTF = np.abs(np.fft.fft(LSF))
        MTF = MTF[:]/np.max(MTF)
        MTF = MTF[:len(MTF)//2]

        ESF_vals.append(ESF)
        LSF_vals.append(LSF)
        MTF_vals.append(MTF)

    if output == 'ESF':
        return np.array(ESF_vals)
    if output == 'LSF':
        return np.array(LSF_vals)
    if output == 'MTF':
        return np.array(MTF_vals)
    


# Function to handle ROI selection and MTF plotting
def process_selected_roi(image_path):
    image = plt.imread(image_path)
    ROI_coords = cv2.selectROI("Select ROI", image)
    ROI = image[int(ROI_coords[1]):int(ROI_coords[1] + ROI_coords[3]),
                int(ROI_coords[0]):int(ROI_coords[0] + ROI_coords[2])]

    method = input("Which method to use? (individual/average): ")
    filtered_signal = input("Do you want to filter the line scan? (yes/no): ")

    if method == "average":
        ESF = MTF_process(ROI, 'ESF', filtered_signal)
        LSF = MTF_process(ROI, 'LSF', filtered_signal)
        MTF = MTF_process(ROI, 'MTF', filtered_signal)

        plt.figure(figsize=(18, 6))
        plt.subplot(131)
        plt.plot(ESF, 'k.-')
        plt.ylabel('Slanted Edge Intensity Profile')
        plt.xlabel('Pixel')

        plt.subplot(132)
        plt.plot(LSF, 'k.-')
        plt.ylabel('Line Spread Function')
        plt.xlabel('Pixel')

        plt.subplot(133)
        plt.plot(MTF, 'k.-')
        plt.ylabel('MTF (normalized)')
        plt.xlabel('Spatial Frequency')
        plt.tight_layout()
        plt.show()

    elif method == "individual":
        ESF_v = MTF_process_individual(ROI, 'ESF', filtered_signal)
        LSF_v = MTF_process_individual(ROI, 'LSF', filtered_signal)
        MTF_v = MTF_process_individual(ROI, 'MTF', filtered_signal)

        plt.figure(figsize=(18, 6))
        plt.subplot(131)
        for ESF in ESF_v:
            plt.plot(ESF, 'k.-')
        plt.ylabel('Slanted Edge Intensity Profile')
        plt.xlabel('Pixel')

        plt.subplot(132)
        for LSF in LSF_v:
            plt.plot(LSF, 'k.-')
        plt.ylabel('Line Spread Function')
        plt.xlabel('Pixel')

        plt.subplot(133)
        for MTF in MTF_v:
            plt.plot(MTF, 'k-', linewidth = width)
        plt.plot(np.percentile(MTF_v, 50, axis = 0), color = 'red')
        plt.ylabel('MTF (normalized)')
        plt.xlabel('Spatial Frequency')
        plt.tight_layout()
        plt.show()

# Function to plot MTF of pre-selected ROIs
def plot_MTF_preselected_ROIs(image_path, filtered_signal):
    image = plt.imread(image_path)
    roi_bounds = pd.read_csv(image_path.replace('01.bmp', 'ROI_bounds.csv'))

    plt.figure(figsize=(12, 4))
    colors = ['red', 'blue', 'yellow', 'green', 'purple']
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    for i in roi_bounds.index:
        plt.gca().add_patch(patches.Rectangle((roi_bounds.iloc[i][0], roi_bounds.iloc[i][1]),
                                              roi_bounds.iloc[i][2], roi_bounds.iloc[i][3],
                                              linewidth=1, edgecolor=colors[i], facecolor='none'))
    plt.xticks([])
    plt.yticks([])
    plt.title("Image with pre-selected ROIs")

    plt.subplot(122)
    for i in roi_bounds.index:
        ROI = image[int(roi_bounds.iloc[i][1]):int(roi_bounds.iloc[i][1] + roi_bounds.iloc[i][3]),
                    int(roi_bounds.iloc[i][0]):int(roi_bounds.iloc[i][0] + roi_bounds.iloc[i][2])]

        MTF_vals = MTF_process(ROI, 'MTF', filtered_signal)
        plt.plot(np.arange(0, len(MTF_vals)), MTF_vals, color=colors[i], linewidth=1, linestyle='solid',
                 label='ROI ' + str(i))
    plt.ylabel('Normalized MTF')
    plt.xlabel('Spatial Frequency')
    plt.legend(fontsize=8)
    if filtered_signal == 'yes':
        plt.title('Using filtered edge profile')
    elif filtered_signal == 'no':
        plt.title('Using unfiltered edge profile')
    plt.tight_layout()
    plt.show()

# Function to plot MTF of pre-selected ROIs using individual line scans
def plot_MTF_preselected_ROIs_individual(image_path, filtered_signal):
    image = plt.imread(image_path)
    roi_bounds = pd.read_csv(image_path.replace('01.bmp', 'ROI_bounds.csv'))

    plt.figure(figsize=(12, 4))
    colors = ['red', 'blue', 'yellow', 'green', 'purple']
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    for i in roi_bounds.index:
        plt.gca().add_patch(patches.Rectangle((roi_bounds.iloc[i][0], roi_bounds.iloc[i][1]),
                                              roi_bounds.iloc[i][2], roi_bounds.iloc[i][3],
                                              linewidth=1, edgecolor=colors[i], facecolor='none'))
    plt.xticks([])
    plt.yticks([])
    plt.title("Image with pre-selected ROIs")

    plt.subplot(122)
    for i in roi_bounds.index:
        ROI = image[int(roi_bounds.iloc[i][1]):int(roi_bounds.iloc[i][1] + roi_bounds.iloc[i][3]),
                    int(roi_bounds.iloc[i][0]):int(roi_bounds.iloc[i][0] + roi_bounds.iloc[i][2])]

        MTF_vals = MTF_process_individual(ROI, 'MTF', filtered_signal)
        MTF_values_median = np.percentile(MTF_vals, 50, axis=0)
        plt.plot(np.arange(0, len(MTF_values_median)), MTF_values_median, color=colors[i], linewidth=1,
                 linestyle='solid', label='ROI ' + str(i))
    plt.ylabel('Normalized MTF')
    plt.xlabel('Spatial Frequency')
    plt.legend(fontsize=8)
    if filtered_signal == 'yes':
        plt.title('Using filtered edge profile')
    elif filtered_signal == 'no':
        plt.title('Using unfiltered edge profile')
    plt.tight_layout()
    plt.show()

# Function to handle file selection and initiate visualization
def browse_file():
    filename = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp")])
    if filename:
        visualize_image(filename)

# Function to handle folder selection and initiate ROI visualization
def browse_folder():
    foldername = filedialog.askdirectory()
    if foldername:
        visualize_image_ROI(foldername + '/01.bmp')

# Function to handle plotting MTF of pre-selected ROIs
def plot_preselected_rois():
    filtered_signal = input("Do you want to filter the line scan? (yes/no): ")
    plot_MTF_preselected_ROIs('Visualization/01.bmp', filtered_signal)

# Function to handle plotting MTF of pre-selected ROIs using individual scans
def plot_preselected_rois_individual():
    filtered_signal = input("Do you want to filter the line scan? (yes/no): ")
    plot_MTF_preselected_ROIs_individual('Visualization/01.bmp', filtered_signal)

# Creating the main GUI window
root = tk.Tk()
root.title("MTF Analysis")

# Adding buttons to browse and visualize functionalities
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

browse_file_button = tk.Button(frame, text="Browse Image", command=browse_file)
browse_file_button.pack(side=tk.LEFT, padx=5)

browse_folder_button = tk.Button(frame, text="Visualize ROIs", command=browse_folder)
browse_folder_button.pack(side=tk.LEFT, padx=5)

process_roi_button = tk.Button(frame, text="Select ROI and Process", command=lambda: process_selected_roi('Visualization/01.bmp'))
process_roi_button.pack(side=tk.LEFT, padx=5)

# Adding buttons for additional plot functionalities
plot_preselected_rois_button = tk.Button(frame, text="Plot MTF of Pre-selected ROIs", command=plot_preselected_rois)
plot_preselected_rois_button.pack(side=tk.LEFT, padx=5)

plot_preselected_rois_individual_button = tk.Button(frame, text="Plot MTF of Pre-selected ROIs (Individual)", command=plot_preselected_rois_individual)
plot_preselected_rois_individual_button.pack(side=tk.LEFT, padx=5)

# Displaying the GUI
root.mainloop()
