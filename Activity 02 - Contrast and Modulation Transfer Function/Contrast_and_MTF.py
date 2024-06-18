import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import simpledialog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import savgol_filter
import pandas as pd
import cv2
from PIL import Image, ImageTk


# Define global styles and paths
background_color = "#282C34"
button_color = "#61DAFB"
text_color = "white"
font_style = "Helvetica"
default_data_path = "Visualization/"
default_image_path = os.path.join(default_data_path, "target_image.png")
image_path = "Visualization/01.bmp"

def load_data():
    """Step 1: Load data folders from the specified path and update the folder combobox."""
    global data_path, folders
    data_path = data_path_entry.get()
    
    if not data_path:
        messagebox.showerror("Error", "Please select a data folder.")
        return
    
    folders = [f for f in os.listdir(data_path) if not f.startswith('.')]
    folders.sort()
    
    # Update the combobox with the folders
    folder_combobox['values'] = folders
    folder_combobox.set('Select a folder')
    
    messagebox.showinfo("Info", f"Loaded {len(folders)} folders successfully.")

# Function to display the image in the GUI
def display_image(path):
    try:
        img = Image.open(path)
        img = img.resize((800, 600), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        panel.configure(image=img_tk)
        panel.image = img_tk  # Keep a reference to avoid garbage collection
    except FileNotFoundError:
        messagebox.showerror("Error", f"File '{path}' not found.")

# Function to handle file selection
def browse_file():
    filename = filedialog.askopenfilename(initialdir=default_data_path, filetypes=[("Image files", "*.bmp;*.png;*.jpg"), ("All files", "*.*")])
    if filename:
        display_image(filename)

# Function to visualize and process the image for MTF analysis
# def process_selected_roi(image_path):
#     # Placeholder for image processing logic
#     messagebox.showinfo("Info", "Process the selected ROI from the image.")


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

 # This function will now get options directly from the GUI components instead of input() 
    method = method_var.get()
    filtered_signal = filter_var.get()
    width = 0.1

    if method == "average":
        ESF = MTF_process(ROI, 'ESF', filtered_signal)
        LSF = MTF_process(ROI, 'LSF', filtered_signal)
        MTF = MTF_process(ROI, 'MTF', filtered_signal)

        plt.figure(figsize=(18, 6))
        plt.subplot(131)
        plt.plot(ESF, 'k.-', )
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
            plt.plot(ESF, 'k.-', linewidth = width)
        plt.ylabel('Slanted Edge Intensity Profile')
        plt.xlabel('Pixel')

        plt.subplot(132)
        for LSF in LSF_v:
            plt.plot(LSF, 'k.-', linewidth = width)
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

# # Function to handle plotting MTF of pre-selected ROIs
# def plot_preselected_rois():
#     filtered_signal = input("Do you want to filter the line scan? (yes/no): ")
#     plot_MTF_preselected_ROIs('Visualization/01.bmp', filtered_signal)

# # Function to handle plotting MTF of pre-selected ROIs using individual scans
# def plot_preselected_rois_individual():
#     filtered_signal = input("Do you want to filter the line scan? (yes/no): ")
#     plot_MTF_preselected_ROIs_individual('Visualization/01.bmp', filtered_signal)

def plot_MTF_preselected_ROIs_2(image_path):
    method = method_var.get()
    filtered_signal = filter_var.get()
    if method == "average":
        # Assume plot_MTF_average is a function that handles the "average" MTF calculation
        plot_MTF_preselected_ROIs(image_path, filtered_signal)
    else:
        messagebox.showerror("Error", "Unsupported method selected")

def plot_MTF_preselected_ROIs_individual_2(image_path):
    method = method_var.get()
    filtered_signal = filter_var.get()
    if method == "individual":
        # Assume plot_MTF_individual is a function that handles the "individual" MTF calculation
        plot_MTF_preselected_ROIs_individual(image_path, filtered_signal)
    else:
        messagebox.showerror("Error", "Unsupported method selected")

    
def upload_image():
    """Upload and display an image."""
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((800, 600), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img

def display_default_image():
    """Display the default image on the GUI."""
    try:
        img = Image.open(default_image_path)
        img = img.resize((800, 600), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        img_label = tk.Label(right_frame, image=img_tk)
        img_label.image = img_tk  # Keep a reference to avoid garbage collection
        img_label.pack(fill="both", expand=True)
    except FileNotFoundError:
        messagebox.showerror("Error", f"File '{default_image_path}' not found.")


# Initialize the main window
root = tk.Tk()
root.title("MTF Analysis Tool")
root.geometry("1280x1024")
root.configure(bg="#4D00A9")

# Create a frame for buttons on the left
left_frame = tk.Frame(root, bg="#282C34")
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=20)

# Create a frame for displaying the image on the right
right_frame = tk.Frame(root, bg="#282C34")
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=20)

# Function to display the default image
default_image_path = "target_image.png"

def display_default_image():
    try:
        img = Image.open(default_image_path)
        img = img.resize((800, 600), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        img_label = tk.Label(right_frame, image=img_tk)
        img_label.image = img_tk  # Keep a reference to avoid garbage collection
        img_label.pack(fill="both", expand=True)
    except FileNotFoundError:
        messagebox.showerror("Error", f"File '{default_image_path}' not found.")

# Add the title
tk.Label(left_frame, text="MTF Analysis Tool", font=("Helvetica", 20, "bold"), bg="#282C34", fg="#61DAFB").pack(pady=20)

# Step 1: Browse and Display Image
# Add the steps
data_path_entry = tk.Entry(left_frame, width=50)
data_path_entry.insert(0, "")

tk.Label(left_frame, text="Step 1: Browse and Display Image", font=(font_style, 16, "bold"), bg=background_color, fg=text_color).pack(fill='x', padx=10, pady=10)
data_path_entry.pack(pady=5, padx=10)

browse_button = tk.Button(left_frame, text="Browse Image", command=browse_file, bg=button_color, fg=background_color, font=(font_style, 14))
browse_button.pack(pady=5)

tk.Button(left_frame, text="Load Data", command=load_data, bg="#61DAFB", fg="#282C34", font=("Helvetica", 14)).pack(pady=10, padx=10)
tk.Label(left_frame, text="Loads data from the selected folder and lists available subfolders.", font=("Helvetica", 12), bg="#282C34", fg="white", anchor="w").pack(fill='x', padx=10)


# Step 2 and Step 3 placeholders with descriptions
tk.Label(left_frame, text="Step 2: Analyze MTF for Pre-Selected ROIs", font=(font_style, 16, "bold"), bg=background_color, fg=text_color).pack(fill='x', padx=10, pady=10)
tk.Button(left_frame, text="Average Line Scan", command=lambda: plot_MTF_preselected_ROIs_2(image_path), bg=button_color, fg=background_color, font=(font_style, 14)).pack(pady=5)
tk.Button(left_frame, text="Individual Line Scans", command=lambda: plot_MTF_preselected_ROIs_individual_2(image_path), bg=button_color, fg=background_color, font=(font_style, 14)).pack(pady=5)

# tk.Button(left_frame, text="Average of line scan", command=plot_preselected_rois, bg=button_color, fg=background_color, font=(font_style, 14)).pack(pady=5)
# tk.Button(left_frame, text="Individual line scans", command=plot_preselected_rois_individual, bg=button_color, fg=background_color, font=(font_style, 14)).pack(pady=5)

tk.Label(left_frame, text="Step 3: MTF for a Selected ROI", font=(font_style, 16, "bold"), bg=background_color, fg=text_color).pack(fill='x', padx=10, pady=10)

# GUI Elements for method selection
method_var = tk.StringVar(value="average")  # Default value set to "average"
tk.Label(left_frame, text="Choose Method:", font=(font_style, 14), bg=background_color, fg=text_color).pack(padx=10, pady=10)
tk.Radiobutton(left_frame, text="Individual", variable=method_var, value="individual", bg=background_color, fg=text_color, selectcolor=background_color).pack(anchor='w', padx=20)
tk.Radiobutton(left_frame, text="Average", variable=method_var, value="average", bg=background_color, fg=text_color, selectcolor=background_color).pack(anchor='w', padx=20)


# GUI Element for filter option
filter_var = tk.StringVar(value="no")  # Default value set to "no"
tk.Label(left_frame, text="Filter Line Scan:", font=(font_style, 14), bg=background_color, fg=text_color).pack(padx=10, pady=10)
tk.Checkbutton(left_frame, text="Yes", variable=filter_var, onvalue="yes", offvalue="no", bg=background_color, fg=text_color, selectcolor=background_color).pack(anchor='w', padx=20)

# Button to execute the process with the selected options
#tk.Button(left_frame, text="Process MTF", command=lambda: process_selected_roi(default_image_path), bg=button_color, fg=background_color, font=(font_style, 14)).pack(pady=20)
tk.Button(left_frame, text="Select ROI and Process", command=lambda: process_selected_roi(default_image_path), bg=button_color, fg=background_color, font=(font_style, 14)).pack(pady=5)


# Display the default image
display_default_image()

# Run the GUI
root.mainloop()