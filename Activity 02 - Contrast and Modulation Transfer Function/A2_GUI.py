import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import savgol_filter
import pandas as pd
import cv2
from tkinter import Tk, Button, Label, filedialog, simpledialog, StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MTFAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("MTF Analyzer")
        self.root.geometry("800x600")

        self.image_path = None
        self.image = None
        self.roi_bounds = None
        self.filtered_signal = StringVar(value="no")
        
        # Create buttons
        self.load_image_button = Button(root, text="Load Image", command=self.load_image)
        self.load_image_button.pack()

        self.load_roi_button = Button(root, text="Load ROI Bounds", command=self.load_roi_bounds)
        self.load_roi_button.pack()

        self.visualize_button = Button(root, text="Visualize Image", command=self.visualize_image)
        self.visualize_button.pack()

        self.visualize_roi_button = Button(root, text="Visualize Image with ROIs", command=self.visualize_image_ROI)
        self.visualize_roi_button.pack()

        self.filter_button = Button(root, text="Filter Line Scan", command=self.set_filter)
        self.filter_button.pack()

        self.mtf_button = Button(root, text="Visualize MTF", command=self.visualize_mtf)
        self.mtf_button.pack()

        self.select_roi_button = Button(root, text="Select ROI", command=self.select_roi)
        self.select_roi_button.pack()

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        self.image = plt.imread(self.image_path)
        self.show_message("Image loaded successfully.")

    def load_roi_bounds(self):
        roi_bounds_path = filedialog.askopenfilename()
        self.roi_bounds = pd.read_csv(roi_bounds_path)
        self.show_message("ROI bounds loaded successfully.")

    def visualize_image(self):
        if self.image is not None:
            plt.imshow(self.image, cmap='gray')
            plt.yticks([])
            plt.xticks([])
            plt.title("Image of Interest")
            self.show_plot()
        else:
            self.show_message("Please load an image first.")

    def visualize_image_ROI(self):
        if self.image is not None and self.roi_bounds is not None:
            colors = ['red', 'blue', 'yellow', 'green', 'purple']
            plt.imshow(self.image, cmap='gray')
            for i in self.roi_bounds.index:
                plt.gca().add_patch(patches.Rectangle((self.roi_bounds.iloc[i][0], self.roi_bounds.iloc[i][1]),
                                                      self.roi_bounds.iloc[i][2], self.roi_bounds.iloc[i][3],
                                                      linewidth=1, edgecolor=colors[i], facecolor='none'))
            plt.xticks([])
            plt.yticks([])
            plt.title("Image with Pre-selected ROIs")
            self.show_plot()
        else:
            self.show_message("Please load both image and ROI bounds.")

    def set_filter(self):
        self.filtered_signal.set(simpledialog.askstring("Input", "Do you want to filter the line scan? (yes/no)"))

    def visualize_mtf(self):
        if self.image is not None and self.roi_bounds is not None:
            colors = ['red', 'blue', 'yellow', 'green', 'purple']
            plt.figure(figsize=(9, 3), dpi=300)
            plt.subplot(121)
            plt.imshow(self.image, cmap='gray')
            for i in self.roi_bounds.index:
                plt.gca().add_patch(patches.Rectangle((self.roi_bounds.iloc[i][0], self.roi_bounds.iloc[i][1]),
                                                      self.roi_bounds.iloc[i][2], self.roi_bounds.iloc[i][3],
                                                      linewidth=1, edgecolor=colors[i], facecolor='none'))
            plt.xticks([])
            plt.yticks([])

            plt.subplot(122)
            for i in self.roi_bounds.index:
                ROI = self.image[int(self.roi_bounds.iloc[i][1]):int(self.roi_bounds.iloc[i][1] + self.roi_bounds.iloc[i][3]), 
                                int(self.roi_bounds.iloc[i][0]):int(self.roi_bounds.iloc[i][0] + self.roi_bounds.iloc[i][2])]
                
                MTF_vals = self.MTF_process(ROI, 'MTF', self.filtered_signal.get())
                
                plt.plot(np.arange(0, len(MTF_vals)), MTF_vals, color=colors[i], linewidth=1, linestyle='solid', label='ROI ' + str(i))
                plt.ylabel('Normalized MTF')
                plt.xlabel('Spatial Frequency')
                plt.legend(fontsize=5)
                if self.filtered_signal.get() == 'yes': plt.title('Using filtered edge profile')
                if self.filtered_signal.get() == 'no': plt.title('Using unfiltered edge profile')
            self.show_plot()
        else:
            self.show_message("Please load both image and ROI bounds.")

    def select_roi(self):
        if self.image is not None:
            roi_coords = cv2.selectROI(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            ROI = self.image[int(roi_coords[1]):int(roi_coords[1] + roi_coords[3]), int(roi_coords[0]):int(roi_coords[0] + roi_coords[2])]
            plt.imshow(ROI, cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.show()

            method = simpledialog.askstring("Input", "Which method to use? (individual/average)")
            self.filtered_signal.set(simpledialog.askstring("Input", "Do you want to filter the line scan? (yes/no)"))

            if method == "average":
                self.visualize_mtf_roi_average(ROI)
            elif method == "individual":
                self.visualize_mtf_roi_individual(ROI)
        else:
            self.show_message("Please load an image first.")

    def MTF_process(self, ROI, output, filtered):
        ESF = ROI.mean(axis=0)
        if filtered == 'yes':
            ESF = savgol_filter(ESF, 5, 1)
        LSF = np.abs(np.diff(ESF))
        
        MTF = np.abs(np.fft.fft(LSF))
        MTF = MTF[:]/np.max(MTF)
        MTF = MTF[:len(MTF)//2]
        
        if output == 'ESF': return np.array(ESF)
        if output == 'LSF': return np.array(LSF)
        if output == 'MTF': return np.array(MTF)

    def MTF_process_individual(self, ROI, output, filtered):
        ESF_vals, LSF_vals, MTF_vals = [], [], []
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
        
        if output == 'ESF': return np.array(ESF_vals)
        if output == 'LSF': return np.array(LSF_vals)
        if output == 'MTF': return np.array(MTF_vals)

    def visualize_mtf_roi_average(self, ROI):
        ESF = self.MTF_process(ROI, 'ESF', self.filtered_signal.get())
        LSF = self.MTF_process(ROI, 'LSF', self.filtered_signal.get())
        MTF = self.MTF_process(ROI, 'MTF', self.filtered_signal.get())
        
        plt.figure(figsize=(18, 3), dpi=150)
        plt.subplot(131)
        plt.plot(ESF, 'k.-', linewidth=0.5)
        plt.ylabel('Slanted Edge Intensity Profile')
        plt.xlabel('Pixel')
        
        plt.subplot(132)
        plt.plot(LSF, 'k.-', linewidth=0.5)
        plt.ylabel('Line Spread Function')
        plt.xlabel('Pixel')
        
        plt.subplot(133)
        plt.plot(MTF, 'k.-', linewidth=0.5)
        plt.ylabel('MTF (normalized)')
        plt.xlabel('Spatial Frequency')
        plt.show()

    def visualize_mtf_roi_individual(self, ROI):
        ESF_v = self.MTF_process_individual(ROI, 'ESF', self.filtered_signal.get())
        LSF_v = self.MTF_process_individual(ROI, 'LSF', self.filtered_signal.get())
        MTF_v = self.MTF_process_individual(ROI, 'MTF', self.filtered_signal.get())
        
        plt.figure(figsize=(18, 3), dpi=150)
        plt.subplot(131)
        for ESF in ESF_v:
            plt.plot(ESF, 'k.-', linewidth=0.5)
        plt.ylabel('Slanted Edge Intensity Profile')
        plt.xlabel('Pixel')
        
        plt.subplot(132)
        for LSF in LSF_v:
            plt.plot(LSF, 'k.-', linewidth=0.5)
        plt.ylabel('Line Spread Function')
        plt.xlabel('Pixel')
        
        plt.subplot(133)
        for MTF in MTF_v:
            plt.plot(MTF, 'k.-', linewidth=0.5)
        plt.ylabel('MTF (normalized)')
        plt.xlabel('Spatial Frequency')
        plt.show()

    def show_plot(self):
        fig = plt.gcf()
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def show_message(self, message):
        message_label = Label(self.root, text=message)
        message_label.pack()
        
if __name__ == "__main__":
    root = Tk()
    app = MTFAnalyzer(root)
    root.mainloop()
