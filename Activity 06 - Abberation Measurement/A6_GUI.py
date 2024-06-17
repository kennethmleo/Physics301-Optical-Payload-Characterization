import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2


def normalize_smooth_signal(ROI_R, ROI_G, ROI_B):
    profiles = []
    for image_1d in [ROI_R, ROI_G, ROI_B]:
        smoothed_profile = gaussian_filter1d(image_1d, sigma=1)
        min_val = np.min(smoothed_profile)
        max_val = np.max(smoothed_profile)
        normalized_profile = (smoothed_profile - min_val) / (max_val - min_val)

        edge_center = np.argmax(np.abs(np.diff(normalized_profile)))
        profiles.append(normalized_profile)

    x = np.arange(0 - edge_center, len(normalized_profile) - edge_center)
    return x, profiles[0], profiles[1], profiles[2]


def compute_area_between_curves(curve1, curve2, curve3):
    """Compute the area between the highest and lowest curves."""
    area1 = np.trapz(np.abs(curve1 - curve2)) + np.trapz(np.abs(curve2 - curve1))
    area2 = np.trapz(np.abs(curve1 - curve3)) + np.trapz(np.abs(curve3 - curve1))
    area3 = np.trapz(np.abs(curve2 - curve3)) + np.trapz(np.abs(curve3 - curve2))

    max_area = max(area1, area2, area3)
    if max_area == area1:
        return "curve12", max_area
    elif max_area == area2:
        return "curve13", max_area
    elif max_area == area3:
        return "curve23", max_area


def compute_lsf_and_mtf(curve):
    """Compute Line Spread Function (LSF) and Modulation Transfer Function (MTF)."""
    lsf = np.abs(np.diff(curve))
    mtf = np.abs(np.fft.fftshift(np.fft.fft(lsf)))
    mtf = mtf / np.max(mtf)

    n = len(mtf)
    sampling_rate = 1  # Assuming each sample corresponds to one pixel
    freq = np.fft.fftshift(np.fft.fftfreq(n)) * sampling_rate

    return lsf, mtf[len(mtf) // 2:], freq[len(mtf) // 2:]


class ImageAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Analysis Tool")

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.select_image_button = tk.Button(self.root, text="Select Image", command=self.load_image)
        self.select_image_button.pack()

        self.analyze_button = tk.Button(self.root, text="Analyze ROI", command=self.analyze_roi, state=tk.DISABLED)
        self.analyze_button.pack()

        self.canvas = None
        self.image = None
        self.roi_coords = None
        self.roi_patch = None
        self.current_roi = None
        self.image_path = None

    def load_image(self):
        self.image_path = filedialog.askopenfilename(initialdir="./", title="Select Image",
                                                     filetypes=(("Image files", "*.jpg;*.png;*.jpeg"), ("all files", "*.*")))
        if self.image_path:
            self.image = plt.imread(self.image_path)
            self.display_image()
            self.analyze_button.config(state=tk.NORMAL)

    def display_image(self):
        if self.canvas:
            self.canvas.destroy()

        self.current_roi = None
        self.roi_coords = None
        self.roi_patch = None

        img = Image.open(self.image_path)
        img = img.resize((400, 300), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image=img)

        self.canvas = tk.Label(self.root, image=img)
        self.canvas.image = img
        self.canvas.pack()

    def analyze_roi(self):
        if self.roi_coords:
            ROI = self.image[int(self.roi_coords[1]):int(self.roi_coords[1] + self.roi_coords[3]),
                  int(self.roi_coords[0]):int(self.roi_coords[0] + self.roi_coords[2])]

            ROI_R = ROI[:, :, 0]
            ROI_G = ROI[:, :, 1]
            ROI_B = ROI[:, :, 2]

            x, r_norm, g_norm, b_norm = normalize_smooth_signal(ROI_R[len(ROI_R) // 2], ROI_G[len(ROI_G) // 2],
                                                               ROI_B[len(ROI_B) // 2])
            curve1, curve2, curve3 = r_norm, g_norm, b_norm
            which, area = compute_area_between_curves(curve1, curve2, curve3)
            print(f"Area between highest and lowest curves: {area}")

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(x, curve1, color='red', label='Red Edge Profile')
            plt.plot(x, curve2, color='green', label='Green Edge Profile')
            plt.plot(x, curve3, color='blue', label='Blue Edge Profile')
            if which == "curve12":
                plt.fill_between(x, curve1, curve2, where=(curve1 > curve2), interpolate=True, alpha=0.5, color='black')
            elif which == "curve13":
                plt.fill_between(x, curve1, curve3, where=(curve1 > curve3), interpolate=True, alpha=0.5, color='black')
            elif which == "curve23":
                plt.fill_between(x, curve2, curve3, where=(curve2 > curve3), interpolate=True, alpha=0.5, color='black')
            plt.xlabel('Pixels (Horizontal)')
            plt.ylabel('Normalized Edge Profile')
            plt.title(f"Edge profile of RGB channels (CA area = {np.round(area, 2)} pixels)")
            plt.legend()

            plt.subplot(1, 2, 2)
            MTF_values_R = []
            MTF_values_G = []
            MTF_values_B = []
            for j in range(len(ROI_R)):
                ESF_R = ROI_R[j]
                ESF_G = ROI_G[j]
                ESF_B = ROI_B[j]

                MTF_R = compute_lsf_and_mtf(ESF_R)[1]
                MTF_G = compute_lsf_and_mtf(ESF_G)[1]
                MTF_B = compute_lsf_and_mtf(ESF_B)[1]

                MTF_values_R.append(MTF_R)
                MTF_values_G.append(MTF_G)
                MTF_values_B.append(MTF_B)

            MTF_values_R = np.array(MTF_values_R)
            MTF_values_R_median = np.percentile(MTF_values_R, 50, axis=0)
            MTF_values_G = np.array(MTF_values_G)
            MTF_values_G_median = np.percentile(MTF_values_G, 50, axis=0)
            MTF_values_B = np.array(MTF_values_B)
            MTF_values_B_median = np.percentile(MTF_values_B, 50, axis=0)

            plt.plot(np.arange(0, len(MTF_values_R_median)), MTF_values_R_median, color='red', linestyle='solid',
                     label='red MTF')
            plt.plot(np.arange(0, len(MTF_values_R_median)), MTF_values_G_median, color='green', linestyle='solid',
                     label='green MTF')
            plt.plot(np.arange(0, len(MTF_values_R_median)), MTF_values_B_median, color='blue', linestyle='solid',
                     label='blue MTF')
            plt.ylabel('Normalized MTF')
            plt.xlabel('Spatial Frequency')
            plt.legend(fontsize=8)
            plt.tight_layout()

            plt.show()

    def on_click(self, event):
        if self.image is not None:
            if self.roi_patch:
                self.roi_patch.remove()

            if self.roi_coords:
                self.roi_coords = None

            x1, y1 = event.x - 25, event.y - 25
            x2, y2 = event.x + 25, event.y + 25
            self.roi_coords = (x1, y1, x2 - x1, y2 - y1)
            self.roi_patch = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1), linewidth=1, edgecolor='white',
                                               facecolor='none')
            self.canvas.create_window(400, 300, window=self.roi_patch)


# Initialize tkinter app
root = tk.Tk()
app = ImageAnalysisApp(root)
root.mainloop()
