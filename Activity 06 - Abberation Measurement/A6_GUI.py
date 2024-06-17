import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter1d


# Function definitions from your original code
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
    else:
        return "curve23", max_area


def compute_lsf_and_mtf(curve):
    lsf = np.abs(np.diff(curve))
    mtf = np.abs(np.fft.fftshift(np.fft.fft(lsf)))
    mtf /= np.max(mtf)

    n = len(mtf)
    sampling_rate = 1
    freq = np.fft.fftshift(np.fft.fftfreq(n)) * sampling_rate

    return lsf, mtf[len(mtf) // 2:], freq[len(mtf) // 2:]


# Create the main application window
class ImageAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Analysis Tool")

        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Create widgets
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        self.btn_analyze = tk.Button(self.root, text="Analyze Image", command=self.analyze_image)
        self.btn_analyze.grid(row=1, column=0, pady=10)

        self.btn_show_mtf = tk.Button(self.root, text="Show MTF", command=self.show_mtf)
        self.btn_show_mtf.grid(row=1, column=1, pady=10)

        self.btn_show_ca = tk.Button(self.root, text="Show CA", command=self.show_ca)
        self.btn_show_ca.grid(row=1, column=2, pady=10)

        self.btn_exit = tk.Button(self.root, text="Exit", command=self.root.quit)
        self.btn_exit.grid(row=1, column=3, pady=10)

        # Initialize variables
        self.image_path = None
        self.image = None
        self.ROI_coords = [599, 285, 48, 49]
        self.ROI_R = None
        self.ROI_G = None
        self.ROI_B = None

    def open_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.tif;*.tiff")])
        if self.image_path:
            self.image = cv2.imread(self.image_path)
            self.show_image_with_roi()

    def show_image_with_roi(self):
        if self.image is not None:
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            img_with_roi = img_rgb.copy()
            cv2.rectangle(img_with_roi, (self.ROI_coords[0], self.ROI_coords[1]),
                          (self.ROI_coords[0] + self.ROI_coords[2], self.ROI_coords[1] + self.ROI_coords[3]),
                          (255, 255, 255), 2)

            plt_img = np.squeeze(img_with_roi.astype(np.uint8))
            plt_img = np.transpose(plt_img, (0, 1, 2))

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(plt_img))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        else:
            messagebox.showerror("Error", "No image loaded!")

    def analyze_image(self):
        if self.image is not None:
            ROI = self.image[self.ROI_coords[1]:self.ROI_coords[1] + self.ROI_coords[3],
                             self.ROI_coords[0]:self.ROI_coords[0] + self.ROI_coords[2]]

            self.ROI_R = ROI[:, :, 0]
            self.ROI_G = ROI[:, :, 1]
            self.ROI_B = ROI[:, :, 2]

            x, r_norm, g_norm, b_norm = normalize_smooth_signal(self.ROI_R[len(self.ROI_R) // 2],
                                                                self.ROI_G[len(self.ROI_G) // 2],
                                                                self.ROI_B[len(self.ROI_B) // 2])

            curve1, curve2, curve3 = r_norm, g_norm, b_norm
            which, area = compute_area_between_curves(curve1, curve2, curve3)

            plt.figure(figsize=(8, 6))
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
            plt.show()
        else:
            messagebox.showerror("Error", "No image loaded!")

    def show_mtf(self):
        if self.image is not None:
            ROI = self.image[self.ROI_coords[1]:self.ROI_coords[1] + self.ROI_coords[3],
                             self.ROI_coords[0]:self.ROI_coords[0] + self.ROI_coords[2]]

            self.ROI_R = ROI[:, :, 0]
            self.ROI_G = ROI[:, :, 1]
            self.ROI_B = ROI[:, :, 2]

            x, r_norm, g_norm, b_norm = normalize_smooth_signal(self.ROI_R[len(self.ROI_R) // 2],
                                                                self.ROI_G[len(self.ROI_G) // 2],
                                                                self.ROI_B[len(self.ROI_B) // 2])

            red_lsf, red_mtf, freq = compute_lsf_and_mtf(r_norm)
            green_lsf, green_mtf, freq = compute_lsf_and_mtf(g_norm)
            blue_lsf, blue_mtf, freq = compute_lsf_and_mtf(b_norm)

            plt.figure(figsize=(8, 6))
            plt.plot(freq, red_mtf, color='red', label='Red MTF')
            plt.plot(freq, green_mtf, color='green', label='Green MTF')
            plt.plot(freq, blue_mtf, color='blue', label='Blue MTF')
            plt.title('Modulation Transfer Function (MTF)')
            plt.xlabel('Spatial Frequency (cycles per pixel)')
            plt.ylabel('Normalized MTF')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            messagebox.showerror("Error", "No image loaded!")

    def show_ca(self):
        if self.image is not None:
            ROI = self.image[self.ROI_coords[1]:self.ROI_coords[1] + self.ROI_coords[3],
                             self.ROI_coords[0]:self.ROI_coords[0] + self.ROI_coords[2]]

            self.ROI_R = ROI[:, :, 0]
            self.ROI_G = ROI[:, :, 1]
            self.ROI_B = ROI[:, :, 2]

            # Calculate chromatic aberration
            ca_r = self.ROI_G.astype(np.float32) - self.ROI_R.astype(np.float32)
            ca_b = self.ROI_G.astype(np.float32) - self.ROI_B.astype(np.float32)

            # Display the chromatic aberration
            plt.figure(figsize=(10, 8))

            plt.subplot(2, 1, 1)
            plt.imshow(ca_r, cmap='RdYlBu', vmin=-50, vmax=50)
            plt.colorbar()
            plt.title('Chromatic Aberration (Red-Green)')
            plt.xlabel('Pixel Column')
            plt.ylabel('Pixel Row')

            plt.subplot(2, 1, 2)
            plt.imshow(ca_b, cmap='RdYlBu', vmin=-50, vmax=50)
            plt.colorbar()
            plt.title('Chromatic Aberration (Blue-Green)')
            plt.xlabel('Pixel Column')
            plt.ylabel('Pixel Row')

            plt.tight_layout()
            plt.show()
        else:
            messagebox.showerror("Error", "No image loaded!")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalysisApp(root)
    root.mainloop()
