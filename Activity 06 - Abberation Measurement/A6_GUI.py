import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from tkinter import *
from tkinter import filedialog, messagebox
import cv2

class ImageAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Analysis App")

        self.image_path = None
        self.image = None
        self.ROI_coords = None
        self.ROI = None
        self.ROI_R = None
        self.ROI_G = None
        self.ROI_B = None

        self.create_widgets()

    def create_widgets(self):
        self.btn_open_image = Button(self.master, text="Open Image", command=self.open_image)
        self.btn_open_image.pack(pady=10)

        self.btn_analyze_image = Button(self.master, text="Analyze Image", command=self.analyze_image)
        self.btn_analyze_image.pack(pady=5)

        self.btn_show_mtf = Button(self.master, text="Show MTF", command=self.show_mtf)
        self.btn_show_mtf.pack(pady=5)

        self.btn_show_ca = Button(self.master, text="Show Chromatic Aberration", command=self.show_chromatic_aberration)
        self.btn_show_ca.pack(pady=5)

        self.btn_select_roi = Button(self.master, text="Select ROI", command=self.select_roi)
        self.btn_select_roi.pack(pady=5)

        self.btn_quit = Button(self.master, text="Quit", command=self.master.quit)
        self.btn_quit.pack(pady=10)

    def open_image(self):
        self.image_path = filedialog.askopenfilename(initialdir="/", title="Select Image File",
                                                     filetypes=(("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"), ("all files", "*.*")))
        if self.image_path:
            self.image = plt.imread(self.image_path)
            self.show_image_with_roi()

    def show_image_with_roi(self):
        plt.figure(figsize=(6, 4))
        plt.imshow(self.image)
        if self.ROI_coords:
            plt.gca().add_patch(patches.Rectangle((self.ROI_coords[0], self.ROI_coords[1]), self.ROI_coords[2], self.ROI_coords[3],
                                                  linewidth=1, edgecolor='white', facecolor='none'))
        plt.title("Image with ROI")
        plt.axis('off')
        plt.show()

    def analyze_image(self):
        if self.image is not None and self.ROI_coords is not None:
            ROI = self.image[int(self.ROI_coords[1]):int(self.ROI_coords[1] + self.ROI_coords[3]),
                  int(self.ROI_coords[0]):int(self.ROI_coords[0] + self.ROI_coords[2])]
            self.ROI_R = ROI[:, :, 0]
            self.ROI_G = ROI[:, :, 1]
            self.ROI_B = ROI[:, :, 2]

            # Visualize normalized edge profile and chromatic aberration
            x, r_norm, g_norm, b_norm = self.normalize_smooth_signal(self.ROI_R[len(self.ROI_R) // 2],
                                                                     self.ROI_G[len(self.ROI_G) // 2],
                                                                     self.ROI_B[len(self.ROI_B) // 2])
            curve1, curve2, curve3 = r_norm, g_norm, b_norm
            which, area = self.compute_area_between_curves(curve1, curve2, curve3)
            #self.show_normalized_edge_profile(x, curve1, curve2, curve3, area)
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
            messagebox.showerror("Error", "No image loaded or ROI not selected!")
            
    def show_mtf(self):
        if self.image is not None and self.ROI_coords is not None:
            mtf_vals = []
            for i in range(len(self.ROI_R)):
                ROI_SFR_R = self.ROI_R[i]
                ROI_SFR_G = self.ROI_G[i]
                ROI_SFR_B = self.ROI_B[i]

                x, r_norm, g_norm, b_norm = self.normalize_smooth_signal(ROI_SFR_R, ROI_SFR_G, ROI_SFR_B)
                red_lsf, red_mtf, freq = self.compute_lsf_and_mtf(r_norm)
                green_lsf, green_mtf, freq = self.compute_lsf_and_mtf(g_norm)
                blue_lsf, blue_mtf, freq = self.compute_lsf_and_mtf(b_norm)

                mtf_vals.append([red_mtf, green_mtf, blue_mtf, freq])
            mtf_vals = np.array(mtf_vals)

            mtf_r_median = np.percentile(mtf_vals[:, 0], 50, axis=0)
            mtf_g_median = np.percentile(mtf_vals[:, 1], 50, axis=0)
            mtf_b_median = np.percentile(mtf_vals[:, 2], 50, axis=0)
            freq = mtf_vals[:, 3].mean(axis=0)

            mtf_r_30, mtf_r_70 = np.interp(0.3, mtf_r_median[::-1], freq[::-1]), np.interp(0.7, mtf_r_median[::-1], freq[::-1])
            mtf_g_30, mtf_g_70 = np.interp(0.3, mtf_g_median[::-1], freq[::-1]), np.interp(0.7, mtf_g_median[::-1], freq[::-1])
            mtf_b_30, mtf_b_70 = np.interp(0.3, mtf_b_median[::-1], freq[::-1]), np.interp(0.7, mtf_b_median[::-1], freq[::-1])

            plt.figure(figsize=(8, 6))
            plt.plot(freq, mtf_r_median, color = 'red', linestyle = 'solid', 
                    label = f'red MTF (30% - {np.round(mtf_r_30,2)}, 70% - {np.round(mtf_r_70,2)})')
            plt.plot(freq, mtf_g_median, color = 'green', linestyle = 'solid', 
                    label = f'green MTF (30% - {np.round(mtf_g_30,2)}, 70% - {np.round(mtf_g_70,2)})')
            plt.plot(freq, mtf_b_median, color = 'blue', linestyle = 'solid', 
                    label = f'blue MTF (30% - {np.round(mtf_b_30,2)}, 70% - {np.round(mtf_b_70,2)})')
            plt.axhline(y = 0.3, color = 'black', linestyle = 'dashed')
            plt.axhline(y = 0.7, color = 'black', linestyle = 'dashed')
            plt.title('Modulation Transfer Function (MTF)')
            plt.xlabel('Spatial Frequency (cycles per pixel)')
            plt.ylabel('Normalized MTF')
            plt.legend(fontsize = 5)
            plt.show()
        else:
            messagebox.showerror("Error", "No image loaded or ROI not selected!")

    def show_chromatic_aberration(self):
        if self.image is not None and self.ROI_coords is not None:
            area_v = []
            for i in range(len(self.ROI_R)):
                ROI_SFR_R = self.ROI_R[i]
                ROI_SFR_G = self.ROI_G[i]
                ROI_SFR_B = self.ROI_B[i]

                x, r_norm, g_norm, b_norm = self.normalize_smooth_signal(ROI_SFR_R, ROI_SFR_G, ROI_SFR_B)
                which, area = self.compute_area_between_curves(r_norm, g_norm, b_norm)
                area_v.append(area)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.imshow(self.ROI)
            ax2.scatter(area_v, np.arange(0, len(self.ROI_R)), s=5, color='black')
            ax2.set_aspect('auto')
            ax2.set_ylabel('Pixel location (Vertical)')
            ax2.set_xlabel('Area between highest and lowest curves')
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showerror("Error", "No image loaded or ROI not selected!")

    def select_roi(self):
        if self.image is not None:
            self.ROI_coords = cv2.selectROI("Select ROI", self.image, fromCenter=False)
            self.ROI = self.image[int(self.ROI_coords[1]):int(self.ROI_coords[1] + self.ROI_coords[3]),
                       int(self.ROI_coords[0]):int(self.ROI_coords[0] + self.ROI_coords[2])]
            messagebox.showinfo("ROI Selected", "ROI selected successfully!")
            self.show_image_with_roi()
        else:
            messagebox.showerror("Error", "No image loaded!")

    def normalize_smooth_signal(self, ROI_R, ROI_G, ROI_B):
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

    def compute_area_between_curves(self, curve1, curve2, curve3):
        """Compute the area between the highest and lowest curves."""
        # Compute the area between all pairs of curves
        area1 = np.trapz(np.abs(curve1 - curve2)) + np.trapz(np.abs(curve2 - curve1))
        area2 = np.trapz(np.abs(curve1 - curve3)) + np.trapz(np.abs(curve3 - curve1))
        area3 = np.trapz(np.abs(curve2 - curve3)) + np.trapz(np.abs(curve3 - curve2))

        # Find the maximum area among all pairs
        max_area = max(area1, area2, area3)
        if max_area == area1:
            return "curve12", max_area
        elif max_area == area2:
            return "curve13", max_area
        elif max_area == area3:
            return "curve23", max_area

    def compute_lsf_and_mtf(self, curve):
        """Compute Line Spread Function (LSF) and Modulation Transfer Function (MTF)."""
        # Compute Line Spread Function (LSF)
        lsf = np.abs(np.diff(curve))
        # lsf /= np.sum(lsf)  # Normalize LSF to ensure area under curve is 1

        # Compute Modulation Transfer Function (MTF)
        mtf = np.abs(np.fft.fftshift(np.fft.fft(lsf)))
        mtf = mtf[:] / np.max(mtf)

        n = len(mtf)
        sampling_rate = 1  # Assuming each sample corresponds to one pixel
        freq = np.fft.fftshift(np.fft.fftfreq(n)) * sampling_rate

        return lsf, mtf[len(mtf) // 2:], freq[len(mtf) // 2:]

def main():
    root = Tk()
    app = ImageAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
