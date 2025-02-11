import h5py
import numpy as np
import fabio
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
import matplotlib.pyplot as plt
import pyFAI.detectors
import os
import cv2
import glob
import sys
from IPython.display import display, clear_output

class SwingData:
    def __init__(self, filename, maskfile, sectors=None):
        # Constructor: initialize the SwingData instance with a data file, a mask file, and sectors.
        self.filename = filename
        self.sectors = sectors if sectors is not None else [(0, 180)]  # Default value for sectors
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist.")
        if not os.path.exists(maskfile):
            raise FileNotFoundError(f"Mask file {maskfile} does not exist.")
        # Load mask data using fabio (commonly used for reading image files from synchrotron detectors)
        self.maskdata = fabio.open(maskfile).data
        # The sectors parameter is not stored here (commented out) but is used as input in integration methods.
        # self.sectors = sectors
        # Extract parameters and metadata from the HDF5 file; results stored in self.params dictionary.
        self.params = self.extract_from_h5()

    def extract_from_h5(self):
        """Extract data from the HDF5 file and return a dictionary of parameters."""
        with h5py.File(self.filename, "r") as f:
            # Retrieve sample name from the first group in the file.
            group = list(f.keys())[0]
            sample_name = f[group+'/sample_info/ChemSAXS/sample_name'][()].decode('utf-8')
            # Retrieve the Eiger SAXS data (2D detector images)
            target = group + '/scan_data/eiger_image'
            eiger = np.array(f[target])
            # Sometimes the shape is (n_images, 1, height, width); if so, squeeze out the extra dimension.
            if eiger.shape[1] == 1:
                eiger = eiger.squeeze(axis=1)
            # Calculate the mean image (averaging over all frames)
            eiger_mean = np.mean(eiger, axis=0)
            # Retrieve Basler microscope image
            basler_image = f[group + '/SWING/i11-c-c08__dt__basler_analyzer/image'][()]
            # Retrieve positions (start and end) for X and Z (for possible mapping or alignment)
            position_x_start = f[group + '/SWING/i11-c-c08__ex__tab-mt_tx.4/position'][()]
            position_x_end = f[group+'/SWING/i11-c-c08__ex__tab-mt_tx.4/position_post'][()]
            position_z_start = f[group + '/SWING/i11-c-c08__ex__tab-mt_tz.4/position'][()]
            position_z_end = f[group + '/SWING/i11-c-c08__ex__tab-mt_tz.4/position_post'][()]
            position = {'X_start': position_x_start, 'X_end': position_x_end,
                        'Z_start': position_z_start, 'Z_end': position_z_end}
            # Retrieve experimental parameters from the Eiger-4M group
            target = group + '/SWING/EIGER-4M'
            distance_m = f[target + '/distance'][0] / 1000  # Convert distance to meters
            pixel_size_x = f[target + '/pixel_size_x'][0] * 1e-6  # Convert pixel size to meters
            pixel_size_z = f[target + '/pixel_size_z'][0] * 1e-6
            x_center = f[target + '/dir_beam_x'][0]
            z_center = f[target + '/dir_beam_z'][0]
            nb_frames = f[target + '/nb_frames'][0]
            bin_x = f[target + '/binning_x'][0]
            bin_y = f[target + '/binning_y'][0]
            exposure_time = f[target + '/exposure_time'][0]
            # Retrieve monochromator (wavelength) information
            target = group + '/SWING/i11-c-c03__op__mono'
            wl = f[target + '/wavelength'][0]
            # Check for sample transmission data
            if 'sample_info/sample_transmission' in f[group]:
                transmission = np.array(f[group + '/sample_info/sample_transmission'])
            else:
                transmission = 1  # or set a default value
                print("Warning: 'sample_transmission' does not exist in the file.")

            # Check for average_mi8b data
            if 'scan_data/averagemi8b' in f[group]:
                averagemi8b = np.array(f[group + '/scan_data/averagemi8b'])
            else:
                averagemi8b = 1  # or set a default value
                print("Warning: 'averagemi8b' does not exist in the file.")

            # Retrieve time stamps for the images
            target = group + '/scan_data/eiger_timestamp'
            time_stamps = np.array(f[target])
            # Generate a reduced filename by removing a prefix (e.g., proposal name) for later use in naming outputs.
            reduce_filename = os.path.basename(self.filename)
            reduce_filename = "_".join(reduce_filename.split('_')[1:])
            # Get the folder containing the HDF5 file
            folder = os.path.dirname(self.filename)

            # Store all extracted parameters in a dictionary
            self.params = {
                "Sample_Name": sample_name,
                "eiger": eiger,
                "eiger_mean": eiger_mean,
                "basler_image": basler_image,
                "WaveLength": wl,
                "Center_1": x_center, "Center_2": z_center,
                "PixSize_x": pixel_size_x, "PixSize_z": pixel_size_z,
                "SampleDistance": distance_m,
                "Dim_1": eiger.shape[1], "Dim_2": eiger.shape[2],
                "Binning_1": int(bin_x), 'Binning_2': int(bin_y),
                "nb_frames": nb_frames,
                "exposure_time": exposure_time,
                "averagemi8b": averagemi8b,
                "transmission": transmission,
                "time_stamps": time_stamps,
                "position": position,
                "filename": self.filename,
                "reduce_filename": reduce_filename,
                "folder": folder,
            }
        return self.params

    def integrate(self, sectors=None, mean=True):
        """ 
        Integrate eiger_mean or all eiger images.
        Returns the integrated intensity [i(q)] corrected for exposure time and monitor (mi8).
        
        Parameters:
            sectors (list of tuples): List of sectors defined as (angle, delta_angle) in degrees.
            mean (bool): If True, integrate the mean image; otherwise, integrate each frame individually.
        
        Returns:
            results (dict): Dictionary with keys (angle, delta_angle) and values containing:
                - "q": The scattering vector (q) values.
                - "i_raw": The raw integrated intensity.
                - "i_normalize": The normalized intensity.
            Also, the chi-map is computed and stored in self.chi_array.
        """
        # Retrieve detector and experimental parameters
        pixel_size_x = self.params['PixSize_x']
        pixel_size_z = self.params['PixSize_z']
        bin_x = self.params['Binning_1']
        bin_y = self.params['Binning_2']
        distance_m = self.params['SampleDistance']
        x_center = self.params['Center_1']
        z_center = self.params['Center_2']
        wl = self.params['WaveLength']
        nb_frames = self.params["nb_frames"]
        nbins = 1000  # Number of points for integration
        unit_type = "q_A^-1"  # Unit for scattering vector q

        # Create a detector object with proper pixel size (accounting for binning)
        detector = pyFAI.detectors.Detector(pixel1=pixel_size_x * bin_x, pixel2=pixel_size_z * bin_y)
        # Create the AzimuthalIntegrator object with the detector and sample distance
        ai = AzimuthalIntegrator(dist=distance_m, detector=detector)
        # Set the geometry parameters (distance in mm, beam center coordinates, and wavelength)
        ai.setFit2D(distance_m * 1000, x_center, z_center, wavelength=wl)
        results = {}  # Dictionary to store integration results

        # Calculate chi-map (azimuthal angle array) using the image shape from the HDF5 file
        image_shape = (self.params["Dim_1"], self.params["Dim_2"])
        self.chi_array = ai.chiArray(shape=image_shape)

        # Integration over sectors:
        if mean:
            # Use the mean image for integration
            image = self.params['eiger_mean']
            for angle, delta_angle in self.sectors:
                azimuth_range = (angle - delta_angle, angle + delta_angle)
                q, i = ai.integrate1d(image, nbins, azimuth_range=azimuth_range, unit=unit_type, mask=self.maskdata)
                results[(angle, delta_angle)] = {"q": np.array(q), "i_raw": np.array(i)}
        else:
            # Use all individual frames for integration
            image = self.params['eiger']
            for angle, delta_angle in self.sectors:
                q_list = []
                i_list = []
                azimuth_range = (angle - delta_angle, angle + delta_angle)
                for k in range(nb_frames):
                    q, i = ai.integrate1d(image[k], nbins, azimuth_range=azimuth_range, unit=unit_type, mask=self.maskdata)
                    q_list.append(q)
                    i_list.append(i)
                results[(angle, delta_angle)] = {"q": np.array(q_list), "i_raw": np.array(i_list)}
        
        # Correction: Normalize the integrated intensity using the monitor (mi8) and exposure time.
        mi8 = np.mean(self.params["averagemi8b"])
        exposure_time = self.params["exposure_time"]

        for key in results:
            i_raw = results[key]["i_raw"]
            i_normalize = i_raw / (mi8 * exposure_time * 30700)  # Normalization factor applied
            results[key]["i_normalize"] = i_normalize

        # Return both the integration results and the computed chi-array
        return results

    def save_integration(self, integration_results, normalized=False):
        """ 
        Save integration results to text files, with file names including sector and frame information.
        
        Args:
            integration_results (dict): Dictionary containing integration data with:
                - Keys: (angle, delta_angle) tuples (defining the sector).
                - Values: dictionaries with keys 'q', 'i_raw', and optionally 'i_normalize'.
            normalized (bool, optional): If True, save normalized intensity; otherwise, save raw intensity.
        """
        self.integration_folder = os.path.join(self.params["folder"], "integration")
        os.makedirs(self.integration_folder, exist_ok=True)

        self.filename_suffix = "_integ_norm" if normalized else "_integ_raw"
        self.intensity_key = "i_normalize" if normalized else "i_raw"
        
        for (angle, delta_angle), data in integration_results.items():
            q = data["q"]
            i = data[self.intensity_key]

            if q.ndim == 2 and i.ndim == 2:  # Multiple frames
                for frame in range(q.shape[0]):
                    filename = os.path.join(
                        self.integration_folder,
                        f"{self.params['reduce_filename']}{self.filename_suffix}_frame{frame + 1:03d}_sector_{angle}_{delta_angle}.txt"
                    )
                    np.savetxt(filename, np.column_stack((q[frame], i[frame])), header="q, i")
            else:  # Single frame (mean image)
                filename = os.path.join(
                    self.integration_folder, 
                    f"{self.params['reduce_filename']}{self.filename_suffix}_sector_{angle}_{delta_angle}.txt"
                )
                np.savetxt(filename, np.column_stack((q, i)), header="q, i")

    def plot_data(self, integration_results, normalized=True, basler_coords=(0, 0), output_filename=None):
        """Save plots combining Basler image, Eiger mean image, and integration results.
        
        Args:
            integration_results (dict): Dictionary with integration data per sector.
                - Keys: (angle, delta_angle) tuples.
                - Values: dictionaries containing 'q', 'i_raw', and 'i_normalize'.
            normalized (bool): If True, plot normalized intensity; else, plot raw intensity.
            basler_coords (tuple): Coordinates (x, y) to mark on the Basler image.
            output_filename (str, optional): Custom filename template.
        """
        plots_folder = os.path.join(self.params["folder"], "plots")
        os.makedirs(plots_folder, exist_ok=True)
        for (angle, delta_angle), data in integration_results.items():
            q = data["q"]
            i = data[self.intensity_key]
            # Handle multi-frame data
            if q.ndim == 2 and i.ndim == 2:
                for frame_idx in range(q.shape[0]):
                    filename = os.path.join(
                        plots_folder,
                        f"{self.params['reduce_filename']}{self.filename_suffix}_frame{frame_idx + 1:03d}_sector_{angle}_{delta_angle}.png"
                    )
                    self._create_plot(
                        q=q[frame_idx],
                        i=i[frame_idx],
                        angle=angle,
                        delta_angle=delta_angle,
                        frame_idx=frame_idx+1,
                        normalized=normalized,
                        basler_coords=basler_coords,
                        plots_folder=plots_folder,
                        output_filename=filename
                    )
            else:  # Single frame or mean image
                filename = os.path.join(
                        plots_folder, 
                        f"{self.params['reduce_filename']}{self.filename_suffix}_sector_{angle}_{delta_angle}.png"
                    )
                self._create_plot(
                    q=q,
                    i=i,
                    angle=angle,
                    delta_angle=delta_angle,
                    frame_idx=None,
                    normalized=normalized,
                    basler_coords=basler_coords,
                    plots_folder=plots_folder,
                    output_filename=filename
                )

    def _create_plot(self, q, i, angle, delta_angle, frame_idx, normalized, basler_coords, plots_folder, output_filename):
        """Helper function to create and save individual plots.
        
        This function creates a figure with three panels:
            - Panel 1: Basler image with a marker at the specified coordinates.
            - Panel 2: Eiger mean image with an overlay of the chi map for the given sector.
            - Panel 3: Integration plot (log-log) of intensity vs. q.
        The filename is generated based on the sample's reduced filename, frame, and sector information.
        """
        # Generate filename components based on parameters
        base_name = self.params["reduce_filename"]
        suffix = "_norm" if normalized else "_raw"
        sector_info = f"sector_{angle}_{delta_angle}"
        
        if frame_idx:
            frame_info = f"frame{frame_idx:03d}"
            default_name = f"{base_name}_{frame_info}_{sector_info}{suffix}.png"
        else:
            default_name = f"{base_name}_{sector_info}{suffix}.png"
        
        if output_filename:
            final_filename = output_filename.format(
                sector=angle,
                delta=delta_angle,
                frame=frame_idx or 0
            )
        else:
            final_filename = os.path.join(plots_folder, default_name)

        # Create the figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        
        # Panel 1: Display the Basler image and mark the measurement point.
        axes[0].imshow(self.params['basler_image'], cmap='gray')
        axes[0].scatter(basler_coords[0], basler_coords[1], color='red')
        axes[0].set_title('Basler Image')
        axes[0].axis('off')
        
        # Panel 2: Display the Eiger mean image and overlay the chi map.
        # Compute the chi map in degrees.
        chi_array_deg = np.degrees(self.chi_array)  # Convert from radians to degrees
        chi_array_deg = (chi_array_deg + 180) % 360 - 180  # Recenter to [-180, 180]
        # Create a mask for the current sector (between angle-delta_angle and angle+delta_angle)
        mask = (chi_array_deg > angle - delta_angle) & (chi_array_deg < angle + delta_angle)
        if q.ndim == 2 and i.ndim == 2:  # For multi-frame data, use the corresponding frame's Eiger mean image.
            axes[1].imshow(np.log1p(self.params['eiger_mean'][frame_idx]), cmap='jet')
        else:
            axes[1].imshow(np.log1p(self.params['eiger_mean']), cmap='jet')
        axes[1].set_title('Eiger Mean Image')
        axes[1].axis('off')
        # Overlay the chi map mask (displayed in gray)
        axes[1].imshow(mask.astype(float), cmap='gray', alpha=0.5)
        
        # Panel 3: Plot the integrated intensity as a function of q in log-log scale.
        axes[2].loglog(q, i)
        axes[2].set_xlabel('q (Å⁻¹)')
        axes[2].set_ylabel('Intensity')
        plot_title = f"{'Normalized' if normalized else 'Raw'} Integration"
        if frame_idx:
            plot_title += f" - Frame {frame_idx}"
        axes[2].set_title(plot_title)
        
        # Overall figure title includes the sample name and sector information.
        fig.suptitle(f"{self.params['Sample_Name']}\nSector: {angle}±{delta_angle}°")
        plt.tight_layout()
        
        # Save the figure to file and close the figure to free memory.
        fig.savefig(final_filename, dpi=300)
        plt.close(fig)

class SwingBatch:
    def __init__(self, data_dir, mask_path, integrate_mean=True, normalize=False, 
                basler_coords=(0, 0), log_file="processing_log.txt",sectors=None):
        """
        SwingBatch class to process a batch of HDF5 files from synchrotron experiments.
        
        Args:
            data_dir (str): Directory containing the HDF5 files.
            mask_path (str): Path to the mask file.
            sectors (list of tuples): List of sectors defined as (angle, delta_angle) in degrees for integration.
                                      Default is [(0, 180)].
            integrate_mean (bool): If True, integrate the mean image; if False, integrate each frame individually.
            normalize (bool): If True, normalized integrated intensity will be used.
            basler_coords (tuple): (x, y) coordinates for marking the measurement point on the Basler image.
            log_file (str): Name of the log file to record processing details.
        """
        print('------------ Process Start ------------')
        self.sectors = sectors if sectors is not None else [(0, 180)]  # Default value for sectors
        self.data_dir = data_dir
        self.mask_path = mask_path
        self.sectors = sectors
        self.integrate_mean = integrate_mean
        self.normalize = normalize
        self.basler_coords = basler_coords
        self.processed_files = []
        self.failed_files = []  # List of tuples (file, error) for files that failed processing
        self.log_file = os.path.join(self.data_dir, log_file)
        

    def get_h5_files(self):
        """Return list of HDF5 files in the data directory."""
        files = []
        for f in os.listdir(self.data_dir):
            if f.endswith((".h5", ".nxs")):

                files.append(os.path.join(self.data_dir, f))
        if not files:
            raise FileNotFoundError(f"No *.h5 file in {self.data_dir}")
        return files

    def process(self):
        """Process all HDF5 files with a progress indicator."""
        files = self.get_h5_files()
        total_files = len(files)
        
        if total_files == 0:
            print("No .h5 files found.")
            return

        for i, file in enumerate(files, 1):
            try:
                self._process_single_file(file)
                self.processed_files.append(file)
            except Exception as e:
                self.failed_files.append((file, str(e)))
            
            progress_percentage = int((i / total_files) * 100)

            # If running in a Jupyter Notebook, update progress display
            if 'ipykernel' in sys.modules:
                clear_output(wait=True)
                display(f"Processing... {i}/{total_files} files ({progress_percentage}%)")
            else:
                sys.stdout.write(f"\rProcessing... {i}/{total_files} files ({progress_percentage}%)")
                sys.stdout.flush()

        self._log_results()
   

    def _process_single_file(self, file_path):
        """Process a single HDF5 file: extraction, integration, and plotting."""
        data = SwingData(file_path, self.mask_path, self.sectors)
        integration = data.integrate(sectors=self.sectors, mean=self.integrate_mean)
        data.save_integration(integration, self.normalize)
        data.plot_data(integration, self.normalize, basler_coords=self.basler_coords)

    def _log_results(self):
        """Write a log file summarizing the processing results."""
        try:
            with open(self.log_file, "w") as log:
                log.write("Processing HDF5 files\n")
                log.write(f"Successfully processed files: {len(self.processed_files)}\n")
                log.write("Processed files:\n")
                for file in self.processed_files:
                    log.write(f"  {file}\n")
                log.write(f"\nFailed files: {len(self.failed_files)}\n")
                if self.failed_files:
                    log.write("Error details:\n")
                    for file, error in self.failed_files:
                        log.write(f"  {file}: {error}\n")
            print(f"Log file created: {self.log_file}")
        except Exception as e:
            print(f"Error writing log file: {e}")

    def get_report(self):
        """Return a processing report as a string."""
        report = f"Files converted: {len(self.processed_files)}\nFailures: {len(self.failed_files)}\n"
        if self.failed_files:
            report += "Failed conversions:\n" + "\n".join(f"{f[0]} : {f[1]}" for f in self.failed_files)
        return report

    def create_video_from_images(self, output_filename="output_video.mp4", fps=10):
        """
        Convert the images in the 'plots' folder into an MP4 video.
        
        Parameters:
            output_filename (str): Output video file name (default: 'output_video.mp4').
            fps (int): Frames per second (default: 10).
        """
        plots_folder = os.path.join(self.data_dir, "plots")
        image_files = sorted(glob.glob(os.path.join(plots_folder, "*.png")))
        if not image_files:
            raise FileNotFoundError(f"No images found in {plots_folder}")
        first_image = cv2.imread(image_files[0])
        height, width, _ = first_image.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        save_path = os.path.join(plots_folder, output_filename)
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        for image_file in image_files:
            img = cv2.imread(image_file)
            video_writer.write(img)
        video_writer.release()
        print(f"Video saved in: {save_path}")
        print('------------ Process Finish ------------')


import numpy as np

import numpy as np

class TextFileProcessor:
    def __init__(self):
        """Initialize the object with a dictionary to store file data."""
        self.files = {}

    def load_txt(self, path):
        """Loads q and i data from a Foxtrot-format text file, trying different skiprows values."""
        last_exception = None

        for skip in range(100):  # Try different skiprows values from 0 to 99
            try:
                sample = np.loadtxt(path, skiprows=skip)
                if sample.shape[1] >= 2:  # Ensure there are at least 2 columns
                    q = sample[:, 0]
                    i = sample[:, 1]
                    self.files[path] = (q, i)  # Store data as a tuple (q, i)
                    print(f"Successfully loaded {path} with skiprows={skip}")
                    return q, i  # Return data immediately if successful
            except Exception as e:
                last_exception = e  # Store the last exception for reporting

        print(f"Failed to load {path}: {last_exception}")  # Final error message
        return None, None  # Return None if all attempts fail

    def get_data(self, path):
        """Returns the (q, i) data for a loaded file."""
        return self.files.get(path, (None, None))  # Return None if file not found

    def subtract_files(self, file1, file2):
        """Subtracts data from two files and saves the result in a new file inside a 'sub' folder."""
        # Retrieve data from both files
        q1, i1 = self.get_data(file1)
        q2, i2 = self.get_data(file2)

        if q1 is None or q2 is None:
            print("Error: One or both files could not be loaded.")
            return

        # Interpolation if q-values don't match
        if not np.array_equal(q1, q2):
            print("Interpolating data to match q-values...")
            i2_interpolated = np.interp(q1, q2, i2)
        else:
            i2_interpolated = i2  # No interpolation needed if q-values are already the same

        # Perform the subtraction
        i_subtracted = i1 - i2_interpolated

        # Prepare the output file name
        folder = os.path.dirname(file1)  # Get the folder containing the first file
        subfolder = os.path.join(folder, "sub")
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)  # Create the subfolder if it doesn't exist

        # File name consists of the first 5 characters of both files
        file1_name = os.path.basename(file1)[:5]
        file2_name = os.path.basename(file2)[:5]
        output_filename = f"{file1_name}-{file2_name}_subtracted.txt"
        output_path = os.path.join(subfolder, output_filename)

        # Save the result to the new file
        np.savetxt(output_path, np.column_stack((q1, i_subtracted)), header="q  i_subtracted", comments="")
        print(f"Subtraction completed. Result saved to {output_path}")

