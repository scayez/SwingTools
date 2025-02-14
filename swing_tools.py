import h5py
import numpy as np
import fabio
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import matplotlib.pyplot as plt
import pyFAI.detectors
import os
import cv2 
import glob
import sys
from IPython.display import display, clear_output
from pathlib import Path
import subprocess

class Mask:
    def __init__(self,file:str):
        """file : str can be *.h5 file (methods nxs2edf and draw_mask) or foxtrot mask (method convertfoxtrotmask)"""
        self.file=file
        self.path=os.path.dirname(file)
        self.folder=self.path


    def convertfoxtrotmask(self):
            """
            maskfile:str Path to foxtrot mask file
            Convert a mask file created in foxtrot to *.edf format, used in SwingData class, sasview,...
            Note that the convention is different in foxtrot (1 becomes 0 and vice-versa)
            """
            maskfile=self.file
            maskarray=np.loadtxt(maskfile,delimiter=';')
            mask_ok=1-maskarray
            outputname=os.path.dirname(maskfile)+'/'+os.path.splitext(os.path.basename(maskfile))[0]+".edf"
            obj = fabio.edfimage.EdfImage(data=mask_ok)
            obj.write(outputname)
        
    

    def draw_mask(self):
        with h5py.File(self.file, "r") as f:
            group = list(f.keys())[0]
            self.folder=os.getcwd()
            # Retrieve experimental parameters from the Eiger-4M group
            target = group + '/SWING/EIGER-4M'
            self.distance_m = f[target + '/distance'][0] / 1000  # Convert distance to meters
            self.pixel_size_x = f[target + '/pixel_size_x'][0] * 1e-6  # Convert pixel size to meters
            self.pixel_size_z = f[target + '/pixel_size_z'][0] * 1e-6
            self.x_center = f[target + '/dir_beam_x'][0]
            self.z_center = f[target + '/dir_beam_z'][0]
            self.nb_frames = f[target + '/nb_frames'][0]
            self.bin_x = f[target + '/binning_x'][0]
            self.bin_y = f[target + '/binning_y'][0]
            self.exposure_time = f[target + '/exposure_time'][0]
            # Retrieve monochromator (wavelength) information
            target = group + '/SWING/i11-c-c03__op__mono'
            self.wl = f[target + '/wavelength'][0]
            # Retrieve scattering data
            target = group + '/scan_data/eiger_image'
            eiger_raw = np.array(f[target])
        
        if eiger_raw.ndim==2:
            data=eiger_raw
        else:
            data=np.mean(eiger_raw,axis=0)
        header={'SampleDistance':self.distance_m,
                'WaveLength':self.wl,
                'Dim_1':data.shape[0],
                'Dim_2':data.shape[1],
                'Pixel_1':self.pixel_size_x,
                'Pixel_2':self.pixel_size_z,
                'Center_1':self.x_center,
                'Center_2':self.z_center}
        file=self.folder+'/temp.edf'
        obj = fabio.edfimage.EdfImage(header=header,data=data)
        obj.write(file) 
        command = ['pyFAI-drawmask', file]
        
        # Run the command using subprocess.run() and capture output and errors
        # for some reason it may be neccesary to remove the QT_QPA_PLATFORM_PLUGIN_PATH environment variable
        try:
            del(os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'])
        except:
            pass
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True, cwd=self.folder)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            print(f"stderr: {e.stderr}")
            print(f"stdout: {e.stdout}")
        os.remove(file)

class SwingData:
    def __init__(self, 
                filename: str,
                maskfile: str,
                basler_coords : tuple =[0,0],
                basler_calibration: tuple =[3.73,3.67],
                anisotropic_data: bool =False,
                sectors: list=None,
                average_frames:bool=True,
                transmission_correction:bool=True):
        """ 
        filename: str path to data faile (*.h5 or *.nxs)
        maskfile: str path to mask file (*.edf format)
        basler_coords: tuple pixel coordinates of X-ray beam on basler image
        balser_calibration: tuple calibration of basler image (in µm/pixels), along X and along Z
        anisotropic_data: bool True top optimize integration of anisotropic data (slow down computation)
        sectors: List(tuple) [(angle, delta)]
        average_frames: bool Tag to perform averaging of scattering frames contained in h5 file
        transmission_correction: bool Tag to perform transmission corrections
        """
        # Constructor: initialize the SwingData instance with a data file, a mask file, and sectors.
        self.filename = filename
        self.folder=os.path.dirname(filename)
        self.path=self.folder
        self.sectors = sectors if sectors is not None else [(0, 180)]  # Default value for sectors
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist.")
        #if not os.path.exists(maskfile):
        #    raise FileNotFoundError(f"Mask file {maskfile} does not exist.")
        # Load mask data using fabio (commonly used for reading image files from synchrotron detectors)
        
        self.cavingtag=anisotropic_data
        # Initialize basler coordinates
        self.basler_coords=basler_coords
        self.basler_calibration=basler_calibration
        # Initialize mean (True to merge/average frames contained in h5 files)
        self.mean=average_frames
        # Extract parameters and metadata from the HDF5 file; results stored in self.params dictionary.
        self.extract_from_h5()
        
        #Initialize normalized (True to use transmission corrections)
        self.normalized=transmission_correction
        self.maskfile=maskfile
        if self.maskfile is not None:
            self.maskdata = fabio.open(maskfile).data
        else:
            print('Please provide a mask file. You can create one using Utils.draw_mask')
            
        # compute intensity taking into accoun average and anisotropic tags
        self.eiger=self.extract_scatteringdata()

    
    def extract_from_h5(self):
        """Extract data from the HDF5 file and return a dictionary of parameters."""
        with h5py.File(self.filename, "r") as f:
            # Retrieve sample name from the first group in the file.
            group = list(f.keys())[0]
            self.sample_name = f[group+'/sample_info/ChemSAXS/sample_name'][()].decode('utf-8')
                       
            # Retrieve Basler microscope image
            self.basler_image = f[group + '/SWING/i11-c-c08__dt__basler_analyzer/image'][()]
            # Retrieve positions (start and end) for X and Z (for possible mapping or alignment)
            self.position_x_start = f[group + '/SWING/i11-c-c08__ex__tab-mt_tx.4/position'][()]
            self.position_x_end = f[group+'/SWING/i11-c-c08__ex__tab-mt_tx.4/position_post'][()]
            self.position_z_start = f[group + '/SWING/i11-c-c08__ex__tab-mt_tz.4/position'][()]
            self.position_z_end = f[group + '/SWING/i11-c-c08__ex__tab-mt_tz.4/position_post'][()]
            self.position = {'X_start': self.position_x_start, 'X_end': self.position_x_end,
                        'Z_start': self.position_z_start, 'Z_end': self.position_z_end}
            # Retrieve experimental parameters from the Eiger-4M group
            target = group + '/SWING/EIGER-4M'
            self.distance_m = f[target + '/distance'][0] / 1000  # Convert distance to meters
            self.pixel_size_x = f[target + '/pixel_size_x'][0] * 1e-6  # Convert pixel size to meters
            self.pixel_size_z = f[target + '/pixel_size_z'][0] * 1e-6
            self.x_center = f[target + '/dir_beam_x'][0]
            self.z_center = f[target + '/dir_beam_z'][0]
            if self.mean:
                self.nb_frames=1
            else:
                self.nb_frames = f[target + '/nb_frames'][0]
            self.bin_x = f[target + '/binning_x'][0]
            self.bin_y = f[target + '/binning_y'][0]
            self.exposure_time = f[target + '/exposure_time'][0]
            # Calculate step_x  and step_z
            step_x=1000*(self.position_x_end-self.position_x_start)/self.nb_frames
            step_z=1000*(self.position_z_end-self.position_z_start)/self.nb_frames

            # Compute step_x and step_z in pixel coordinates (on basler_image)
            self.step_x=int(step_x/self.basler_calibration[0]) # Calibration of basler image
            self.step_z=int(step_z/self.basler_calibration[1])
            
            # Retrieve monochromator (wavelength) information
            target = group + '/SWING/i11-c-c03__op__mono'
            self.wl = f[target + '/wavelength'][0]
            # Check for sample transmission data
            if 'sample_info/sample_transmission' in f[group]:
                self.transmission = np.array(f[group + '/sample_info/sample_transmission'])
            else:
                self.transmission = 1  # or set a default value
                print("Warning: 'sample_transmission' does not exist in the file.")

            # Check for average_mi8b data
            if 'scan_data/averagemi8b' in f[group]:
                self.averagemi8b = np.array(f[group + '/scan_data/averagemi8b'])
            else:
                self.averagemi8b = 1  # or set a default value
                print("Warning: 'averagemi8b' does not exist in the file.")

            # Retrieve time stamps for the images
            target = group + '/scan_data/eiger_timestamp'
            self.time_stamps = np.array(f[target])
            # Generate a reduced filename by removing a prefix (e.g., proposal name) for later use in naming outputs.
            reduce_filename = os.path.basename(self.filename)
            self.reduce_filename = "_".join(reduce_filename.split('_')[1:])
            # Get the folder containing the HDF5 file
            self.folder = os.path.dirname(self.filename)
        return 

    def extract_scatteringdata(self):

        with h5py.File(self.filename, "r") as f:
            # Retrieve sample name from the first group in the file.
            group = list(f.keys())[0]
            # Retrieve the Eiger SAXS data (2D detector images)
            target = group + '/scan_data/eiger_image'
            eiger_raw = np.array(f[target])
            self.Dim_1=eiger_raw.shape[1];self.Dim_2=eiger_raw.shape[2]
            # Sometimes the shape is (n_images, 1, height, width); if so, squeeze out the extra dimension.
            if eiger_raw.shape[1] == 1:
                eiger_raw = eiger_raw.squeeze(axis=1)
 
        # Replace data with average data when averageframes=True
        if self.mean:
            eiger_mean = np.expand_dims(np.mean(eiger_raw, axis=0),axis=0)
            
            # Perform "caving" if anistropic
            if self.cavingtag:
                if self.maskfile is not None:
                    cave_data = eiger_mean.copy() # shape[0,Dim_1,Dim_2]

                    tag=1000000000000.0
                    
                    for y in np.arange(int(self.Dim_1)):
                        for x in np.arange(int(self.Dim_2)):
                            # tag masked pixels with very large value
                            if self.maskdata[y,x] == 1.0:
                                cave_data[0,y,x] = tag
                                                
                    for y in np.arange(int(self.Dim_1)):
                        for x in np.arange(int(self.Dim_2)):
                        # apply inversion center
                            xsym=int(2*self.x_center-x)
                            ysym=int(2*self.z_center-y)
                            if xsym<=int(self.Dim_1)-1 and xsym>=0 and ysym<=int(self.Dim_2)-1 and ysym>=0:
                                if cave_data[0,y,x] == tag :                                
                                    if cave_data[0,ysym,xsym] != tag:
                                        self.maskdata[y,x]=0 #unmask pixel
                                        cave_data[0,y,x]=cave_data[0,ysym,xsym]  # replace masked pixel by its equivalent              
                    self.eiger=cave_data
                    # create caving_mask file
                    maskpath=self.folder+'/caving_mask.edf'
                    self.writeedf(maskpath,self.maskdata)
                    # reattribute caving_mask for integration
                    self.maskfile=maskpath
                else:
                    print('Please specify a maskfile. Maskfiles can be created using Mask.draw_mask() method')
            else:
                self.eiger=eiger_mean
        # Case of no averaging
        else:
            
            if self.cavingtag:
                if self.maskfile is not None:
                    cave_data = eiger_raw.copy()
                    # Tag masked pixels (np.argwhere could be more elegant?)
                    for n in range(self.nb_frames):
                        tag=100000000.0
                        for y in np.arange(int(self.Dim_1)):
                            for x in np.arange(int(self.Dim_2)):
                                # tag masked pixels with very large value
                                if self.maskdata[y,x] == 1.0:
                                    cave_data[n,y,x] = tag
                    # Replace tagged pixels with symmetry equivalent ones
                    for n in range(self.nb_frames):
                        for y in np.arange(int(self.Dim_1)):
                            for x in np.arange(int(self.Dim_2)):
                            # apply inversion center
                                xsym=int(2*self.x_center-x)
                                ysym=int(2*self.z_center-y)
                                if xsym<=int(self.Dim_1)-1 and xsym>=0 and ysym<=int(self.Dim_2)-1 and ysym>=0:
                                    if cave_data[n,y,x] == tag :
                                        if cave_data[n,ysym,xsym] != tag:
                                            self.maskdata[y,x]=0 #unmask pixel
                                            cave_data[n,y,x]=cave_data[n,ysym,xsym]                
                        #self.eiger=cave_data
                        
                    # create caving_mask file
                    maskpath=self.folder+'/caving_mask.edf'
                    self.writeedf(maskpath,self.maskdata)
                    # reattribute caving_mask for integration
                    self.maskfile=maskpath
                    # set self.eiger          
                    self.eiger=cave_data
                
                else:
                    print('Please specify a maskfile. Maskfiles can be created using Mask.draw_mask() method')
            else:# no average, no caving
                self.eiger=eiger_raw
            
        return self.eiger
             
    def writeedf(self,name,data):
        with h5py.File(self.filename, "r") as f:
            group = list(f.keys())[0]
            
            # Retrieve experimental parameters from the Eiger-4M group
            target = group + '/SWING/EIGER-4M'
            self.distance_m = f[target + '/distance'][0] / 1000  # Convert distance to meters
            self.pixel_size_x = f[target + '/pixel_size_x'][0] * 1e-6  # Convert pixel size to meters
            self.pixel_size_z = f[target + '/pixel_size_z'][0] * 1e-6
            self.x_center = f[target + '/dir_beam_x'][0]
            self.z_center = f[target + '/dir_beam_z'][0]
                       
            # Retrieve monochromator (wavelength) information
            target = group + '/SWING/i11-c-c03__op__mono'
            self.wl = f[target + '/wavelength'][0]
            # Retrieve scattering data
            target = group + '/scan_data/eiger_image'
            
        header={'SampleDistance':self.distance_m,
                'WaveLength':self.wl,
                'Dim_1':data.shape[0],
                'Dim_2':data.shape[1],
                'Pixel_1':self.pixel_size_x,
                'Pixel_2':self.pixel_size_z,
                'Center_1':self.x_center,
                'Center_2':self.z_center}
        
        obj = fabio.edfimage.EdfImage(header=header,data=data)
        obj.write(name) 

    
    def integrate(self):
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
        # set current mask        
        if self.maskfile is not None:
            
            self.maskdata = fabio.open(self.maskfile).data


        nbins = 1000  # Number of points for integration
        unit_type = "q_A^-1"  # Unit for scattering vector q

        # Create a detector object with proper pixel size (accounting for binning)
        detector = pyFAI.detectors.Detector(pixel1=self.pixel_size_x * self.bin_x, pixel2=self.pixel_size_z * self.bin_y)
        # Create the AzimuthalIntegrator object with the detector and sample distance
        ai = AzimuthalIntegrator(dist=self.distance_m, detector=detector)
        # Set the geometry parameters (distance in mm, beam center coordinates, and wavelength)
        ai.setFit2D(self.distance_m * 1000, self.x_center, self.z_center, wavelength=self.wl)
        
        results = {}  # Dictionary to store integration results

        # Calculate chi-map (azimuthal angle array) using the image shape from the HDF5 file
        image_shape = (self.Dim_1, self.Dim_2)
        self.chi_array = ai.chiArray(shape=image_shape)

        # Integration over sectors:
        
        
        for angle, delta_angle in self.sectors:
            q_list = []
            i_list = []
            azimuth_range = (angle - delta_angle, angle + delta_angle)
            for k in range(self.nb_frames):
                q, i = ai.integrate1d(self.eiger[k], nbins, azimuth_range=azimuth_range, unit=unit_type, mask=self.maskdata)
                q_list.append(q)
                i_list.append(i)
            results[(angle, delta_angle)] = {"q": np.array(q_list), "i_raw": np.array(i_list)}
        # Correction: Normalize the integrated intensity using the monitor (mi8) and exposure time.
        mi8 = np.mean(self.averagemi8b)
        exposure_time = self.exposure_time

        for key in results:
            i_raw = results[key]["i_raw"]
            i_normalize = i_raw / (mi8 * exposure_time * 30700)  # Normalization factor applied
            results[key]["i_normalize"] = i_normalize

        # Return both the integration results and the computed chi-array
        return results

    def integrate2D(self):

        """ 
        performs 2D integration for anisotropic data only
        Integrated maps are stored in integration2D directory
        qmap: array of q values for each individual frames (shape=(nb_frames,n_bins))
        chimap: array of chi values for each individual frames (shape=(nb_frames,360))
        Imap: array of intensities for each individual frames (shape=()
        """
        if self.cavingtag:
            
            if self.maskfile is not None:
                self.maskdata = fabio.open(self.maskfile).data


            nbins = 1000  # Number of points for integration
            unit_type = "q_A^-1"  # Unit for scattering vector q

            # Create a detector object with proper pixel size (accounting for binning)
            detector = pyFAI.detectors.Detector(pixel1=self.pixel_size_x * self.bin_x, pixel2=self.pixel_size_z * self.bin_y)
            # Create the AzimuthalIntegrator object with the detector and sample distance
            ai = AzimuthalIntegrator(dist=self.distance_m, detector=detector)
            # Set the geometry parameters (distance in mm, beam center coordinates, and wavelength)
            ai.setFit2D(self.distance_m * 1000, self.x_center, self.z_center, wavelength=self.wl)
            # initialize arrays to store results
            qmap_array=np.zeros([self.nb_frames,nbins])
            chimap_array=np.zeros([self.nb_frames,360])
            imap_array=np.zeros([self.nb_frames,360,nbins])
            # integrate individual frames
            for i in range(self.nb_frames):
                res2d = ai.integrate2d(self.eiger[i],nbins,360,unit=unit_type,mask=self.maskdata)
                Imap,qmap,chimap=res2d
                qmap=np.squeeze(qmap)
                chimap=np.squeeze(chimap)
                qmap_array[i] = qmap; chimap_array[i] = chimap ; imap_array[i,:,:] = Imap
            
            results={'qmap':qmap_array,'chimap':chimap_array,'imap':imap_array}
        return results



    def save_integration(self):
        """ 
        Save integration results to text files, with file names including sector and frame information.
        
        Args:
            integration_results (dict): Dictionary containing integration data with:
                - Keys: (angle, delta_angle) tuples (defining the sector).
                - Values: dictionaries with keys 'q', 'i_raw', and optionally 'i_normalize'.
            normalized (bool, optional): If True, save normalized intensity; otherwise, save raw intensity.
        """
        integration_results=self.integrate()
        self.integration_folder = os.path.join(self.folder, "integration")
        os.makedirs(self.integration_folder, exist_ok=True)

        self.filename_suffix = "_integ_norm" if self.normalized else "_integ_raw"
        self.intensity_key = "i_normalize" if self.normalized else "i_raw"
        
        integration_results=self.integrate()
        
        for (angle, delta_angle), data in integration_results.items():
            q = data["q"]
            i = data[self.intensity_key]

            if q.ndim == 2 and i.ndim == 2:  # Multiple frames
                for frame in range(self.nb_frames):
                    filename = os.path.join(
                        self.integration_folder,
                        f"{self.reduce_filename}{self.filename_suffix}_frame{frame + 1:03d}_sector_{angle}_{delta_angle}.txt"
                    )
                    np.savetxt(filename, np.column_stack((q[frame], i[frame])), header="q, i")
            else:  # Single frame (mean image)
                filename = os.path.join(
                    self.integration_folder, 
                    f"{self.reduce_filename}{self.filename_suffix}_sector_{angle}_{delta_angle}.txt"
                )
                np.savetxt(filename, np.column_stack((q, i)), header="q, i")
        
        # For anisotropic data, save 2D integration
        if self.cavingtag:
            #create outputdir
            outdir=self.folder+'/integration2D'
            os.makedirs(outdir,exist_ok=True)
            results=self.integrate2D()
            for frame in range(self.nb_frames):
                filenameq = os.path.join(outdir,f"{self.reduce_filename}{self.filename_suffix}_frame{frame + 1:03d}_q.txt")
                filenamechi = os.path.join(outdir,f"{self.reduce_filename}{self.filename_suffix}_frame{frame + 1:03d}_chi.txt")
                filenamei = os.path.join(outdir,f"{self.reduce_filename}{self.filename_suffix}_frame{frame + 1:03d}_i.txt")
               
                q=results['qmap'][frame];chi=results['chimap'][frame];i=results['imap'][frame]
                np.savetxt(filenameq,q)
                np.savetxt(filenamechi,chi)
                np.savetxt(filenamei,i)
                


    def plot_data(self):
        """Save plots combining Basler image, Eiger mean image, and integration results.
        
        Args:
            integration_results (dict): Dictionary with integration data per sector.
                - Keys: (angle, delta_angle) tuples.
                - Values: dictionaries containing 'q', 'i_raw', and 'i_normalize'.
            normalized (bool): If True, plot normalized intensity; else, plot raw intensity.
            basler_coords (tuple): Coordinates (x, y) to mark on the Basler image.
            output_filename (str, optional): Custom filename template.
        """
        integration_results=self.integrate()
        self.filename_suffix = "_integ_norm" if self.normalized else "_integ_raw"
        self.intensity_key = "i_normalize" if self.normalized else "i_raw"
        plots_folder = os.path.join(self.folder, "plots")
        os.makedirs(plots_folder, exist_ok=True)
        for frame_idx in range(self.nb_frames):
            filename = os.path.join(
                    plots_folder,
                    f"{self.reduce_filename}{self.filename_suffix}_frame{frame_idx + 1:03d}.png"
                )
            self._create_plot(self.integrate(),frame_idx,plots_folder,filename)

    def _create_plot(self,integration_results,frame_idx,plots_folder,output_filename):
        """ 
        function to plot all sectors on same figure instead of single sector integration on each file
        """
        # Generate filename components based on parameters
        base_name = self.reduce_filename
        suffix = "_norm" if self.normalized else "_raw"
                
        if frame_idx:
            frame_info = f"frame{frame_idx:03d}"
            default_name = f"{base_name}_{frame_info}_{suffix}.png"
        else:
            default_name = f"{base_name}_{suffix}.png"
        
        if output_filename:
            final_filename = output_filename.format(
                frame=frame_idx or 0
            )
        else:
            final_filename = os.path.join(plots_folder, default_name)
        if self.cavingtag==False:
            # Create the figure with three subplots
            fig, axes = plt.subplots(1, 3, figsize=(12, 5))
            
            # Panel 1: Display the Basler image and mark the measurement point.
            axes[0].imshow(self.basler_image, cmap='gray')
            if self.mean:
                axes[0].scatter(self.basler_coords[0], self.basler_coords[1], color='red')
            else:
                pos_x=self.basler_coords[0]+frame_idx*self.step_x
                pos_z=self.basler_coords[1]+frame_idx*self.step_z
                axes[0].scatter(int(pos_x), int(pos_z), color='red')

            axes[0].set_title('Basler Image')
            axes[0].axis('off')
            
            # Panel 2: Display the Eiger mean image and overlay the chi map.
            # Compute the chi map in degrees.
            chi_array_deg = np.degrees(self.chi_array)  # Convert from radians to degrees
            chi_array_deg = (chi_array_deg + 180) % 360 - 180  # Recenter to [-180, 180]
            # Create a mask for the current sector (between angle-delta_angle and angle+delta_angle)
            mask=[]
            for (angle,delta_angle),data in integration_results.items():
                mask.append( (chi_array_deg > angle - delta_angle) & (chi_array_deg < angle + delta_angle))
            axes[1].imshow(np.log1p(self.eiger[frame_idx]), cmap='jet')
            axes[1].set_title('Eiger Mean Image')
            axes[1].axis('off')
            # Overlay the chi map mask (displayed in gray)
            for k in range(len(mask)):
                axes[1].imshow(mask[k].astype(float), cmap='gray', alpha=0.1)

            # Panel 3: Plot the integrated intensity as a function of q in log-log scale.
            for (angle, delta_angle), data in integration_results.items():
                q=data["q"][frame_idx]
                i=data[self.intensity_key][frame_idx]
                axes[2].loglog(q, i,label=f'Sector:{angle}±{delta_angle}')
            axes[2].set_xlabel('q (Å⁻¹)')
            axes[2].set_ylabel('Intensity')
            plot_title = f"{'Normalized' if self.normalized else 'Raw'} Integration"
            if frame_idx:
                plot_title += f" - Frame {frame_idx}"
            axes[2].set_title(plot_title)
            axes[2].legend()
            fig.suptitle(f"{self.sample_name}\nSector: {angle}±{delta_angle}°")
            plt.tight_layout()
            
            # Save the figure to file and close the figure to free memory.
            fig.savefig(final_filename, dpi=300)
            plt.close(fig)
        else:
            # Create the figure with four subplots 
            # # the 4th panel is the 2D map
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            # Panel 1: Display the Basler image and mark the measurement point.
            axes[0,0].imshow(self.basler_image, cmap='gray')
            if self.mean:
                axes[0,0].scatter(self.basler_coords[0], self.basler_coords[1], color='red')
            else:
                pos_x=self.basler_coords[0]+frame_idx*self.step_x
                pos_z=self.basler_coords[1]+frame_idx*self.step_z
                axes[0,0].scatter(int(pos_x), int(pos_z), color='red')

            axes[0,0].set_title('Basler Image')
            axes[0,0].axis('off')
            
            # Panel 2: Display the Eiger mean image and overlay the chi map.
            # Compute the chi map in degrees.
            chi_array_deg = np.degrees(self.chi_array)  # Convert from radians to degrees
            chi_array_deg = (chi_array_deg + 180) % 360 - 180  # Recenter to [-180, 180]
            # Create a mask for the current sector (between angle-delta_angle and angle+delta_angle)
            
            mask=[]
            for (angle,delta_angle),data in integration_results.items():
                mask.append( (chi_array_deg > angle - delta_angle) & (chi_array_deg < angle + delta_angle))
            axes[0,1].imshow(np.log1p(self.eiger[frame_idx]), cmap='jet')
            axes[0,1].set_title('Eiger Mean Image')
            axes[0,1].axis('off')
            # Overlay the chi map mask (displayed in gray)
            for k in range(len(mask)):
                axes[0,1].imshow(mask[k].astype(float), cmap='gray', alpha=0.1)

            # Panel 3: Plot the integrated intensity as a function of q in log-log scale.
            for (angle, delta_angle), data in integration_results.items():
                q=data["q"][frame_idx]
                i=data[self.intensity_key][frame_idx]
                axes[1,0].loglog(q, i,label=f'Sector:{angle}±{delta_angle}')
            axes[1,0].set_xlabel('q (Å⁻¹)')
            axes[1,0].set_ylabel('Intensity')
            plot_title = f"{'Normalized' if self.normalized else 'Raw'} Integration"
            if frame_idx:
                plot_title += f" - Frame {frame_idx}"
            axes[1,0].set_title(plot_title)
            axes[1,0].legend()

            # Panel 4 plot 2D integration map
            res2D=self.integrate2D()
            qmap=res2D['qmap'][frame_idx];chimap=res2D['chimap'][frame_idx];i=res2D['imap'][frame_idx]
            axes[1,1].set_xscale('log')
            axes[1,1].set_xlabel('q $\\AA^{-1}$')
            axes[1,1].set_ylabel('Azimuhtal angle (°)')
            cax=axes[1,1].imshow(np.log1p(i),origin="lower",extent=[qmap.min(),qmap.max(),chimap.min(),chimap.max()],aspect="auto",cmap='jet')


            plt.tight_layout()
            
            # Save the figure to file and close the figure to free memory.
            fig.savefig(final_filename, dpi=300)
            plt.close(fig)
    
    
    def plot_and_save(self):
        self.save_integration()
        self.plot_data()
  

class SwingBatch:

    def __init__(self, 
                data_dir: str,
                mask_path:str,
                anisotropic_data: bool =False,
                average_frames: bool =True,
                transmission_correction: bool =True, 
                basler_coords: tuple =[0, 0],
                basler_calibration: tuple =[3.73,3.67],
                log_file: str="processing_log.txt",
                sectors: list =None):
        """
        SwingBatch class to process a batch of HDF5 files from synchrotron experiments.
        
        Args:
            data_dir (str): Directory containing the HDF5 files.
            mask_path (str): Path to the mask file.
            anisotropic_data: bool Set True to use data caving (image reconstruction by inversion symmetry, time consuming)
            average_frames: bool If True, integrate the mean image; if False, integrate each frame individually.
            transmission_correction: bool Set True to perform transmission correction and intensity normalization, 
            basler_coords: tuple =(0, 0), pixel coordiantes of X-ray beam on balser image
            log_file: str="processing_log.txt", 
            sectors: list of tuples): List of sectors defined as (angle, delta_angle) in degrees for integration.
                                      Default is [(0, 180)].
            
        """
        print('------------ Process Start ------------')
        self.sectors = sectors if sectors is not None else [(0, 180)]  # Default value for sectors
        self.data_dir = data_dir
        self.mask_path = mask_path
        self.sectors = sectors
        self.anisotropic=anisotropic_data
        self.integrate_mean = average_frames
        self.normalize = transmission_correction
        self.basler_coords = basler_coords
        self.basler_calibration=basler_calibration
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
        data = SwingData(file_path, 
        self.mask_path, 
        sectors=self.sectors,
        basler_coords=self.basler_coords,
        basler_calibration=self.basler_calibration,
        anisotropic_data=self.anisotropic,
        average_frames=self.integrate_mean,
        transmission_correction=self.normalize)
        
        data.save_integration()
        data.plot_data()

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

    def create_video_from_images(self, output_filename="output_video.mp4", fps=3):
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

