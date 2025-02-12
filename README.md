## Overview

The `SwingData` and `SwingBatch` classes are designed to extract, integrate, and plot synchrotron data from HDF5 files. They leverage libraries such as **pyFAI** for azimuthal integration and **fabio** for reading image data. The code also supports saving integration results to text files and generating summary plots and videos.

---

## Classes

### 1. SwingData

**Purpose:**

- Extract experimental parameters and images from an HDF5 file.
- Perform azimuthal integration on the Eiger detector data.
- Save integration results and generate plots.

**Key Methods:**

#### `extract_from_h5()`
Extracts parameters (sample name, detector images, beam center, pixel size, wavelength, etc.) and stores them in a dictionary (`self.params`).

#### `integrate(sectors=[(0, 180)], mean=False)`
Integrates either the mean image (`eiger_mean`) or all individual frames (`eiger`) over specified sectors.

**Parameters:**
- `sectors`: List of tuples `(angle, delta_angle)` specifying integration sectors in degrees.
- `mean`: If `True`, uses the mean image; if `False`, integrates each frame.

**Returns:**
- A dictionary of integration results with keys `(angle, delta_angle)` and values containing arrays for `q`, `i_raw`, and `i_normalize`.
- Computes a chi-map (`self.chi_array`) using the AzimuthalIntegrator.

#### `save_integration(integration_results, normalized=False)`
Saves the integration results to text files. Filenames include the sample’s reduced filename, sector information, and frame number if applicable.

**Parameters:**
- `integration_results`: The dictionary returned by `integrate()`.
- `normalized`: If `True`, saves normalized intensity (`i_normalize`); otherwise, saves raw intensity (`i_raw`).

#### `plot_data(integration_results, normalized=True, basler_coords=(0, 0), output_filename=None)`
Creates plots that combine:
- The **Basler image** (with a marker at the measurement point).
- The **Eiger mean image** with an overlay of the chi-map for the selected sector.
- A **log–log plot** of the integrated intensity versus *q*.

#### `_create_plot(...)` *(helper function)*
Used internally to generate and save individual plots based on the integration data. It displays three panels (Basler image, Eiger mean image with chi-map, and the integration plot) and saves the figure.

---

### 2. SwingBatch

**Purpose:**

- Process a batch of HDF5 files using the `SwingData` class.
- Provide progress feedback.
- Generate text file outputs for integration results.
- Create a video from the generated plot images.

**Key Methods:**

#### `get_h5_files()`
Returns a list of HDF5 files in the specified data directory.

#### `process()`
Iterates over all HDF5 files in the directory and processes each file (extraction, integration, saving results, plotting). Displays progress in the terminal or in a Jupyter Notebook.

#### `_process_single_file(file_path)`
Processes one file by instantiating `SwingData`, performing integration, saving the results, and generating plots.

#### `_log_results()`
Writes a log file summarizing the processing (which files succeeded and which failed, with error details).

#### `get_report()`
Returns a text report summarizing the processing results.

#### `create_video_from_images(output_filename="output_video.mp4", fps=10)`
Combines the generated plot images (stored in the "plots" folder) into an MP4 video.

---

### 3. **TextFileProcessor**

**Purpose:**

- This class is designed to load, store, and process data from text files in the Foxtrot format.
- It supports loading multiple files with different `skiprows` values to find the correct data, retrieving stored data, and subtracting data from two files.

---

**Key Methods:**

#### `__init__(self)`
Initializes the `TextFileProcessor` object with an empty dictionary to store the file data.

**Parameters:**
- No parameters for initialization.

#### `load_txt(self, path)`
Loads q and i data from a Foxtrot-format text file, attempting different `skiprows` values.

**Parameters:**
- `path`: The file path of the text file to load.

**Returns:**
- `(q, i)`: The q and i data arrays if the file is successfully loaded, otherwise `None, None`.

**Details:**
- The method tries different values of `skiprows` (from 0 to 99) to load the file. If successful, it stores the data in a dictionary (`self.files`), with the file path as the key and the `(q, i)` tuple as the value.

#### `get_data(self, path)`
Retrieves the stored q and i data for a given file path.

**Parameters:**
- `path`: The file path of the text file.

**Returns:**
- `(q, i)`: The stored q and i data if the file is found, otherwise `(None, None)`.

#### `subtract_files(self, file1, file2)`
Subtracts data from two files and saves the result in a new file inside a `sub` subdirectory.

**Parameters:**
- `file1`: The first file to subtract from.
- `file2`: The second file to subtract.

**Returns:**
- None. The result is saved as a new text file in the `sub` folder.

**Details:**
- The method retrieves the data from `file1` and `file2`. If the `q` values are different, it performs interpolation to match them.
- The subtraction of `i` values is done element-wise (`i1 - i2_interpolated`).
- The result is saved in a new text file, with the name formatted as `file1_name-file2_name_subtracted.txt`.

---
