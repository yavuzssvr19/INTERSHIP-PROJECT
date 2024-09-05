## Brain Connectivity Analysis with SAR and LAM Models

This repository contains Jupyter notebooks that analyze brain connectivity using Spatial Autoregressive (SAR) and Laplacian Autoregressive (LAM) models at different brain resolution levels. This project was developed as part of my Erasmus+ internship at the Institute of Computational Neuroscience (ICNS), University Medical Center Hamburg-Eppendorf, where I had the opportunity to work alongside researchers focused on cutting-edge brain connectivity studies.

In this project, I utilized the YANAT library, which was developed by the ICNS, for model fitting and analysis. One key component of this library is the simple_fit function, which I updated and modified to meet the specific requirements of this project. My contributions include implementing necessary changes to this function to optimize the fitting of the SAR and LAM models. These updates are now reflected in the YANAT library, enabling more effective use in future projects.

Through this work, I aim to provide insights into how different normalization techniques can affect the accuracy and correlation of these models, ultimately enhancing our understanding of the brain's structural and functional connectivity (FC). Before diving into the project, I thoroughly reviewed the existing literature by reading 5-6 relevant research papers to familiarize myself with the field, ensuring that the methodologies and models used here are aligned with the latest findings from ICNS and the broader neuroscience community.

With this work, I hope to contribute to ongoing research at ICNS, and help others draw meaningful conclusions from similar datasets or related brain connectivity studies.

## Table of Contents
- [Overview](#overview)
- [Notebooks](#notebooks)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
Understanding brain connectivity is a crucial aspect of neuroscience. This project investigates the effects of `res_parcellation` values on brain functional connectivity (FC) matrices and aims to identify the optimal model and normalization techniques for accurately modeling human brain connections.

Two main models are used:

- **SAR (Spatial Autoregressive) Model**
- **LAM (Laplacian Autoregressive) Model**

These models are applied to functional and structural brain connectivity matrices to uncover which resolution and normalization techniques provide the best correlation with the target human brain FC matrix.

## Notebooks

### Notebook 1: SAR and LAM Models Brain Discovery at Different Resolutions
- Focuses on comparing SAR and LAM models across five different resolution (`res_parcellation`) values.
- Uses **log min-max normalization** to preprocess connectivity matrices.
- Fits SAR and LAM models and evaluates their performance based on correlation with the human FC matrix.

### Notebook 2: A Comparative Analysis of Normalization Techniques and SAR/LAM Models
- Compares various normalization techniques (log min-max, min-max, binarizing, spectral normalization) and their impact on SAR and LAM models.
- The notebook determines the best normalization technique for each model.
- Visualizes the brain network using heatmaps and plots.

## Dataset

The analysis is based on the **Human Connectome Dataset**, specifically the **Consensus Connectomes** dataset, which includes:

- **Structural Connectivity (SC) Matrix**
- **Functional Connectivity (FC) Matrix**
- **Fiber Lengths**
- **Coordinates** of brain regions
- **Labels and Modules**

## Requirements

To run the notebooks, you will need the following Python libraries:

- `numpy`
- `matplotlib`
- `scipy`
- `seaborn`
- `yanat` (That library created by the Institute)
- `warnings`

## Usage

Open the two main notebooks:

- analaysis_of_models.ipynb
- best_resolution_value.ipynb

## Results

Both notebooks contain detailed visualizations such as:

- Heatmaps of connectivity matrices
- Bar charts comparing model performance
- 3D brain network plots for different resolutions

The main findings include:

- The **log min-max normalization** technique provides the best correlation for both SAR and LAM models.
- **Res_parcellation = 0** yields the highest correlation for both models, making it the optimal resolution for further analysis.

## Contributing

Contributions are welcome! If you find a bug or want to propose an enhancement, feel free to open an issue or a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.