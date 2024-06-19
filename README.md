# Bachelor's Thesis
### A Comparative Analysis of Drift and Diffusion in Power Grid Frequencies

The increasing integration of renewable energy sources such as wind and solar into power grids introduces significant variability and unpredictability, challenging the maintenance of the power grid stability. This stability is crucial for the reliability of power systems and the efficient operation of electrical devices. As the power grid frequency reflects the balance between electricity supply and demand, understanding its dynamics is crucial for managing the impact of renewable sources and ensuring grid stability.

To address these challenges, our research aims to estimate the drift (systematic changes) and diffusion (random fluctuations) within the grid frequency. The central question of our research explores the effectiveness of data-driven models in predicting and mitigating grid frequency fluctuations induced by the integration of renewable energy, thereby enhancing grid stability and reliability.

The planned approach for this research involves a comprehensive analysis of frequency data from various power grids, including the Continental European (CE) grid and the Australian (AUS) grid. The methodology centres on determining drift and diffusion coefficients using the Kramers-Moyal expansion for 1 hour time blocks across selected time intervals. Additionally, the research will employ simple models to predict drift and diffusion based on external features and examine features importance by using SHapley Additive exPlanations (SHAP).

## Using this code
The folder scripts contains a pipeline to calculate the main results and plots. The utils folder contains needed settings or additional files to create other results that did not fit in the pipeline or were used to find good parameters. 

### Data
The data is available at the following sources:
- CE frequency data (https://zenodo.org/records/5105820)
- CE feature data (https://zenodo.org/records/7273665)
- AUS frequency data (https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/system-operations/ancillary-services/frequency-and-time-deviation-monitoring)
- AUS feature data (https://opennem.org.au/energy/nem/?range=7d&interval=30m&view=discrete-time)

### Scripts
- 01_calculate_drift_diffusion.py: retrieves frequency data, filters out hours with missing values, converts it to angular velocity and plots it. Then calculates drift and diffusion for original and detrended data and stores each file.
- 02_plot_drift_diffusion.py: uses calculated results and filters outlier to save stats like mean and variance to file, plots the histogram and the drift/diffusion over time. Either as an average over the whole data in given intervals or as multiple adjacent intervals on top of each other.
- 03_prepare_features.py: retrieves feature data files, removes unneeded columns, averages the features per hour, filters outliers and adds time features for hour, day and month. Then merges it with the calculated drift and diffusion and stores the files
- 04_fit_models_drift_diffusion.py: optionally performs a grid search to find optimal hyperparameters, then fits each model (for original and detrended data: LightGBM Gradient Boosted Tree, XGBoost Gradient Boosted Tree), saves the results and parameters used, and creates and stores SHAP explainers
- 05_plot_ml_results.py: creates a box plot of the calculated results, plots the prediction over the whole dataset, and plots shap feature importances for each model

The v2 scripts contain the most up-to-date code after adding models but also removing prediction on original (not detrended) data. They add another XGBoost model using the absolute error loss function, a LightGBM Random Forest model and a Multi Layer Perceptron. They also add the functionality of adding random noise to the feature data as well as removing the most important feature from the feature set to analyse the impacts. 

### Utils
- clean_data.py: Cleaning of Australian data, CE data was pre cleaned
- file_setup.py: creates all needed folders for storing the created plots
- gaussian_filter_testing.py: Utility to visualize the trend that would be used for detrending depending on different sigma values
- helper_functions.py: Functions that are used by multiple scripts or utils. Currently only frequency data retrieval
- km_functions.py: A collection of all used kramers-moyal functions for drift and diffusion calculation
- ml_parameters.py: defining the parameters to be used by each ml-model when not performing a grid/random-search
- plot_correlation_drift_diffusion.py: calculate and plot the correlation between drift and diffusion 
- plot_correlation_load_with_drift_diffusion.py: calculate total load and plot correlation with drift and diffusion
- plot_deadbands.py: Plotting drift coefficient (1. Kramers Moyal coefficient) for random hours of the data as an overview
- plot_diffusion.py: Plotting diffusion coefficient (2. Kramers Moyal coefficient) for random hours of the data
- plot_drift_per_hour.py: Plotting the drift on original data for each hour to check for patterns
- plot_load_curves.py: Plotting of both load curves
- settings.py: File containing settings/shared parameters for all scripts and utils 
- validate_feature.py: Used to find features or patterns with missing data and validate correctness of Solar features for AUS data 

### Results
The results of the calculated drift and diffusion for original data as well as used model parameters and model predictions are included. Shap explainers as well as plots and results for modified features (random noise or knockout) are not included and can be calculated with the included scripts.