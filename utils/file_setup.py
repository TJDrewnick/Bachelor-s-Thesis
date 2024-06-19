# sets up all folders not in git for where plots and explainers are stored
import os

# create km folder
os.makedirs("../results/km/plots", exist_ok=True)
os.mkdir("../results/km/plots/diffusions")
os.mkdir("../results/km/plots/drift_diffusion_averaged")
os.mkdir("../results/km/plots/drift_diffusion_averaged_with_freq")
os.mkdir("../results/km/plots/drift_diffusion_sequential")
os.mkdir("../results/km/plots/frequency")
os.mkdir("../results/km/plots/histograms")

# create ml folders
os.makedirs("../results/ml_v1/explainers", exist_ok=True)
os.makedirs("../results/ml_v1/plots/boxplot", exist_ok=True)
os.mkdir("../results/ml_v1/plots/over_time")
os.mkdir("../results/ml_v1/plots/shap")

os.makedirs("../results/ml_v2/explainers", exist_ok=True)
os.makedirs("../results/ml_v2/plots/boxplot", exist_ok=True)
os.mkdir("../results/ml_v2/plots/over_time")
os.mkdir("../results/ml_v2/plots/shap")

os.makedirs("../results/ml_v2_knockout/explainers", exist_ok=True)
os.makedirs("../results/ml_v2_knockout/plots/boxplot", exist_ok=True)
os.mkdir("../results/ml_v2_knockout/plots/over_time")
os.mkdir("../results/ml_v2_knockout/plots/shap")
os.mkdir("../results/ml_v2_knockout/predictions")

os.makedirs("../results/ml_v2_random_noise/explainers", exist_ok=True)
os.makedirs("../results/ml_v2_random_noise/plots/boxplot", exist_ok=True)
os.mkdir("../results/ml_v2_random_noise/plots/over_time")
os.makedirs("../results/ml_v2_random_noise/plots/shap/full", exist_ok=True)
os.mkdir("../results/ml_v2_random_noise/predictions")

# create utils folders
os.makedirs("../results/utils/plots", exist_ok=True)