from sklearn.gaussian_process import GaussianProcessRegressor,kernels
from joblib import load
import pandas as pd
def construct_gaussian_kernel():
    rbf_kernel = kernels.RBF(1.0, length_scale_bounds=(1e-3, 1e5))*1.0
    noise_kernel = 1* kernels.WhiteKernel(noise_level=1, noise_level_bounds=(1e-1, 1e2))
    periodic_kernel = kernels.ExpSineSquared(length_scale=1, periodicity=1)
    full_kernel = rbf_kernel+noise_kernel*periodic_kernel
    return full_kernel
def construct_prediction_pipeline(pretrained_model=None):
    if not pretrained_model:
        raise NotImplementedError()
    else:
        return load(pretrained_model)
def predict_toxicity(features,pretrained_model=None):
    model = construct_prediction_pipeline(pretrained_model)
    predicted_toxicity = pd.DataFrame(model.predict(features), columns=["pred_pLC50"])
    return predicted_toxicity