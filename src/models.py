from sklearn.gaussian_process import GaussianProcessRegressor,kernels
from joblib import load

def construct_gaussian_kernel():
    rbf_kernel = kernels.RBF(1.0, length_scale_bounds=(1e-3, 1e5))*1.0
    noise_kernel = 1* kernels.WhiteKernel(noise_level=1, noise_level_bounds=(1e-1, 1e2))
    periodic_kernel = kernels.ExpSineSquared(length_scale=1, periodicity=1)
    full_kernel = rbf_kernel+noise_kernel*periodic_kernel
    return full_kernel
def construct_prediction_pipeline(pretrained_model=None):
    if not pretrained_model:
        clf1 = linear_model.PoissonRegressor(max_iter=10000)
        estimators = [('standard_scaler',StandardScaler()),
                # ('k_best', SelectKBest(f_regression)), 
                    ('sfs',SequentialFeatureSelector(clf1, direction="forward", cv = 3, n_jobs=-1,scoring="r2")),
                    ("gaussian_process_regressor",GaussianProcessRegressor(kernel=construct_gaussian_kernel(), random_state=1, alpha=0))               
                ]
        return Pipeline(estimators,memory="cache/")
    else:
        return load(pretrained_model)
def predict_toxicity(features,pretrained_model=None):
    model = construct_prediction_pipeline(pretrained_model)
    return model.predict(features)