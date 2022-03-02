import numpy as np
from scipy.optimize import curve_fit

def convert_obj_score(ori_obj_score, MOS):
    """
    func:
        fitting the objetive score to the MOS scale.
        nonlinear regression fit
    """
    def logistic_fun(x, a, b, c, d):
        return (a - b)/(1 + np.exp(-(x - c)/abs(d))) + b
    # nolinear fit the MOSp
    param_init = [np.max(MOS), np.min(MOS), np.mean(ori_obj_score), 1]
    popt, pcov = curve_fit(logistic_fun, ori_obj_score, MOS, 
                           p0 = param_init, ftol =1e-8,  maxfev=500000)
    #a, b, c, d = popt[0], popt[1], popt[2], popt[3]
    
    obj_fit_score = logistic_fun(ori_obj_score, popt[0], popt[1], popt[2], popt[3])
    
    return obj_fit_score
