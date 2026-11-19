
import numpy as np
from scipy.stats import skew as scipy_skew

# Calibration coefficients (fit from 3,772 ground truth frames)
COEF_X = np.array([0.01558235, -0.99001140])
COEF_Y = np.array([-0.02103094, 1.40286245])
MIN_PIXELS = 20
SKEW_THRESHOLD = 0.15

def segment_lander(image):
    img = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    r = img[:,:,0].astype(int)
    g = img[:,:,1].astype(int)
    b = img[:,:,2].astype(int)
    mask = (b > 100) & (b > g + 20) & (r > 50) & (r < 220)
    ys, xs = np.where(mask)
    if len(xs) < MIN_PIXELS:
        return None
    return xs, ys, mask

def extract_xy(image):
    result = segment_lander(image)
    if result is None:
        return None
    xs, ys, mask = result
    cx = xs.mean()
    cy = ys.mean()
    x_hat = np.polyval(COEF_X, cx)
    y_hat = np.polyval(COEF_Y, cy)
    return x_hat, y_hat, len(xs)

def extract_theta(image):
    result = segment_lander(image)
    if result is None:
        return None
    xs, ys, mask = result
    if len(xs) < 10:
        return None
    cx, cy = xs.mean(), ys.mean()
    pts_math = np.stack([xs - cx, -(ys - cy)], axis=1).astype(np.float64)
    cov = np.cov(pts_math.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    narrow_axis = eigvecs[:, np.argmin(eigvals)]
    projections_math = pts_math @ narrow_axis
    s = scipy_skew(projections_math)
    if abs(s) >= SKEW_THRESHOLD:
        head_direction = narrow_axis if s > 0 else -narrow_axis
    else:
        along = projections_math
        pos_mask = along > 0
        neg_mask = along < 0
        avg_y_pos = (ys - cy)[pos_mask].mean() if pos_mask.sum() > 0 else 0
        avg_y_neg = (ys - cy)[neg_mask].mean() if neg_mask.sum() > 0 else 0
        head_direction = -narrow_axis if avg_y_pos > avg_y_neg else narrow_axis
    angle_math = np.arctan2(head_direction[1], head_direction[0])
    angle_sim = angle_math - np.pi/2
    angle_sim = np.arctan2(np.sin(angle_sim), np.cos(angle_sim))
    return angle_sim

def constraint_checker(image, true_x, true_y, true_theta):
    result = extract_xy(image)
    theta_hat = extract_theta(image)
    if result is None:
        return {
            "x_hat": None, "y_hat": None, "theta_hat": theta_hat,
            "x_err": None, "y_err": None, "theta_err": None,
            "visible": False, "theta_valid": theta_hat is not None
        }
    x_hat, y_hat, npix = result
    x_err = abs(x_hat - true_x)
    y_err = abs(y_hat - true_y)
    theta_err = None
    if theta_hat is not None and true_theta is not None:
        theta_err = abs(np.angle(np.exp(1j*(theta_hat - true_theta))))
    return {
        "x_hat": x_hat, "y_hat": y_hat, "theta_hat": theta_hat,
        "x_err": x_err, "y_err": y_err, "theta_err": theta_err,
        "visible": True, "theta_valid": theta_hat is not None,
        "pixel_count": npix
    }
