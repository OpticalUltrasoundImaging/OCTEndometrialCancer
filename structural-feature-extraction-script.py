# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 00:48:35 2025

@author: Yixiao Lin
"""
import os
import numpy as np
from tqdm import tqdm
import cv2

#%% SOME UTIL FUNCTIONS
import torch
from scipy.spatial.transform import Rotation as R
from scipy import stats
from skimage.feature import graycomatrix, graycoprops
from numpy.fft import rfftn, irfftn
from numba import njit
from skimage.transform import radon
from scipy.stats import linregress, entropy
from scipy.interpolate import RegularGridInterpolator

def generate_3d_blob(sigma, rotation, shape=(32,32,32)):
    z, y, x = [np.linspace(-s//2, s//2, s) for s in shape]
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    coords_rot = coords @ rotation.T                     # (N,3)
    
    # 3) scale along each principal axis
    coords_scaled = coords_rot / np.array(sigma)         # broadcast divide
    
    # 4) evaluate Gaussian
    r2 = np.sum(coords_scaled**2, axis=1)
    blob = np.exp(-0.5 * r2).reshape(shape)
    
    # zero-mean & normalize
    blob -= blob.mean()
    return blob / np.sum(np.abs(blob))

def buildBlobDict(L_kernel=32):
    N_size = 4
    N_elongation = int(L_kernel/4)
    size_elongation_matrix = np.outer(np.arange(1,N_elongation+1) , np.array([2,4,8,16]))
    # BUILD SIGMA LIST
    sigma_list = []
    for i in range(N_size):
        for j in range(N_elongation):
            L_max_dim = size_elongation_matrix[j,i]
            if L_max_dim <= L_kernel:
                sigma_list.append((i+1,L_max_dim,i+1))
    # sigma_list.append((32,32,32))
    
    # BUILD ROTATION LIST
    n_theta, n_phi = 6, 6
    cos_t = np.linspace(-1, 1, n_theta)
    thetas = np.arccos(cos_t)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    phis, thetas = np.meshgrid(phis, thetas)
    angles = np.column_stack([phis.ravel(), thetas.ravel()])
    rotations = []
    for i in range(angles.shape[0]):
        rot = R.from_euler('ZY', np.squeeze(angles[i,:])).as_matrix()
        rotations.append(rot)

    blob_dict = []
    for sigma in sigma_list:
        for rot in rotations:
            blob = generate_3d_blob(sigma, rot)
            blob_dict.append(blob)        
    return blob_dict

def precompute_ffts(blob_dict , V_size = [79,128,128]):
    Fb_list = []
    for b in tqdm(blob_dict, desc="Computing blob FFTs"):
        pad = [(0, V_size[i] - b.shape[i]) for i in range(3)]
        b_padded = np.pad(b, pad, mode='constant')
        Fb_list.append(rfftn(b_padded))
    return Fb_list

@njit
def sum_abs_kernel(volume, norm):
    s = 0.0
    for x in volume.flat:
        s += abs(x)
    return s / norm

def bloblet_transform_from_ffts(Fv, Fb_list, norm):
    n_blobs = Fb_list.shape[0]
    out = np.empty(len(Fb_list), dtype=np.float64)
    for i in range(n_blobs):
        Fb = np.squeeze(Fb_list[i,:,:,:])
        conv = irfftn(Fv * Fb, norm="backward")
        out[i] = sum_abs_kernel(conv, norm)
    return out

def bloblet_transform_from_ffts_torch(Fv, Fb_list, norm="backward", chunk_size=128, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Fv_t = torch.as_tensor(Fv, device=device)
    outputs = []

    with torch.no_grad():
        # process filters in small batches
        for i in range(0, len(Fb_list), chunk_size):
            # load one chunk of filters
            Fb_chunk = torch.as_tensor(Fb_list[i : i + chunk_size], device=device)

            # multiply & inverse FFT
            prods = Fb_chunk * Fv_t.unsqueeze(0)
            convs = torch.fft.irfftn(
                prods,
                dim=tuple(range(1, prods.ndim)),
                norm=norm
            )

            out_chunk = convs.abs().flatten(1).sum(dim=1)
            outputs.append(out_chunk.cpu())
            del Fb_chunk, prods, convs, out_chunk
            torch.cuda.empty_cache()
    return torch.cat(outputs).numpy()

def blobletTransformParallel(V, Fb_list):
    # N = V.size
    Fv = rfftn(V)
    return bloblet_transform_from_ffts_torch(Fv, Fb_list)

def glcm_features_3d(vol: np.ndarray, distances: list = [1], angles: list = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                     levels: int = 64, symmetric: bool = True, normed: bool = True) -> dict:
    
    '''
    list: 
        histogram: [mean , std , skewness , kurtosis , energy, entropy]
        glcm: [contrast , dissimilarity , homogeneity , energy , correlation , ASM]
    '''
    features = np.empty(11,dtype=np.float32)
    vals = vol.ravel()
    minv, maxv = vals.min(), vals.max()
    
    features[0]  = stats.skew(vals)*255
    features[1]  = stats.kurtosis(vals)*255
    hist, bin_edges = np.histogram(vals, bins=levels, density=True)
    # hist_nonzero = hist[hist > 0]
    
    if maxv == minv:
        quant = np.zeros_like(vol, dtype=np.uint8)
    else:
        quant = np.floor((vol - minv) / (maxv - minv) * (levels-1)).astype(np.uint8)

    # prepare accumulator
    props = ['mean','std','contrast','dissimilarity','homogeneity','energy','entropy','correlation','ASM']
    acc = {p: [] for p in props}

    # 2) loop through slices
    for z in range(vol.shape[0]):
        img2d = quant[z]
        # 3) compute GLCM for this slice
        glcm = graycomatrix(img2d,
                            distances=distances,
                            angles=angles,
                            levels=levels,
                            symmetric=symmetric,
                            normed=normed)

        for p in props:
            # greycoprops returns array shape (len(distances), len(angles))
            mat = graycoprops(glcm, p)
            acc[p].append(mat.mean())

    # 5) average across slices
    idx_glcm = 2
    for p in props:
        features[idx_glcm] = float(np.mean(acc[p]))
        idx_glcm += 1
    return features

L_kernel = 32
blob_dict = buildBlobDict(L_kernel=L_kernel)
Fb_list = precompute_ffts(blob_dict)
Fb_list = np.asarray(Fb_list)
blob_detection_size = (22,36)

def destripe_fourier_notch(L, u=20, n=2):
    L = np.log1p(np.abs(L))
    M, N = L.shape
    cy, cx = M//2, N//2
    k_rows = np.arange(M) - cy
    H_row = 1.0 / (1.0 + (k_rows/u)**(2*n))
    k_cols = np.arange(N) - cx
    H_col = 1.0 / (1.0 + (k_cols/u)**(2*n))
    H2D = H_row[:,None] * H_col[None,:]
    Fclean = L * H2D
    return Fclean

def destripe_fourier_notch_raw(L, u=20, n=2):
    M, N = L.shape
    cy, cx = M//2, N//2
    k_rows = np.arange(M) - cy
    H_row = 1.0 / (1.0 + (k_rows/u)**(2*n))
    k_cols = np.arange(N) - cx
    H_col = 1.0 / (1.0 + (k_cols/u)**(2*n))
    H2D = H_row[:,None] * H_col[None,:]
    Fclean = L * H2D
    return Fclean

def fit_ellipse_to_spectrum(Fclean):
    # ROI PIXEL SIZE IS 9.5UM
    # THEN FOURIER SPECTRUM PIXEL SIZE DK = 1/9.5
    norm = (Fclean - Fclean.min())/Fclean.ptp()
    bw = (norm > 0.13).astype('uint8')*255
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(cnt)
    (xc, yc), (major, minor), angle = ellipse
    major_axis_length = major/9.5
    minor_axis_length = minor/9.5
    if minor > major:
        angle += 90
    l_major = max(major_axis_length , minor_axis_length)
    l_minor = min(major_axis_length , minor_axis_length)
    eccentricity = np.sqrt(1 - l_minor * l_minor / l_major / l_major)
    return np.array([l_major , l_minor , eccentricity , angle])

def fit_ellipsoid_to_spectrum3d(F_tmp):
    '''
    ellipsoid :
        [l_maj_xy , l_min_xy , theta_xy,
         l_maj_zy , l_min_zy , theta_zy,
         l_maj_zx , l_min_zx , theta_zx]
        l [1/um]
        theta [deg]
        angles are measured from the positive dir of horizontal axis (1st index in subscript)
    '''
    Lz , Lx , Ly = F_tmp.shape
    cz , cx , cy = Lz//2 , Lx//2 , Ly//2
    F_xy_center = destripe_fourier_notch(np.squeeze(np.mean(abs(F_tmp[cz-1:cz+2]),axis=0)))
    F_zy_center = destripe_fourier_notch(np.squeeze(np.mean(abs(F_tmp[:,cx-1:cx+2,:]),axis=1)))
    F_zx_center = destripe_fourier_notch(np.squeeze(np.mean(abs(F_tmp[:,:,cy-1:cy+2]),axis=2)))
    
    ellipsoid = np.zeros((3,4))
    ellipsoid[0] = fit_ellipse_to_spectrum(F_xy_center)
    ellipsoid[1] = fit_ellipse_to_spectrum(F_zy_center)
    ellipsoid[2] = fit_ellipse_to_spectrum(F_zx_center)
    return ellipsoid

def Fcart2sph(F_roi_avg):
    dv = 9.5e-3
    roi_size = F_roi_avg.shape[1]
    kz = np.fft.fftshift(np.fft.fftfreq(F_roi_avg.shape[0], d=dv))
    ky = np.fft.fftshift(np.fft.fftfreq(roi_size, d=dv))
    kx = np.fft.fftshift(np.fft.fftfreq(roi_size, d=dv))

    N_theta = 128
    N_psi = 180
    r     = np.linspace(0, np.linalg.norm([kx.max(), ky.max(), kz.max()]), roi_size)
    theta = np.linspace(0, np.pi,   N_theta) # POLAR ANGLE
    psi   = np.linspace(0, 2*np.pi, N_psi) # AZIMUTHAL
    R, Theta, Psi = np.meshgrid(r, theta, psi, indexing='ij')

    # 4) spherical → Cartesian coordinates
    KX_s = R * np.sin(Theta) * np.cos(Psi)
    KY_s = R * np.sin(Theta) * np.sin(Psi)
    KZ_s = R * np.cos(Theta)

    # 5) interpolate :contentReference[oaicite:2]{index=2}
    interp = RegularGridInterpolator(
        (kz, ky, kx), F_roi_avg,
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    pts = np.stack([KZ_s.ravel(), KY_s.ravel(), KX_s.ravel()], axis=-1)
    F_roi_avg_spherical = interp(pts).reshape(R.shape)
    return F_roi_avg_spherical

def spherical_analysis_spectrum3d(F_tmp):
    Lz , Lx , Ly = F_tmp.shape
    F_denoise = F_tmp.copy()
    for i_z in range(Lz):
        F_denoise[i_z] = destripe_fourier_notch(np.squeeze(np.abs(F_denoise[i_z])))
    F_sph_denoise = Fcart2sph(F_denoise) # -> [N_kr , N_ktheta , N_kpsi] = [128,128,180]
    # RADIAL DECAY
    F_r = np.mean(np.mean(abs(F_sph_denoise) , axis=1),axis=1)
    F_r_log = np.log(F_r[:72] - min(F_r[:72]) + 1e-12)
    F_r_slope, _, _, _, _ = linregress(np.arange(72), F_r_log)
    
    # POLAR TREND
    F_theta = np.mean(np.mean(abs(F_sph_denoise) , axis=0) , axis=1)
    F_theta = F_theta[:64]
    F_theta_log = np.log(F_theta - min(F_theta) + 1e-12)
    F_theta_slope, _, _, _, _ = linregress(np.arange(64), F_theta_log)
    
    
    # AZIMUTHAL (EN FACE) TREND
    F_psi = np.mean(np.mean(abs(F_sph_denoise) , axis=0) , axis=0)
    F_psi = F_psi[:90]
    F_psi /= sum(F_psi)
    F_psi_entropy = entropy(F_psi)
    
    sphectrum = [F_r_slope , F_theta_slope , F_psi_entropy]
    return sphectrum

def bandpass_filter(F, k_cuton , k_cutoff , order = 2):
    M, N = F.shape
    u = np.fft.fftshift(np.fft.fftfreq(N) * N)
    v = np.fft.fftshift(np.fft.fftfreq(M) * M)
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)  
    H_low = 1.0 / (1.0 + (D / k_cutoff)**(2*order))
    H_high = 1.0 / (1.0 + (k_cuton / (D + 1e-12))**(2*order))
    H_band = H_low * H_high
    return F*H_band

def extract_circular_stats(radon_tmp):
    Nangle = radon_tmp.size
    theta = np.linspace(0, 2*np.pi, Nangle, endpoint=False)
    # dtheta = 2*np.pi / Nangle
    p = radon_tmp / np.sum(radon_tmp)
    
    C = np.sum(p * np.cos(theta))
    S = np.sum(p * np.sin(theta))
    mu = np.arctan2(S, C)                   # mean direction
    R = np.hypot(C, S)                    # resultant length
    V = 1 - R                             # circular variance
    circ_stdev = np.sqrt(-2 * np.log(np.clip(R, 1e-20, 1.0)))  # circular std dev
    # skewness & kurtosis
    skewness = np.sum(p * np.sin(theta - mu)) / (V**1.5 + 1e-20)
    kurtosis = np.sum(p * np.cos(2*(theta - mu))) / (V**2 + 1e-20)
    H = entropy(p, base=np.e)  
    return np.array([circ_stdev , skewness , kurtosis , H])

def radon_analysis_spectrum3d(F_tmp: np.ndarray):
    Lz , Lx , Ly = F_tmp.shape
    cz , cx , cy = Lz//2 , Lx//2 , Ly//2
    Nangle = np.max([Lz,Lx,Ly])
    theta = np.linspace(0., 180., Nangle, endpoint=False)
    radonum = np.zeros((3,4))
    Radon_xy_center = radon(bandpass_filter(destripe_fourier_notch(np.squeeze(np.mean(abs(F_tmp[cz-1:cz+2]),axis=0))) , 3 , 30) , theta=theta , circle=False)
    radon_tmp = Radon_xy_center[Radon_xy_center.shape[0]//2,:]
    radonum[0] = extract_circular_stats(radon_tmp)
    
    Radon_zy_center = radon(bandpass_filter(destripe_fourier_notch(np.squeeze(np.mean(abs(F_tmp[:,cx-1:cx+2,:]),axis=1))) , 3 , 30) , theta=theta , circle=False)
    radon_tmp = Radon_zy_center[Radon_zy_center.shape[0]//2,:]
    radonum[1] = extract_circular_stats(radon_tmp)
    
    Radon_zx_center = radon(bandpass_filter(destripe_fourier_notch(np.squeeze(np.mean(abs(F_tmp[:,:,cy-1:cy+2]),axis=2))) , 3 , 30) , theta=theta , circle=False)
    radon_tmp = Radon_zx_center[Radon_zx_center.shape[0]//2,:]
    radonum[2] = extract_circular_stats(radon_tmp)
    return radonum.reshape((12,))

def minnorm(S_tmp: np.ndarray , axis: int) -> np.ndarray:
    if axis == 0:
        col_mins = S_tmp.min(axis=0, keepdims=True) + 1e-12
        I_norm = S_tmp / col_mins
    elif axis == 1:
        row_mins = S_tmp.min(axis=1, keepdims=True) + 1e-12
        I_norm = S_tmp / row_mins
    else:
        raise ValueError(f"Invalid axis={axis}. Must be 0 (columns) or 1 (rows).")
    return I_norm

#%% CROP ROIS FROM ANTERIOR
roi_size = 128
disp = np.squeeze(cscan_resize[:,35,:]).copy()  

orig_h, orig_w = disp.shape[:2]
MAX_W, MAX_H = int(orig_w/2), int(orig_h/2)
scale = min(MAX_W/orig_w, MAX_H/orig_h, 1.0)

preview = cv2.resize(disp, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
cv2.imshow('preview', preview)
cv2.waitKey(1)

rois = []

# 4) repeatedly select on the preview window
while True:
    x_p, y_p, w_p, h_p = cv2.selectROI('preview', preview, showCrosshair=True)
    if w_p == 0 and h_p == 0:
        break

    w_p_fixed, h_p_fixed = roi_size, roi_size    
    x0 = int(round(x_p/scale))
    y0 = int(round(y_p/scale))
    w0 = int(round(w_p_fixed/scale))
    h0 = int(round(h_p_fixed/scale))
    x0 = max(0, min(x0, orig_w - w0))
    y0 = max(0, min(y0, orig_h - h0))
    rois.append((x0, y0))
    cv2.rectangle(preview, (x_p, y_p), (x_p + w_p_fixed, y_p + h_p_fixed), (0,0,255), 3)
    cv2.imshow('preview', preview)
    cv2.waitKey(1)
cv2.destroyAllWindows()
print("ROIs in original-image coords (x,y,w,h):")
for r in rois:
    print(r)

N_roi = len(rois)
# roi_log = np.empty((N_roi , roi_size , roi_size))
mut_log = np.empty((N_roi , roi_size , roi_size))
isp_log = np.empty((N_roi , roi_size , roi_size))
iwc_log = np.empty((N_roi , roi_size , roi_size))
F_roi_avg = np.zeros((cscan_resize.shape[1] , roi_size , roi_size), dtype=np.complex128)
B_roi_avg = np.zeros(blob_detection_size)
R_roi_log = np.empty((N_roi , 11))

for idx_roi in tqdm(range(N_roi) , desc="COMPUTING ANTERIOR FEATURES"):
    roi_coordinates = rois[idx_roi]
    # roi_log[idx_roi] = cscan_resize[roi_coordinates[1]:roi_coordinates[1]+roi_size , roi_coordinates[0]:roi_coordinates[0]+roi_size]
    mut_log[idx_roi] = mut_resize[roi_coordinates[1]:roi_coordinates[1]+roi_size , roi_coordinates[0]:roi_coordinates[0]+roi_size]
    isp_log[idx_roi] = isp_resize[roi_coordinates[1]:roi_coordinates[1]+roi_size , roi_coordinates[0]:roi_coordinates[0]+roi_size]
    iwc_log[idx_roi] = iwc_resize[roi_coordinates[1]:roi_coordinates[1]+roi_size , roi_coordinates[0]:roi_coordinates[0]+roi_size]
    
    roi_vol = cscan_resize[roi_coordinates[1]:roi_coordinates[1]+roi_size , : , roi_coordinates[0]:roi_coordinates[0]+roi_size]
    roi_vol = np.moveaxis(roi_vol , [0,1,2],[1,0,2])
    F_roi_vol = np.fft.fftn(roi_vol)
    F_roi_avg += F_roi_vol / N_roi
    
    F_blob_response = blobletTransformParallel(roi_vol, Fb_list)/L_kernel/L_kernel/L_kernel
    F_blob_response = F_blob_response.reshape(blob_detection_size)
    B_roi_avg += F_blob_response / N_roi
    
    R_roi_log[idx_roi] = glcm_features_3d(roi_vol)

F_roi_avg = np.fft.fftshift(F_roi_avg)
spectrum_all = np.concatenate((fit_ellipsoid_to_spectrum3d(F_roi_avg).reshape(-1) , np.array(spherical_analysis_spectrum3d(F_roi_avg)) , radon_analysis_spectrum3d(F_roi_avg))) # -> (27,), 12 + 3 + 12

filename_ext = os.path.basename(str(resultfiles[0]))
filename = os.path.splitext(filename_ext)[0]
np.savez(os.path.join(os.getcwd() , 'qsoct_results_rois_v2', filename + '_anterior_roi_results') , 
         roi_positions = rois , 
         cscan = cscan_resize[:,35,:] , 
         mut = mut_resize , 
         isp = isp_resize ,
         iwc = iwc_resize ,
         spatial_spectrum = spectrum_all ,
         bloblet_matching = B_roi_avg,
         radiomic_rois = R_roi_log ,
         mut_rois = mut_log ,
         isp_rois = isp_log ,
         iwc_rois = iwc_log)
print("results saved")