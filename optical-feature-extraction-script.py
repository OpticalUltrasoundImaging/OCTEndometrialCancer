# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 00:54:05 2025

@author: Yixiao Lin
"""
import os
from pathlib import Path
import logging
log = logging.getLogger(__package__)
log.setLevel(os.environ.get("LOGLEVEL", "INFO"))

import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import CubicSpline, interp1d
import numba as nb
from numba import njit
import math
import cv2

N_SAMPLES = 6144
def load_fringe_bin_YL(file: Path | str, n_alines: int):
    try:
        
        fringe = np.fromfile(file, dtype=np.uint16)
        total_elements = fringe.size
        n_bscan = total_elements // (n_alines * N_SAMPLES)        
        fringe = fringe.reshape((n_bscan, n_alines, N_SAMPLES), order="C")

    except Exception as e:
        log.critical(f"Failed to load {str(file)}:")
        log.critical(e)
        raise e
    return fringe

def load_fringe_bin_seg_YL(filepath: str, n_alines: int, n_bscan_to_load: int, i_start = int)->np.ndarray:
    elements_per_bscan = n_alines * N_SAMPLES
    bytes_per_element = 2  # uint16
    total_elements = n_bscan_to_load * elements_per_bscan
    offset = i_start * elements_per_bscan * bytes_per_element
    try:
        with open(filepath, 'rb') as f:
            f.seek(offset, 0)
            raw = np.fromfile(f, dtype=np.uint16, count=total_elements)
        actual_n_bscan_loaded = raw.size // elements_per_bscan
        if actual_n_bscan_loaded < n_bscan_to_load:
            print(f"Warning: only {actual_n_bscan_loaded} scans available (requested {n_bscan_to_load})")
        raw = raw[: actual_n_bscan_loaded * elements_per_bscan]
        fringe = raw.reshape((actual_n_bscan_loaded, n_alines, N_SAMPLES)).astype(np.float32)
    except Exception as e:
        print(f"Error loading {filepath}:\n{e}")
        raise

    return fringe


#%% MUT_EXTRACTION + SOCT
def stft_aline_YL(fringe: np.ndarray , N_spectrum: int , win_overlap: float , window_type: str = 'hann')-> np.ndarray:
    n_samples = fringe.shape[0]
    imagedepth = n_samples // 2 + 1
    
    win_len = (n_samples * 2) // (N_spectrum + 1)
    hop_size = int(win_len * win_overlap)
    n_frames = (n_samples - win_len) // hop_size + 1
    
    win = signal.get_window(window_type, win_len)
    fft_fringe = []

    for i in range(n_frames):
        start = i * hop_size
        end = start + win_len
        if end > n_samples:
            break
        segment = fringe[start:end] * win
        padded = np.pad(segment, (0, n_samples - win_len), mode='constant')
        fft = np.fft.ifft(padded, norm="backward")
        fft_mag = np.abs(np.real(fft[:imagedepth])) + np.abs(np.real(np.flip(fft[-imagedepth:], axis=0)))
        # fft_mag = signal.savgol_filter(fft_mag, window_length=9, polyorder=3)
        fft_fringe.append(fft_mag)
    return np.array(fft_fringe) # -> (N_spectrum , imagedepth)
    

def find_surface_bscan_automatic_YL(img_log_ori, tube_depth):
    img_log = img_log_ori.copy()
    Nz , Nx = img_log.shape
    img_log[:tube_depth,:]=0
    
    ub= 0.8
    np.clip(img_log, None, ub, out=img_log)
    img_log /= ub
    
    shrink_factor = 5
    img_log_small = cv2.resize(img_log.astype(np.float32), (Nx//shrink_factor , Nz//shrink_factor))
    img_log_blur = cv2.bilateralFilter(img_log_small,7,2,2)
    
    ret, _ = cv2.threshold(cv2.normalize(img_log_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    otsu_mask = img_log_blur > ((ret - 5) / 255.0)

    top_surface = np.argmax(otsu_mask, axis=0)
    all_zero = ~otsu_mask.any(axis=0)
    top_surface[all_zero] = -1
    
    top_surface = top_surface[np.newaxis, :].astype(np.float32)               # shape (1,Nx)
    top_surface = cv2.medianBlur(top_surface, 5).flatten()
    
    gaps = top_surface == -1
    indices = np.arange(len(top_surface))
    spline = CubicSpline(indices[~gaps], top_surface[~gaps])
    top_surface = spline(indices)*shrink_factor    

    top_surface = np.clip(top_surface, tube_depth + 5, Nz)
    interpolator = interp1d(np.arange(0,len(top_surface)),top_surface,kind='cubic')
    top_surface = interpolator(np.linspace(0,len(top_surface)-1,Nx))
    top_surface = signal.savgol_filter(top_surface, window_length=15, polyorder=3)
    
    img_binary = cv2.resize(otsu_mask.astype(np.uint8), (Nx, Nz), interpolation=cv2.INTER_AREA)
    rows = np.arange(Nz)[:, None]
    thresh = np.round(top_surface)[None, :]    # shape (1, Nx)
    mask = rows < thresh               # shape (Nz, Nx)
    img_binary[mask] = 0
    return top_surface.astype(np.int32) , img_binary

@njit(parallel=True, fastmath=True)
def compute_scattering_power(I_spectrum , tube_depth):
    N_s, full_depth, J = I_spectrum.shape

    # build wavelength vector once
    wavelengths = np.empty(N_s, dtype=np.float32)
    for si in range(N_s):
        wavelengths[si] = 1.22 + (1.40 - 1.22) * si / (N_s - 1)

    # precompute regression denominators
    Sx   = wavelengths.sum()
    Sxx  = (wavelengths * wavelengths).sum()
    denom_lin = N_s* Sxx - Sx * Sx

    # allocate outputs
    I_lambdac = np.empty(J, dtype=np.float32)
    k_line     = np.empty(J, dtype=np.float32)
    I_t = I_spectrum.transpose(2, 1, 0)

    eps = 1e-12

    for j in nb.prange(J):
        # slice out this A‑line: shape (full_depth, N_s)
        I_line = I_t[j]

        # 1) compute noise threshold from last 20 depths
        sum_noise = 0.0
        cnt_noise = 0
        for di in range(full_depth-20, full_depth):
            for si in range(N_s):
                sum_noise += I_line[di, si]
                cnt_noise += 1
        thr = 0.7 * (sum_noise / (cnt_noise + eps))

        # 2) precompute log once per A‑line
        logI = np.empty((full_depth, N_s), dtype=np.float64)
        for di in range(full_depth):
            for si in range(N_s):
                logI[di, si] = math.log(I_line[di, si] + eps)

        # 3) scan depths, accumulate for median + decay
        #    we reuse two scalars per line
        sum_ratio = 0.0
        sum_k     = 0.0
        cnt       = 0
        for di in range(tube_depth+5 , full_depth):
            # test all‑channels > thr
            ok = True
            for si in range(N_s):
                if I_line[di, si] <= thr:
                    ok = False
                    break
            if not ok:
                continue

            # weighted‑avg λ at this depth
            num = 0.0
            den = 0.0
            for si in range(N_s):
                v = I_line[di, si]
                num += wavelengths[si] * v
                den += v
            ratio = num / (den + eps)
            sum_ratio += ratio

            # linear‑fit slope k = −(N Sxy − Sx Sy)/denom_lin
            Sy  = 0.0
            Sxy = 0.0
            for si in range(N_s):
                ly = logI[di, si]
                Sy  += ly
                Sxy += wavelengths[si] * ly
            k = -(N_s * Sxy - Sx * Sy) / (denom_lin + eps)
            sum_k += k

            cnt += 1

        # 4) finalize per‑A‑line
        if cnt > 0:
            I_lambdac[j] = sum_ratio / cnt
            k_line[j]     = sum_k     / cnt
        else:
            I_lambdac[j] = 1.31
            k_line[j]     = 0.0
    return I_lambdac, k_line+4.0

@njit(parallel=True, fastmath=True)
def compute_mut_auto(I , top_surface , img_binary):
    Nz, Nx = I.shape
    mut_line = np.empty(Nx, dtype=np.float32)
    Dz = 55
    hop = Dz // 5

    # precompute x, x_mean and denom for linear‐regression slope
    x = np.arange(Dz, dtype=np.float64)*7.5*1e-4
    x_mean = x.mean()
    denom = 0.0
    for i in range(Dz):
        denom += (x[i] - x_mean) * (x[i] - x_mean)

    eps = 1e-12

    for ix in nb.prange(Nx):
        z_start = int(top_surface[ix])
        if z_start < 0:
            mut_line[ix] = 0.0
            continue

        # find last '1' in this column
        last1 = -1
        for z in range(Nz-1, -1, -1):
            if img_binary[z, ix]:
                last1 = z
                break
        if last1 < 0:
            mut_line[ix] = 0.0
            continue

        z_end = last1 + hop
        if z_end > Nz - Dz:
            z_end = Nz - Dz

        sum_slope = 0.0
        cnt = 0

        z = z_start
        while z <= z_end:
            # compute mean of log‑segment
            y_mean = 0.0
            for k in range(Dz):
                y_mean += math.log(I[z+k, ix] + eps)
            y_mean /= Dz

            # compute covariance for slope
            num = 0.0
            for k in range(Dz):
                y = math.log(I[z+k, ix] + eps)
                num += (x[k] - x_mean) * (y - y_mean)

            slope = num / denom
            slope = -slope / 2.0

            if 0.0 < slope < 35.0:
                sum_slope += slope
                cnt += 1

            z += hop

        mut_line[ix] = (sum_slope / cnt) if cnt > 0 else 0.0
    return mut_line

def compute_mut_wrap(I_log , I , tube_depth):
    top_surface , img_binary = find_surface_bscan_automatic_YL(I_log, tube_depth)
    z_cf = 190
    z_r = 36
    beam_correction = 1/(((np.arange(0,600) - z_cf) / z_r)**2 + 1)
    I = I / beam_correction[:,np.newaxis]
    mut_line = compute_mut_auto(I , top_surface , img_binary)
    mut_line = signal.savgol_filter(mut_line, window_length=11, polyorder=3)
    return mut_line

def recon_bscan_soct_YL(fringe_bscan , calib: Calib , tube_depth, imagedepth=0):
    n_alines, n_samples = fringe_bscan.shape
    if imagedepth == 0:
        imagedepth = n_samples // 2 + 1  # rfft size

    dispdepth = 600    
    I = np.zeros((dispdepth, n_alines))
    N_spectrum = 9
    I_spectrum = np.zeros((N_spectrum , dispdepth , n_alines))
    
    win = HAMM
    background = _mean_axis0(fringe_bscan)
    for j in nb.prange(n_alines):
        fringe_sub = fringe_bscan[j, :] - background
        linear_k_fringe = (
            fringe_sub[calib.ss_idx] * calib.l_coeff + fringe_sub[calib.ss_idx + 1] * calib.r_coeff
        )
        fft_fringe = np.fft.ifft(linear_k_fringe * win, norm = "backward")
        fft_fringe = np.abs(np.real(fft_fringe[:imagedepth])) + np.abs(np.real(np.flip(fft_fringe[n_samples-imagedepth : n_samples])))
        I[:, j] = fft_fringe[:dispdepth]
              
        stft_fringe = stft_aline_YL(linear_k_fringe , N_spectrum , 0.5) # -> (N_spectrum , imagedepth)
        I_spectrum[:,:,j] = stft_fringe[:,:dispdepth]
    # I = SRBF_OCT_faster(I)
    I = SRBF_OCT_gpu(I)
    I_log = log_compress_auto_YL(I , dispdepth)
    
    L_display = calib.theory_alines
    dist_offset = get_distortion_offset(I_log, L_display)
    I_log = cv2.resize(I_log[:, : L_display + dist_offset], (L_display,dispdepth))
    I     = cv2.resize(I[:, : L_display + dist_offset], (L_display,dispdepth))
    I_spectrum_resized = np.empty((N_spectrum , dispdepth , L_display) , dtype = np.float32)
    for j in nb.prange(N_spectrum):
        I_j = np.squeeze(I_spectrum[j,:,:])
        I_spectrum_resized[j] = cv2.resize(I_j[:, : L_display + dist_offset], (L_display,dispdepth))
    I_lambdac, I_sp = compute_scattering_power(I_spectrum_resized , tube_depth)
    mut_line = compute_mut_wrap(I_log , I , tube_depth)
    
    param_fitting = np.empty((3,L_display) , dtype = np.float32)
    param_fitting[0] = mut_line
    param_fitting[1] = signal.savgol_filter(I_sp, window_length=11, polyorder=3)
    param_fitting[2] = signal.savgol_filter(I_lambdac, window_length=11, polyorder=3)
    return I_log , param_fitting

def reconQSOCTSeg(idx_set, case_id , files , calib , N_bscan_segment , i_seg , prev_bscan = None , surface_start_threshold = 131):
    i_start = i_seg * N_bscan_segment
    fringes = load_fringe_bin_seg_YL(files[idx_set] , calib.n_alines , N_bscan_segment , i_start)
    fringes = fringes - calib.background.reshape(1,1,N_SAMPLES)
    N_bscan_loaded = fringes.shape[0] # MAY NOT EQUAL N_BSCAN_SEGMENT
    L_display = calib.theory_alines
    depth_last = 600
    img_all   = np.zeros((N_bscan_loaded , depth_last , L_display), dtype=np.float32)
    param_all = np.zeros((N_bscan_loaded , 3 , L_display), dtype=np.float32)
    # surface_start_threshold = 131
    
    loop_desc = f"Processing B-scan data set: {idx_set}, segment: {i_start}"
    for idx_bscan in tqdm(range(N_bscan_loaded) , desc=loop_desc , leave = False):
        fringe_bscan = np.squeeze(fringes[idx_bscan,:,:])
        
        I_log , param_fitting = recon_bscan_soct_YL(fringe_bscan , calib , surface_start_threshold)
        img_all[idx_bscan]   = I_log
        param_all[idx_bscan] = param_fitting
 
    if prev_bscan is not None:
        # print("Matching this segment to the end of last segment.")
        scan_curr = img_all[0]
        param_curr= param_all[0]
        align_offset = round(cv2.phaseCorrelate(prev_bscan[50:220,:], scan_curr[50:220,:])[0][0])
        img_all[0]   = np.roll(scan_curr , -align_offset , 1)
        param_all[0] = np.roll(param_curr, -align_offset , 1)
               
    for idx_bscan in tqdm(range(1,N_bscan_loaded) , desc="Aligning B scans" , leave = False):
        scan_prev = img_all[idx_bscan-1]
        scan_curr = img_all[idx_bscan]
        param_curr = param_all[idx_bscan]
        align_offset = round(cv2.phaseCorrelate(scan_prev[50:220,:], scan_curr[50:220,:])[0][0])
        
        img_all[idx_bscan]   = np.roll(scan_curr  , -align_offset , 1)
        param_all[idx_bscan] = np.roll(param_curr , -align_offset , 1)
        
    # np.savez(os.path.join(os.getcwd() , case_id+'_'+str(idx_set)+'_'+str(i_seg)+'_parameters') , b_scans = img_all , parameters = param_all)
    np.savez(os.path.join(r"C:\Users\JINHU\Desktop\QSOCT" , case_id+'_'+str(idx_set)+'_'+str(i_seg)+'_parameters') , b_scans = img_all , parameters = param_all)
    print("Segment results saved")
    return img_all[-1] # RETURNS THE LAST B SCAN FOR THE NEXT SEGMENT

def reconQSOCTSegWrap(idx_set , case_id , files , calib , N_bscan_segment , surface_start_threshold):
    '''
    PROCESS A SEQUENCE SEGMENT BY SEGMENT TO SAVE MEMORY
    '''
    filepath = files[idx_set]
    file_size = os.path.getsize(filepath)
    N_bscan_total = file_size // 2 // (calib.n_alines * N_SAMPLES)  # uint 16 = 2 bytes
    N_seg = int(np.ceil(N_bscan_total / N_bscan_segment))
    prev_bscan = None
    print(f"Processing B-scan data set: {idx_set}")
    for idx_segment in tqdm(range(N_seg) , desc = "Processing B scan segments" , leave = True):
        prev_bscan = reconQSOCTSeg(idx_set, case_id , files , calib , N_bscan_segment , idx_segment , prev_bscan , surface_start_threshold)
    return
