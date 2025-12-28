import numpy as np
import cv2
from scipy import ndimage

def anisodiff(img, n_iter=10, kappa=50, gamma=0.1, option=1):
    """
    Anisotropic diffusion (Perona-Malik)
    
    Parameters:
    - img: input image (grayscale, float32)
    - n_iter: number of iterations
    - kappa: conduction coefficient
    - gamma: integration constant (0 <= gamma <= 0.25 for stability)
    - option: 1 = Exponential, 2 = Quadratic
    """
    img = img.astype('float32')
    img_out = img.copy()
    
    # Delta (distances)
    dy, dx = 1, 1
    dd = np.sqrt(2)
 
    for _ in range(n_iter):
        # Calculate gradients (N, S, E, W, NE, NW, SE, SW)
        deltaN = np.roll(img_out, -1, axis=0) - img_out
        deltaS = np.roll(img_out, 1, axis=0) - img_out
        deltaE = np.roll(img_out, -1, axis=1) - img_out
        deltaW = np.roll(img_out, 1, axis=1) - img_out
        
        deltaNE = np.roll(np.roll(img_out, -1, axis=0), -1, axis=1) - img_out
        deltaNW = np.roll(np.roll(img_out, -1, axis=0), 1, axis=1) - img_out
        deltaSE = np.roll(np.roll(img_out, 1, axis=0), -1, axis=1) - img_out
        deltaSW = np.roll(np.roll(img_out, 1, axis=0), 1, axis=1) - img_out
 
        # Conduction coefficients
        if option == 1:
            cN = np.exp(-(deltaN/kappa)**2)
            cS = np.exp(-(deltaS/kappa)**2)
            cE = np.exp(-(deltaE/kappa)**2)
            cW = np.exp(-(deltaW/kappa)**2)
            cNE = np.exp(-(deltaNE/kappa)**2)
            cNW = np.exp(-(deltaNW/kappa)**2)
            cSE = np.exp(-(deltaSE/kappa)**2)
            cSW = np.exp(-(deltaSW/kappa)**2)
        elif option == 2:
            cN = 1./(1 + (deltaN/kappa)**2)
            cS = 1./(1 + (deltaS/kappa)**2)
            cE = 1./(1 + (deltaE/kappa)**2)
            cW = 1./(1 + (deltaW/kappa)**2)
            cNE = 1./(1 + (deltaNE/kappa)**2)
            cNW = 1./(1 + (deltaNW/kappa)**2)
            cSE = 1./(1 + (deltaSE/kappa)**2)
            cSW = 1./(1 + (deltaSW/kappa)**2)
 
        # Update image
        img_out += gamma * (
            (1/(dy**2)) * cN * deltaN + (1/(dy**2)) * cS * deltaS +
            (1/(dx**2)) * cE * deltaE + (1/(dx**2)) * cW * deltaW +
            (1/(dd**2)) * cNE * deltaNE + (1/(dd**2)) * cNW * deltaNW +
            (1/(dd**2)) * cSE * deltaSE + (1/(dd**2)) * cSW * deltaSW
        )
 
    return img_out

def process_hybrid_pipeline(image_pil):
    """
    Full pipeline: Anisotropic Diffusion -> Thresholding -> Morphology -> Contours
    Returns: Filtered Image, Tumor Mask, Bounding Box Image
    """
    # 1. Convert PIL to Grayscale Numpy
    img = np.array(image_pil.convert('L'))
    img = cv2.resize(img, (256, 256))
    
    # 2. Multi-Scale Adaptive Anisotropic Diffusion (MS-AADF)
    # Scale 1: Fine Edges (Low Kappa)
    filtered_fine = anisodiff(img, n_iter=15, kappa=15, gamma=0.15, option=2)
    
    # Scale 2: Medium Structures (Medium Kappa)
    filtered_med  = anisodiff(img, n_iter=15, kappa=30, gamma=0.15, option=2)
    
    # Scale 3: Noise Suppression (High Kappa)
    filtered_coarse = anisodiff(img, n_iter=15, kappa=60, gamma=0.15, option=2)
    
    # MS-AADF Fusion (Approximated Weighted Average)
    # Weights follow the logic: preserve edges (fine) > structures (med) > background (coarse)
    fused = (0.5 * filtered_fine) + (0.3 * filtered_med) + (0.2 * filtered_coarse)
    
    filtered_uint8 = np.clip(fused, 0, 255).astype('uint8')
    
    # 3. Adaptive Thresholding with Fallback Mechanism
    # Sometimes t0=60 is too strict. We try multiple sensitivity levels.
    sensitivity_levels = [60, 40, 20] 
    
    best_bbox = None
    best_mask = None
    best_cnt = None
    
    min_val = np.min(fused)
    max_val = np.max(fused)
    
    for t0 in sensitivity_levels:
        th = t0 + ((max_val + min_val) / 2)
        ret, binary_mask = cv2.threshold(filtered_uint8, th, 255, cv2.THRESH_BINARY)
        
        # 4. Morphological Operations
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_max_area = 0
        found_candidate = False
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100: # Filter small noise
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area) / hull_area
                    # Relaxed solidity check
                    if solidity > 0.5: 
                        if area > current_max_area:
                            current_max_area = area
                            best_cnt = cnt
                            best_mask = binary_mask
                            found_candidate = True
        
        if found_candidate:
            break # Stop if we found a good candidate at this sensitivity
            
    # Prepare final outputs
    final_mask = np.zeros_like(filtered_uint8)
    img_rgb = cv2.cvtColor(filtered_uint8, cv2.COLOR_GRAY2RGB)
    
    if best_cnt is not None:
        cv2.drawContours(final_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
        x,y,w,h = cv2.boundingRect(best_cnt)
        best_bbox = (x, y, w, h)
        
        # Visualization
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 255, 0), 2) # Cyan Box
        contours_poly = cv2.approxPolyDP(best_cnt, 3, True)
        cv2.drawContours(img_rgb, [contours_poly], -1, (255, 0, 0), 2) # Red Outline

    return filtered_uint8, final_mask, img_rgb, best_bbox
