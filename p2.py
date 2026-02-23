import cv2
import numpy as np
import time

# [add] function to analyze alignment and adjust brightness
def analyze_alignment(scanned_img, reference_path):
    ref_color = cv2.imread(reference_path)
    if ref_color is None: return scanned_img, False
    h_ref, w_ref = ref_color.shape[:2]
    h_scan, w_scan = scanned_img.shape[:2]
    
    gray_ref = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
    gray_scan = cv2.cvtColor(scanned_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Template Matching find the best horizontal alignment by matching the center portion of the reference image with the scanned image
    template_w = min(400, w_ref)
    template = gray_ref[:, w_ref//2 - template_w//2 : w_ref//2 + template_w//2]
    scan_triple = np.hstack((gray_scan, gray_scan, gray_scan))
    res = cv2.matchTemplate(scan_triple, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    
    # 2. calculate shift amount: determine how much to roll the scanned image horizontally so that the detected center in the triple scan aligns with the center of the reference image
    # pixel data from template matching is in the triple scan, so we need to mod by w_scan to get the corresponding x in the original scanned image
    cx_match_triple = max_loc[0] + template_w // 2
    x_original = cx_match_triple % w_scan
    shift_amount = (w_ref // 2) - x_original
    aligned = np.roll(scanned_img, shift_amount, axis=1)
    
    # 3. resize aligned image to match reference width if needed (in case the slit scan didn't produce a full width image, we stretch it to match the reference width for better comparison)
    if aligned.shape[1] != w_ref:
        aligned = cv2.resize(aligned, (w_ref, h_scan), interpolation=cv2.INTER_LANCZOS4)

    # rebalance brightness by matching the mean color of the aligned image to the reference image, channel by channel
    for i in range(3):
        mean_ref = np.mean(ref_color[:,:,i])
        mean_aln = np.mean(aligned[:,:,i])
        ratio = mean_ref / (mean_aln + 1e-6)
        aligned[:,:,i] = np.clip(aligned[:,:,i] * ratio, 0, 255).astype(np.uint8)

    return aligned, True

# [add] function to show side-by-side comparison of aligned result and reference image
def show_comparison_window(aligned_img, reference_path):
    ref_color = cv2.imread(reference_path)
    if ref_color is None or aligned_img is None: return
    h, w = aligned_img.shape[:2]
    ref_resized = cv2.resize(ref_color, (w, h))
    comparison = np.hstack((ref_resized, aligned_img))
    scale = 1280 / comparison.shape[1]
    final_view = cv2.resize(comparison, (0, 0), fx=scale, fy=scale)
    cv2.imshow('MATCHING RESULT: [REF] vs [ALIGNED]', final_view)

# --- CONFIGURATION ---

TARGET_H, TARGET_W = 1080, 540 
current_slit_w = 0.5            
SCAN_DURATION = 20.075     
STRETCH_RATIO = 2.5             
LOCKED_FOCUS_VALUE = 300        
WAIT_BEFORE_SCAN = 1.0  

LOWER_GREEN = np.array([35, 40, 40]); UPPER_GREEN = np.array([85, 255, 255])


# --- CAMERA SETUP ---

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0); cap.set(cv2.CAP_PROP_FOCUS, LOCKED_FOCUS_VALUE)


# --- VARIABLES ---

unwrapped_img = None
is_scanning = False
scan_finished = False
start_time = 0
ready_start_time = 0  
is_waiting = False    

print(f">>> SYSTEM READY: วางแช่ไว้ {WAIT_BEFORE_SCAN} วิ เพื่อเริ่มสแกน")

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]; cx, cy = w // 2, h // 2
    
    # 1. check is center is green (UI)
    roi = frame[max(0, cy-10):min(h, cy+10), max(0, cx-10):min(w, cx+10)]
    avg_color_bgr = np.mean(roi, axis=(0, 1)).astype(np.uint8)
    avg_hsv = cv2.cvtColor(np.uint8([[avg_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    is_center_green = (LOWER_GREEN[0] <= avg_hsv[0] <= UPPER_GREEN[0]) and \
                      (LOWER_GREEN[1] <= avg_hsv[1] <= UPPER_GREEN[1]) and \
                      (LOWER_GREEN[2] <= avg_hsv[2] <= UPPER_GREEN[2])

    mask = cv2.bitwise_not(cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), LOWER_GREEN, UPPER_GREEN))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    can_present = False
    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_cnt) > 50000:
            can_present = True; x_b, y_b, w_b, h_b = cv2.boundingRect(max_cnt)

    # 3. Logic control for scanning process:
    if can_present and not is_center_green and not is_scanning and not scan_finished:
        if not is_waiting:
            is_waiting = True; ready_start_time = time.time()
        if time.time() - ready_start_time >= WAIT_BEFORE_SCAN:
            unwrapped_img = None; is_scanning = True; is_waiting = False; start_time = time.time()
            print(">>> [STARTED] Scanning...")
    else:
        if not is_scanning: is_waiting = False

    # 4. scanning process: 
    if is_scanning:
        pts1 = np.float32([[x_b, y_b], [x_b+w_b, y_b], [x_b, y_b+h_b], [x_b+w_b, y_b+h_b]])
        pts2 = np.float32([[0, 0], [TARGET_W, 0], [0, TARGET_H], [TARGET_W, TARGET_H]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        corrected = cv2.warpPerspective(frame, matrix, (TARGET_W, TARGET_H), flags=cv2.INTER_LANCZOS4)
        
        slit = corrected[:, (TARGET_W//2) : (TARGET_W//2) + 1].copy()
        if unwrapped_img is None:
            unwrapped_img = slit.astype(np.float32)
        else:
            unwrapped_img = np.hstack((unwrapped_img, slit.astype(np.float32)))

        if (time.time() - start_time) >= SCAN_DURATION:
            print(">>> [SAVING] Processing final image...")
            raw_data = np.clip(unwrapped_img, 0, 255).astype(np.uint8)
            final_img = cv2.resize(raw_data, (int(raw_data.shape[1] * STRETCH_RATIO), TARGET_H), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(f"scan_final_{int(time.time())}.png", final_img)
            
            # after saving the final image, we analyze the alignment and brightness compared to the reference image, then show the comparison window
            result_img, success = analyze_alignment(final_img, 'ref.png')
            if success:
                cv2.imwrite(f"result_fixed_{int(time.time())}.png", result_img)
                show_comparison_window(result_img, 'ref.png')
            
            is_scanning = False
            scan_finished = True

    if is_center_green and scan_finished:
        print(">>> [RESET] Waiting for next can...")
        scan_finished = False; unwrapped_img = None

    # --- UI & MONITORING ---
    monitor = cv2.resize(frame, (960, 540)); mx, my = 480, 270
    if is_waiting:
        cross_color = (0, 255, 255); status_text = f"READY IN... {max(0, WAIT_BEFORE_SCAN - (time.time()-ready_start_time)):.1f}s"
    elif is_scanning:
        cross_color = (0, 255, 0); status_text = f"SCANNING... {int(((time.time()-start_time)/SCAN_DURATION)*100)}%"
    elif scan_finished:
        cross_color = (0, 0, 255); status_text = "FINISHED - PLEASE REMOVE"
    else:
        cross_color = (0, 0, 255); status_text = "PLACE CAN AT CENTER"

    cv2.drawMarker(monitor, (mx, my), cross_color, cv2.MARKER_CROSS, 40, 2)
    cv2.circle(monitor, (mx, my), 20, cross_color, 1)
    cv2.putText(monitor, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, cross_color, 2)
    cv2.imshow('Cylindrical Scan System', monitor)
    
    if unwrapped_img is not None:
        live_preview = np.clip(unwrapped_img, 0, 255).astype(np.uint8)
        cv2.imshow('Live Result', cv2.resize(live_preview, (min(live_preview.shape[1]*2, 1200), 400)))

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()