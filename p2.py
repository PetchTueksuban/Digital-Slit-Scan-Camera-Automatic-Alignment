import cv2
import numpy as np
import time
import os

# config
TARGET_H, TARGET_W = 1080, 540 
STRETCH_RATIO = 2.5 
LOCKED_FOCUS_VALUE = 350
WAIT_BEFORE_SCAN = 1.0     
DISPLAY_DURATION = 5.0     

# CORE SETTINGS
REQUIRED_PIXELS = 586      
FPS_LIMIT = 30
FRAME_TIME = 1.0 / FPS_LIMIT

# CALIBRATION
PIXEL_PER_MM_FINAL = 7.32  
LOWER_GREEN = np.array([35, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])


# (COMPARISON & SAVE)


def analyze_and_draw_comparison(final_img, reference_path):
    """จัดแนวภาพ วาดเส้นแดงเปรียบเทียบ และบันทึกรูปผลลัพธ์"""
    ref_color = cv2.imread(reference_path)
    if ref_color is None:
        print("!!! ไม่พบไฟล์ ref.png กรุณาตรวจสอบไฟล์อ้างอิง !!!")
        return False
    
    h_ref, w_ref = ref_color.shape[:2]
    
    #Alignment by Template Matching 
    gray_ref = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
    gray_scan = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    
    template_w = min(400, w_ref)
    template = gray_ref[:, w_ref//2 - template_w//2 : w_ref//2 + template_w//2]
    scan_triple = np.hstack((gray_scan, gray_scan, gray_scan))
    res = cv2.matchTemplate(scan_triple, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    
    #calculate Roll 
    shift_amount = (w_ref // 2) - (max_loc[0] + template_w // 2) % final_img.shape[1]
    aligned_scan = np.roll(final_img, shift_amount, axis=1)
    
    #Reference size
    aligned_resized = cv2.resize(aligned_scan, (w_ref, h_ref))
    
    
    cv2.putText(ref_color, "REFERENCE IMAGE (TOP)", (20, 60), 2, 1.2, (0, 0, 255), 3)
    cv2.putText(aligned_resized, "SCANNED RESULT (BOTTOM)", (20, 60), 2, 1.2, (0, 0, 255), 3)


    canvas = np.vstack((ref_color, aligned_resized))
    for x in range(0, w_ref, 100):
        cv2.line(canvas, (x, 0), (x, canvas.shape[0]), (0, 0, 255), 2)
        

    save_path = f"comparison_report_{int(time.time())}.png"
    cv2.imwrite(save_path, canvas)
    

    scale = 900 / canvas.shape[0]
    final_view = cv2.resize(canvas, (0, 0), fx=scale, fy=scale)
    cv2.imshow('INSPECTION: RED LINE COMPARISON', final_view)
    return True

# camera
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0); cap.set(cv2.CAP_PROP_FOCUS, LOCKED_FOCUS_VALUE)

unwrapped_img = None; is_scanning = False; scan_finished = False
ready_start_time = 0; is_waiting = False; final_measured_cm = 0.0
prev_time = time.time(); finish_time = 0; final_img = None 

#main loop
while True:
    loop_start = time.time()
    if cap.grab(): ret, frame = cap.retrieve()
    else: break
    if not ret: break

    h, w = frame.shape[:2]; cx, cy = w // 2, h // 2
    
    # Background check
    roi = frame[max(0, cy-10):min(h, cy+10), max(0, cx-10):min(w, cx+10)]
    avg_hsv = cv2.cvtColor(np.uint8([[np.mean(roi, axis=(0, 1)).astype(np.uint8)]]), cv2.COLOR_BGR2HSV)[0][0]
    is_center_green = (LOWER_GREEN[0] <= avg_hsv[0] <= UPPER_GREEN[0])

    if not is_center_green and not is_scanning and not scan_finished:
        if not is_waiting: is_waiting = True; ready_start_time = time.time()
        if time.time() - ready_start_time >= WAIT_BEFORE_SCAN:
            unwrapped_img = None; is_scanning = True; is_waiting = False

    # Scanning Process 
    if is_scanning:
        raw_slit_area = frame[:, cx : cx + 2] 
        slit = np.mean(raw_slit_area, axis=1, keepdims=True).astype(np.uint8)
        
        if unwrapped_img is None: unwrapped_img = slit
        else:
            if unwrapped_img.shape[1] < REQUIRED_PIXELS:
                unwrapped_img = np.hstack((unwrapped_img, slit))

        if unwrapped_img.shape[1] >= REQUIRED_PIXELS:
            final_img = cv2.resize(unwrapped_img, (int(unwrapped_img.shape[1] * STRETCH_RATIO), TARGET_H), interpolation=cv2.INTER_LANCZOS4)
            
            # Measurement
            gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(cv2.medianBlur(gray, 5), (7, 7), 0)
            _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                fw = cv2.boundingRect(max(cnts, key=cv2.contourArea))[2]
                final_measured_cm = (fw / PIXEL_PER_MM_FINAL) / 10.0
            
            cv2.imwrite(f"scan_ultra_sharp_{int(time.time())}.png", final_img)
            
            
            analyze_and_draw_comparison(final_img, 'ref.png')
            
            is_scanning = False; scan_finished = True; finish_time = time.time() 

    # Result
    if scan_finished:
        if (time.time() - finish_time >= DISPLAY_DURATION):
            scan_finished = False; unwrapped_img = None
            try:
                cv2.destroyWindow('Measurement Result')
                cv2.destroyWindow('Final Scanned Image') 
                cv2.destroyWindow('INSPECTION: RED LINE COMPARISON')
            except: pass
        else:
            cv2.namedWindow('Final Scanned Image', cv2.WINDOW_NORMAL)
            cv2.imshow('Final Scanned Image', final_img)
            res_win = np.zeros((200, 600, 3), dtype=np.uint8)
            cv2.putText(res_win, f"RESULT: {final_measured_cm:.2f} cm", (40, 110), 2, 1.8, (0, 255, 0), 4)
            cv2.imshow('Measurement Result', res_win)

    # 4. Monitor
    monitor = cv2.resize(frame, (960, 540))
    m_color = (0, 255, 0) if is_scanning else (0, 255, 255) if is_waiting else (0, 0, 255)
    cv2.drawMarker(monitor, (480, 270), m_color, cv2.MARKER_CROSS, 40, 2)
    cur_px = 0 if unwrapped_img is None else unwrapped_img.shape[1]
    cv2.putText(monitor, f"PX: {cur_px}/{REQUIRED_PIXELS}", (20, 50), 2, 1, (255, 255, 255), 2)
    cv2.imshow('Cylindrical Scan System', monitor)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()