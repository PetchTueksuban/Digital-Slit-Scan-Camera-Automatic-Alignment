# Digital-Slit-Scan-Camera-Automatic-Alignment
This project is an advanced evolution of the Digital-Slit-Scan-Camera. It transforms a standard webcam into a high-precision cylindrical scanner, capable of unwrapping can labels and automatically aligning them with a reference template.  

# Key Features (Update 2026)
* Intelligent Auto-Trigger: Starts scanning only when an object is detected and stabilized at the center for 1 second.
* Dynamic Perspective Correction: Automatically warps the camera view to align with the can's surface before scanning.
* High-Precision Alignment (SIFT): Uses SIFT and FLANN Matcher to automatically fix alignment errors, ensuring the scanned result matches your ref.png perfectly.
* Triple-Width Stitching Logic: A specialized technique that stacks the scanned image three times to prevent "cut-off" patterns and ensure a seamless 360-degree match.
* USAC_ACCURATE Homography: Employs advanced outlier rejection to handle tricky reflections on metallic can surfaces.

# How it Works (System Logic)  
**1. Detection Phase**  
The system uses HSV Color Masking to detect a green background. When the center area is blocked by a non-green object (the can), the countdown begins.  
**2. Scanning Phase (Slit-Scan)**  
Instead of taking a full photo, the system extracts a 1-pixel wide vertical "slit" from the center of each frame and stacks them horizontally over 21 seconds.  
**3. Post-Processing & Alignment**  
Once the scan is finished, the system performs a "Fixed Alignment":  
   * 1.Triple Stacking: The scan is horizontally duplicated 3 times to create a continuous loop.  
   * 2.Feature Matching: SIFT finds unique points on the scan vs. the reference.  
   * 3.Warping: The best match is extracted and warped to create result_fixed.png.

**the image shown thecans that start scan in differrence position and result will come same.**

<img width="1575" height="1080" alt="result_fixed_1771662590" src="https://github.com/user-attachments/assets/a6a65625-7fed-4cb6-83ab-690be5bff5d9" />

<img width="1575" height="1080" alt="ref" src="https://github.com/user-attachments/assets/cce613f3-2206-4b7e-b989-9cfa30a7ec0f" />
