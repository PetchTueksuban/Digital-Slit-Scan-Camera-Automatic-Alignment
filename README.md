# Digital-Slit-Scan-Camera-Automatic-Alignment  
**Digital Slit-Scan Camera** is a professional-grade imaging tool designed for Cylindrical Surface Inspection and High-Resolution Dimensional Measurement.  
This system unwraps rotating cylindrical objects into a flat 2D map and utilizes an automatic template matching algorithm to align the result with a reference image for defect detection..  

# Key Features 
* **Ultra-Sharp RAW Sampling:** Utilizes a precise 2-pixel width "slit" extraction from the center of each frame, eliminating Bayer pattern artifacts and providing superior clarity over standard scanning methods
* **Automatic Template Matching:** Dynamically calculates the object's center and performs an image "roll" to perfectly align the scanned output with the ref.png master image.
* **Red-Line Inspection Scale:** Generates a vertical comparison grid every 100 pixels between the Reference (Top) and Scan (Bottom), allowing for immediate visual detection of distortions or dents.Red-Line Inspection Scale: Generates a vertical comparison grid every 100 pixels between the Reference (Top) and Scan (Bottom), allowing for immediate visual detection of distortions or dents.
* **Intelligent Auto-Trigger:** Features an HSV-based green screen detection system that automatically starts the scan after a 1.0-second stability delay when an object is detected.
* **High-Precision Measurement:** Automatically calculates the object's diameter in centimeters using calibrated pixel-to-mm ratios.
  
# How it Works (System Logic)  
**Prepare Reference**  
Place your master image in the project folder and name it ref.png. 
**Configuration**  
Adjust LOCKED_FOCUS_VALUE and PIXEL_PER_MM_FINAL in the script to match your hardware setup and camera distance.
**Run the System:**  
Once the scan is finished, the system performs a "Fixed Alignment":  
   * Execute the script: python main.py.
   * Place the object on the rotating platform. The system will wait for 1.0s to stabilize and then begin scanning automatically.  
   * Once the 594 pixels are captured, the INSPECTION window will display the vertical comparison with red scale lines and save the report to your drive.

**the image shown thecans that start scan in differrence position and result will come same.**  
<img width="1495" height="2160" alt="comparison_report_1772036554" src="https://github.com/user-attachments/assets/44803572-666a-4c17-b7d1-3c8ba38bc345" />


