<p align="center">
  <img src="https://user-images.githubusercontent.com/62797431/229082913-834277f7-fd01-448f-b3e2-8d1d750e48d6.png" width="544" height="184">
</p>

# Orbis v2 
## Automated Colony Area Measurement from Petri Dish Images

Orbis v2 is an open-source tool for automated measurement of microbial colony area from images of agar plates.
It is optimized for filamentous fungi, including colonies with irregular morphologies, variable colors, and heterogeneous backgrounds.

Orbis v2 is the second generation of the Orbis tool and is fully rewritten in Python with a Streamlit-based web interface for maximum accessibility.

**🌐 Live app: https://orbis-v2.streamlit.app/**

### Key Features
- Automated colony segmentation and contour detection
- Supports irregular, non-circular colonies
- Real-time parameter preview
- Batch image processing
- Pixel-to-length calibration (in units such as cm, mm, inches, etc.)
- Optional image masking, brightness normalization, and additional processing
- Fully open-source (GPL-3.0)

### Typical Use Case
1. Photograph Petri dishes from above using a smartphone or camera
2. Upload one or multiple images to Orbis v2
3. (Optional) Calibrate scale using a ruler in the image
4. Adjust threshold and preprocessing settings
5. Download:
   - Annotated images with colony outlines
   - CSV file with colony areas
   - Processing log

### Method Summary

Orbis v2 uses a transparent classical image-processing pipeline: 
1. Optional preprocessing (contrast, masking, denoising)
2. Global thresholding
3. Binary mask generation
4. Contour detection
5. Area calculation (pixels or real units)

All steps update in real time, allowing intuitive adjustment without software expertise. 

Orbis v2 was benchmarked against validated Orbis v1 measurements, as well as manual measurements. 
It is also significantly faster (76% faster analysis) than Orbis v1, and the difference is even larger compared to manual measurements.
This program was designed with accessibility in mind, to make it convenient to use across laboratories and teaching environments.
It is deliberately lightweight and explainable, making it suitable for education, routine lab work, and standardized workflows.

### Citation and Additional Information
For additional information, please see our preprint: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5912070
