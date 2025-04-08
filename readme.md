🚗 Smart Vehicle Detection & Tracking System (YOLOv8 + BoT-SORT)
================================================================

This project implements a robust multi-object vehicle detection and tracking system based on YOLOv8 and BoT-SORT. It supports real-time object detection, trajectory tracking, customized line-crossing counting, perspective correction, and video output visualization. A DeepSORT version is also included for benchmarking. Ideal for applications in intelligent traffic monitoring, road analysis, and smart parking.

* * *

🧠 Key Features
---------------

| Module                  | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| 🚗 Object Detection      | Uses YOLOv8 (Nano) for lightweight and efficient vehicle detection |
| 🛰️ Multi-Object Tracking | BoT-SORT integrates Kalman Filter + appearance-based ReID for stable tracking |
| 🎯 Custom Line Counting  | Draw up/down lines interactively to count vehicle crossings and direction |
| 🧾 Report Generation     | Automatically generates screenshots and traffic summary reports |
| 🎥 Video I/O Support     | Supports `.mp4` input/output with real-time annotated visualization |
| 🧪 DeepSORT Version      | Provides a DeepSORT implementation as a comparative baseline |

* * *

📂 Project Structure
--------------------

```
Car-Detection/
├── .venv/                              # Python virtual environment
├── Dataset/                            # (Optional) raw data directory
├── project_data/                       # Temporary or intermediate data
├── runs/                               # Output folder for videos and images
├── better_tracker_yolov8_botsort.ipynb            # Core implementation with YOLOv8 + BoT-SORT
├── functions.py                                     # Utility functions (upload, drawing, reporting)
├── Group_last.ipynb                                 # Final integrated notebook with full pipeline
├── highway.mp4                                      # Sample input video
├── main.ipynb                                       # Main notebook for interactive execution
├── main.py                                          # Executable script for deployment/CLI
├── output.jpg                                       # Sample annotated frame output
├── readme.md                                        # This documentation
├── requirement.txt                                  # Required dependencies
├── road_report.md                                   # Example traffic summary report
├── sample_video.mp4                                 # Backup video file
├── smart_tracking_deepsort_version.ipynb            # DeepSORT-based tracking version
├── tracker_with_linecounting.ipynb                  # Light version with line crossing logic
├── yolov8_botsort_tracker_module.ipynb              # Modularized YOLOv8 + BoT-SORT tracker functions
├── yolov8n.pt                                       # Pretrained YOLOv8 Nano model weights
```

* * *

🔍 Core Code Overview
---------------------

### 🧩 `functions.py`

*   Contains general-purpose helper functions:

    *   `calculate_md5()`, `upload_file()`, `get_report()` for API interaction and logging

    *   Drawing utilities for visualization

    *   File handling and hashing support

### 🧩 `main.py`

*   Encapsulates GUI creation and event logic:

    *   `create_widgets()`, `open_video()` for user interface

    *   `start_drawing_up_line()` and `start_drawing_down_line()` for user-defined line placement

    *   `process_video()` to orchestrate detection + tracking

    *   `check_line_crossing()` to handle vehicle direction detection

    *   `update_frame()` for real-time UI updates

### 📒 Notebooks (`main.ipynb`, `Group_last.ipynb`)

*   Provide an interactive interface to test and visualize the detection pipeline

*   Include perspective calibration and multi-line drawing

*   Replicate the same logic as in `main.py` with real-time visual feedback

### 📒 `better_tracker_yolov8_botsort.ipynb`

*   Demonstrates core object detection + BoT-SORT tracking

*   Visualizes tracked IDs, bounding boxes, and outputs annotated video frames

### 📒 `tracker_with_linecounting.ipynb`

*   Focuses on manual line drawing with OpenCV

*   Tracks object transitions across counting lines

*   Includes minimal setup for quick testing

### 📒 `yolov8_botsort_tracker_module.ipynb`

*   Highly modular implementation for reusability

*   Includes:

    *   `detect_speed()` for estimating object velocity

    *   `check_line_cross()` for rule-based counting logic

    *   `process_video()` as a centralized logic function

*   * *

⚙️ Installation
---------------

1.  Use Python 3.8+

2.  Install dependencies:

```bash
pip install -r requirement.txt
```

Or manually:

```bash
pip install ultralytics opencv-python numpy matplotlib filterpy norfair
```

3.  Make sure `yolov8n.pt` is available in the project root.

* * *

🚀 Getting Started
------------------

### ✅ Interactive Mode (Notebook)

Open `main.ipynb` or `Group_last.ipynb`:

*   Select video (e.g., `highway.mp4`)

*   Draw lines for counting

*   Run detection, tracking, and see visual results in real time

### 🧑‍💻 Script Mode (Command Line)

```bash
python main.py --video_path highway.mp4 --output_path ./runs/result.mp4
```

Optional arguments:

| Parameter       | Description                  |
| --------------- | ---------------------------- |
| `--video_path`  | Path to input video          |
| `--output_path` | Path to save output video    |
| `--enable_line` | Enable or disable line count |

* * *

🧪 DeepSORT Version
-------------------

The `smart_tracking_deepsort_version.ipynb` notebook uses DeepSORT for tracking. You can compare its performance against BoT-SORT in terms of ID stability, occlusion handling, and ReID effectiveness.

* * *

📊 Output Samples
-----------------

*   🖼 `output.jpg`: Sample frame with detected boxes and IDs
*   🎞 `runs/result.mp4`: Annotated video output
*   📄 `road_report.md`: Markdown-based summary of vehicle counts (can be extended)

🔄 Potential Extensions
-----------------------

* Add pedestrian, cyclist, or bus class support

* Integrate RTSP/stream camera input

* Build a RESTful API or web-based dashboard

* Visualize hourly traffic trends using charts (e.g., seaborn)

  
