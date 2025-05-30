# Can VLMs be Used to Control Bionic Hands? An Evaluation of Object Perception and Grasp Inference Capabilities

This repository contains the code and data for benchmarking Vision-Language Models (VLMs) on bionic hand grasp decision tasks. Our work evaluates how well different VLMs can perceive objects and predict grasp parameters from single RGB images.

## Overview

We evaluate 8 state-of-the-art VLMs on their ability to:
- Identify objects (naming, shape, orientation)
- Estimate physical dimensions (width, height, depth)
- Predict grasp parameters (type, rotation, aperture, number of fingers)

The benchmark uses 34 real-world images of everyday objects captured with an Intel RealSense camera.

## Repository Structure

```
benchmarking_vlms/
├── photo_booth_images/        # Input images for benchmarking
│   └── new/                   # 34 test images (PNG format)
├── ground_truth.json          # Human-annotated ground truth data
├── openrouter_benchmarker.py  # Main benchmarking script
├── analyze.py                 # Metrics computation and analysis
├── generate_latex_tables.py   # Generate paper results tables
├── realsense_photo_booth.py   # Image capture utility (optional)
├── requirements.txt           # Python dependencies
├── vlm_benchmark_results-*.csv # Raw benchmark outputs
└── metrics_results.json       # Computed metrics from analysis
```

## Pipeline

The complete benchmarking pipeline consists of four steps:

### 1. Image Collection (Optional)
If you want to capture your own images, in the repo we provide 34 object images that we used on paper already, feel free to remove them and run:
```bash
python3 realsense_photo_booth.py
```
This uses an Intel RealSense camera to capture RGB images with consistent setup. But if you need, you can adjust with openCV cap.read() function to use any camera.

### 2. Run VLM Benchmarks
```bash
python3 openrouter_benchmarker.py
```

Before running, you need to:
- Set your OpenRouter API key: `export OPENROUTER_API_KEY="your_key_here"`
- The script will benchmark all models on all images in `photo_booth_images/new/`
- Results are saved to a timestamped CSV file

The script evaluates these models:
- Claude 3.5 Sonnet, Claude 3.7 Sonnet, Claude Sonnet 4
- GPT-4.1
- Gemini 2.5 Flash (two versions)
- Gemma 3 27B-IT
- Mistral Medium

### 3. Analyze Results
```bash
python3 analyze.py
```

This script:
- Loads the benchmark results CSV and ground truth data
- Computes metrics for each model:
  - Categorical accuracy (naming, shape, orientation, grasp type)
  - Mean Absolute Error (MAE) for continuous values
  - Mean Error (ME) for bias analysis
  - Latency and cost statistics
- Outputs `metrics_results.json`

### 4. Generate Paper Tables
```bash
python3 generate_latex_tables.py
```

This generates all LaTeX tables from the paper's results section, ready for publication.

## Ground Truth Format

The `ground_truth.json` file contains human annotations for each image:

```json
{
  "photo_20250507-102628.png": {
    "object_name": "can",
    "object_shape": "cylinder",
    "object_orientation": "vertical",
    "object_dimensions_mm": [67, 123, 67],
    "grasp_type": 0,
    "hand_rotation_deg": 0,
    "hand_aperture_mm": 70,
    "num_fingers": 4
  }
}
```

## Prompt Design

We use a zero-shot prompt that requests structured JSON output, feel free to ask different questions in INDIVIDUAL_PROMPTS or edit the COMPLEX_PROMPT:

```
Given this RGB image of a single object on a table, analyze it and provide:
1. Object name, shape, orientation
2. Dimensions in mm (width, height, depth)
3. Grasp parameters for a robotic hand

Output format: {"object_name": "...", ...}
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `openai`: OpenRouter API client
- `pandas`: Data processing
- `numpy`: Numerical computations
- `Pillow`: Image handling
- `pyrealsense2`: Camera interface (optional)
- `opencv-python`: Image processing (optional)

## API Costs

Approximate costs per model for 34 images:
- Claude models: ~$0.41
- GPT-4.1: ~$0.14
- Gemini models: ~$0.04
- Mistral Medium: ~$0.03
- Gemma 3: ~$0.01

## Citation

If you use this benchmark in your research, please cite:

### To be completed with the actual citation once the paper is published.
```bibtex
@article{karaali_vlms_2025,
  title={Can VLMs be Used to Control Bionic Hands? An
Evaluation of Object Perception and Grasp
Inference Capabilities},
  author={Karaali et al.},
  journal={ICAT 2O25},
  year={2025}
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Contributing

We welcome contributions! Please feel free to:
- Add new VLMs to the benchmark
- Contribute additional test images
- Improve the analysis metrics
- Report issues or suggest enhancements

## Reproducibility

To reproduce our paper's results:
1. Use the provided `photo_booth_images/new/` images
2. Run the benchmarking script with the same model versions
3. The analysis will generate identical metrics
4. LaTeX tables will match those in the paper

Note: Model responses may vary slightly between runs due to sampling, but aggregate metrics should remain consistent.