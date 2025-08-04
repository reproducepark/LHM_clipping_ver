# LHM_for_SMPLest-X

This repository enables LHM to function even on GPUs with low VRAM, and it processes SMPLest-X outputs to smooth out motion and camera movements.

For installation and env settings, refer to original repo.

**Original Repository: https://github.com/aigc3d/LHM**

## Quick Start

### 1. Video to Motion Processing
```bash
./engine/pose_estimation/v2mseg.py --video_path ./myvideos/<your_video>.mp4 --output_path ./myoutputs
bash inference.sh LHM-500M train_data/example_imgs/<full_body_img>.jpg myoutputs/<your_video>/smplx_params
```

### 2. SMPLest-X Motion Data Processing
```bash
python integrated_processing.py
```

## SMPLest-X Motion Data Processing Options

### Basic Usage
```bash
# Basic execution (simplest method)
python integrated_processing.py

# Execute with custom directories
python integrated_processing.py input_folder output_folder
```

### Advanced Options

| Option | Description | Default | Range |
|--------|-------------|---------|-------|
| `--trans-z-scale` or `-z` | trans z-value scaling factor | 0.4 | 0.0 ~ 2.0 |
| `--stabilize-strength` or `-s` | trans stabilization strength | 0.7 | 0.0 ~ 1.0 |
| `--pose-strength` | rotation data stabilization strength | 0.3 | 0.0 ~ 1.0 |
| `--pose-window-size` | rotation data stabilization window size | 3 | 1 ~ 15 |
| `--no-stabilize-pose` | disable rotation data stabilization | - | - |

### Usage Examples

```bash
# Apply stronger stabilization
python integrated_processing.py datas/test_video datas/processed_output \
  --stabilize-strength 0.8 --pose-strength 0.5

# Disable rotation data stabilization
python integrated_processing.py datas/test_video datas/processed_output \
  --no-stabilize-pose

# Process with larger window for smoother results
python integrated_processing.py datas/test_video datas/processed_output \
  --pose-window-size 5 --pose-strength 0.4

# Use all options
python integrated_processing.py datas/test_video datas/processed_output \
  --trans-z-scale 0.4 \
  --stabilize-strength 0.7 \
  --stabilize-pose \
  --pose-window-size 3 \
  --pose-strength 0.3
```

### What the Script Does

This script processes SMPLest-X model output JSON files:

1. **Zero-value frame detection and interpolation**: Find frames where all values are 0 and interpolate using previous/next frames
2. **trans z-value scaling**: Scale the z-value of trans by the specified factor
3. **trans stabilization**: Smooth trans values using moving average
4. **Rotation data stabilization**: Smooth rotation data using quaternion-based approach

**SMPLest-X parameters being processed:**
- `trans`: Global position (3D)
- `root_pose`: Root rotation (3D axis-angle)
- `body_pose`: Body joint rotations (21×3 axis-angle)
- `jaw_pose`: Jaw rotation (3D axis-angle)
- `leye_pose`, `reye_pose`: Eye rotations (3D axis-angle)
- `lhand_pose`, `rhand_pose`: Hand rotations (15×3 axis-angle)

### Help
```bash
python integrated_processing.py --help
```

## Background Merging

Updated: I added background merging process with ffmpeg. Auto background resizing.

This feature replaces the white (ffffff) background of the generated video with a desired image.

```bash
python merge_background.py <background_img_path>.jpg <forground_video_path>.mp4 <output_path>.mp4
```

## Notes

- Input folder must contain SMPLest-X output JSON files
- Files are sorted numerically (e.g., 000001.json, 000002.json, ...)
- Output folder will be created automatically if it doesn't exist
- Processing progress is displayed step by step
- Each JSON file should contain SMPLest-X parameters (trans, root_pose, body_pose, etc.)

## Directory Structure

```
LHM_clipping_ver/
├── datas/
│   ├── test_video/          # SMPLest-X output JSON files
│   └── processed_output/    # Processed SMPLest-X JSON files
├── integrated_processing.py # SMPLest-X data processing script
└── merge_background.py      # Background merging script
```
