# Fire Spread Prediction with PhysFire-WM 🔥🌍

This repository implements a **Physics-Informed World Model (PhysFire-WM)** for predicting the spread of fire.

The base model used for this implementation is **Wan2.1-VACE-1.3B**. For more details, please refer to the original model repository:  [GitHub Repository](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo).

## Dataset 📊

The following datasets are used for training and evaluation:

1. **FireSentry Dataset** [GitHub Repository](https://github.com/Munan222/FireSentry-Benchmark-Dataset)

2. **Sim2Real-Fire** (Real-world data part) [GitHub Repository](https://github.com/TJU-IDVLab/Sim2Real-Fire)

The dataset is organized as follows:

- **Data Paths**:
  - `data/5_Regions` – Contains data related to multiple regions for fire spread prediction.
  - `data/sim2real` – Contains real-world fire data for validation and testing.

- **Task Metadata**:
  - `data/5_Regions/metadata_multi-Region-Task.csv` – Metadata describing the multi-region tasks for fire prediction.

## Model Training ⚙️

To train the **PhysFire-WM** model, use the script below with **LoRA fine-tuning**:

```bash
# Run training script with LoRA fine-tuning
./examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B.sh
```

### Training Configuration
**Key parameters to modify:**
- `dataset_base_path` – Path to your dataset.
- `dataset_metadata_path` – Path to the metadata CSV.
- `data_file_keys` – Keys that define the structure of the data.
- `extra_inputs` – Any additional input parameters required by the model.

**Recommended settings:**
- Image Height: `480`
- Image Width: `832`

## Model Inference 🧠

To run inference or validate the model, use the following script:

```bash
python examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-1.3B.py
```
### Inference Configuration ⚡

Ensure the following parameters are correctly set in the inference script:

- `pipe.load_lora()` – Load the trained LoRA weights.
- `vace_video` – Path to the input video for prediction.
- `vace_video_mask` – Mask to define areas of interest in the video.
- `reference_image` – An image used as a reference for predicting fire spread.

The full dataset and model weights will be publicly released after paper acceptance.

## Acknowledgments 🙏

This work is based on and builds upon the following open-source projects:

- **DiffSynth-Studio**: [GitHub Repository](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo)
- **FireSentry Dataset**: [GitHub Repository](https://github.com/Munan222/FireSentry-Benchmark-Dataset)
- **Sim2Real-Fire Dataset**: [GitHub Repository](https://github.com/TJU-IDVLab/Sim2Real-Fire)
