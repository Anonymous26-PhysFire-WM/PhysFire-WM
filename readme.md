# Fire Spread Prediction with PhysFire-WM

This repository contains the implementation for our physics-informed world model for fire spread prediction, based on Wan2.1-VACE-1.3B architecture.

## Dataset

- **Data Path**: `data/5_Regions` and `data/sim2real`
- **Task Metadata**: `data/5_Regions/metadata_multi-Region-Task.csv`

## Model Training

```bash
# Run training script with LoRA fine-tuning
./examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B.sh
```

### Training Configuration
**Key parameters to modify:**
- `dataset_base_path`
- `dataset_metadata_path` 
- `data_file_keys`
- `extra_inputs`

**Recommended settings:**
- Height: `480`
- Width: `832`

## Model Inference

Run inference/validation script:

```bash
python examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-1.3B.py
```
### Inference Configuration

**Key parameters to update:**
- `pipe.load_lora()`
- `vace_video`
- `vace_video_mask`
- `reference_image`

## Acknowledgments

This work is built upon the following open-source project:
- **DiffSynth-Studio**: https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo
