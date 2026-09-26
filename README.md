# DENSE Depth Estimation

This repository provides the training, testing, and evaluation code for depth estimation on the DENSE dataset.

## 1. Preparation

### 1.1 Pretrained Backbone

Please place the pretrained `maxvit_tiny` backbone in:

```bash
model/networks.py
```

The pretrained `maxvit_tiny` weights can be downloaded from:

```bash
https://huggingface.co/timm/maxvit_tiny_tf_224.in1k
```

You can also download it from the following link:

- [maxvit_tiny_tf_224.in1k](https://huggingface.co/timm/maxvit_tiny_tf_224.in1k)

### 1.2 Dataset

Before training the model, please download the DENSE dataset from the official project page:

```bash
https://rpg.ifi.uzh.ch/E2DEPTH.html
```

Dataset link:

- [DENSE Dataset](https://rpg.ifi.uzh.ch/E2DEPTH.html)

After downloading the dataset, please replace the `xxx` in `configs/default.json` with the path to the downloaded dataset.

For example:

```json
{
    "data_dir": "/path/to/DENSE"
}
```

Please make sure that the dataset path is correctly set before training.

## 2. Training

To train the model, run:

```bash
python train.py
```

By default, the checkpoints will be saved to:

```bash
checkpoints/DENSE
```

You can change the checkpoint saving directory by modifying the `save_dir` field in `configs/default.json`.

For example:

```json
{
    "save_dir": "checkpoints/DENSE"
}
```

## 3. Testing

After training, run:

```bash
python dense_test.py
```

Before running `dense_test.py`, please replace the `xxx` fields with the corresponding checkpoint path and configuration file path.

For example:

```python
checkpoint_path = "checkpoints/DENSE/xxx.pth"
config_path = "configs/default.json"
```

The testing script will generate the predicted depth files, such as:

```bash
pred_depth.npy
```

After testing, you will obtain a folder containing the predicted depth maps and the corresponding ground-truth depth maps.

## 4. Evaluation

To evaluate the results, please replace the `xxx` fields in `evaluation.py` with the corresponding `.npy` file paths.

For example:

```python
pred_path = "path/to/pred_depth.npy"
gt_path = "path/to/gt_depth.npy"
```

Then run:

```bash
python evaluation.py
```

The final quantitative results will be printed after the evaluation is completed.

## 5. Overall Pipeline

The complete pipeline is as follows:

```bash
# 1. Train the model
python train.py

# 2. Generate predicted depth maps
python dense_test.py

# 3. Evaluate the results
python evaluation.py
```

## Notes

- Please ensure that the pretrained `maxvit_tiny` backbone is correctly loaded before training.
- Please ensure that the dataset path in `configs/default.json` points to the downloaded DENSE dataset.
- Please ensure that the checkpoint path and configuration file path in `dense_test.py` are correctly set before testing.
- Please ensure that the predicted depth file and ground-truth depth file are correctly specified in `evaluation.py` before evaluation.