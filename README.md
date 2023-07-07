# OTMTD
**A Transferability-Based Method for Evaluating the Protein Representation Learning** \
A novel method for quantitatively estimating the transfer performance from a multi-task pre-trained protein representation to downstream tasks.

## Directory Structure
```bash
├── example.ipynb                  # Example of OTMTD
├── otmtd_cal.ipynb                # OTMDT metric calculation
├── emprical_relate.ipynb          # Analysis of OTMTD and empirical results correlation
├── ot_baselines_cal.ipynb         # OT-based baseline metrics
├── non_ot_baselines_cal.ipynb     # Non-OT-based baseline metrics
├── metrics_perf_comp.ipynb        # Comparison of different metrics
├── requirements.txt               # Environment dependencies
├── README.md                      # Readme file
├── otmtd/                         # Definition files for OTMTD
├── otdd/                          # Definition files for OTDD
├── otce/                          # Definition files for OTCE
├── represent/                     # Represent protein tasks using MASSA
├── processed_data                 # Processed protein downstream task text data
├── protein_embeddings_MultiTasks  # Pre-trained or downstream task embeddings
├── cv_emb                         # Represent CV tasks as embeddings
```

## Requirements
[![python >3.9.1](https://img.shields.io/badge/python-3.9.1-brightgreen)](https://www.python.org/) [![torch-1.8.1](https://img.shields.io/badge/torch-1.8.1-orange)](https://github.com/pytorch/pytorch)

```bash
conda create -n otmtd python==3.9.1
conda activate otmtd
cd OTMTD
pip install -r requirements.txt
```

## Example
`example.ipynb` demonstrates the transferability calculation from pre-training to the Fluorescence task.
Note that the embeddings of the example need to be download at \
`https://drive.google.com/drive/folders/1RTphom46oGlJlnw52NSABMNQurWldhJi?usp=sharing`.

## 1. Data Processing
Processed data of protein downstream tasks should be placed in the `processed_data` directory. The data format is as follows:
|uniprot_id|pro_seq|GO|label|
| :----- | :-----: | :-----: | -----: |
|fluo_train_17878|SKGEELFT...|No goterm|2|

## 2. Embeddings generation
Use `represent/model_interpreter_multi.py` to represent protein tasks, and modify `represent/config.yaml` to configure the downstream task paths. For example,
```python
python model_interpreter_multi.py --batch_size=32 --gpu=0 --ft=multi
```
The generated embeddings have the following format:
| |pro_id|pro_seq|pro_emb|
| :----- | :-----: | :-----: | -----: |
|0|fluo_train_0|SKGEELFT...|[-0.5087447166442871, -2.313387870788574, -0.1...|

## 3. OTMTD Calculation
Run `otmtd_cal.ipynb` to calculate the transferability metrics from multi-modal multi-task pre-training to downstream tasks.

## Acknowledgement
The SOFTWARE will be used for teaching or not-for-profit research purposes only. Permission is required for any commercial use of the Software.
