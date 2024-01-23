# OTMTD

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
Raw data of protein downstream tasks could be download at `https://drive.google.com/drive/folders/1BYzf2RJFcMnT_8Cf_F0Gu_ZWGvM7Z0eY?usp=sharing`. The data format and size of datasets are as follows:

* **Without uniprot id**
    |task|seq|label|
    | :----- | :-----: | :-----: |
    |Stability|DQSVRKLV...|-0.2099|
    |Fluorescence|SKGEELFT...|3.7107|
    |Remote Homology|PKKVLTGV...|51|
    |Secondary Structure|MNDKRLQF...|22222000...|
    |Signal Peptide|MLGMIRNS...|0|
    |Fold Classes|MSPFTGSA...|c|

* **With uniprot id**
  * PDBBind
    |uniprot_id|seq|smiles|rdkit_smiles|label|dataset_type|
    | :-----: | :-----: | :-----: | :----- | :-----: | :-----: |
    |11gs|PYTVVYFP...|OC(=O)c1cc...|O=C(O)c1cc...|4.62|train|

  * Kinase
    |molecule|uniprot_id|seq|label|
    | :-----: | :-----: | :-----: | :----- |
    |COC1C(N(C)C(C)=O)...|P05129|MAGLGPGV...|1|

* **Size of datasets**
    <table border="1">
        <tr>
        <th></th>
        <th>Stability</th>
        <th>Fluorescence</th>
        <th>Remote Homology</th>
        <th>Secondary Structure</th>
        <th>Signal Peptide</th>
        <th>Fold Classes</th>
        <th>Pdbbind</th>
        <th>Kinase</th>
        </tr>
        <tr>
        <td>Train</td>
        <td>53614</td>
        <td>21446</td>
        <td>12312</td>
        <td>8678</td>
        <td>16606</td>
        <td>15680</td>
        <td>11906</td>
        <td>91552</td>
        </tr>
        <tr>
        <td>Valid</td>
        <td>2512</td>
        <td>5362</td>
        <td>736</td>
        <td>2170</td>
        <td>/</td>
        <td>/</td>
        <td>1000</td>
        <td>/</td>
        </tr>
        <tr>
        <td>Test</td>
        <td>12851</td>
        <td>27217</td>
        <td>718</td>
        <td>513</td>
        <td>4152</td>
        <td>3921</td>
        <td>290</td>
        <td>19685</td>
    </table>

Then, the datasets without uniprot ids are manually added with the uniprot id following the template `<task>_<dataset_type>_<number>`, e.g., `fluo_train_17878`. 

Next, the corresponding Gene Ontology(GO) is retrieved from the [ idmapping_selected.tab](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/idmapping_selected.tab.gz) according to the uniprot id, and `No goterm` will be returned if no GO is retrieved. The command is as follow:

```bash
grep -w <uniprot_id> idmapping_selected.tab -m 1
```

The reference code about processing data can be found in the file named `processing_data.ipynb`, which includes retrieving GO and processing labels.

Processed data of protein pretraining and downstream tasks should be placed in the `processed_data` directory. The data format is as follows:

|task|uniprot_id|seq|GO|label|
| :----- | :-----: | :-----: | :-----: | :-----: |
|Stability|stab_train_0|DQSVRKLV...|No goterm|2|
|Fluorescence|fluo_train_17878|SKGEELFT...|No goterm|0|
|Remote Homology|remo_train_0|PKKVLTGV...|No goterm|0|
|Secondary Structure|secstruc_train_0|MNDKRLQF...|No goterm|1|
|Signal Peptide|sign_train_0|MLGMIRNS...|No goterm|0|
|Fold Classes|fold_train_0|MSPFTGSA...|No goterm|2|
|PDBBind|11gs|PYTVVYFP...|GO:0005737;GO:0005829;...|2|
|Kinase|P05129|MAGLGPGV...|GO:0004672;GO:0004674;...|1|
    
## 2. Embeddings generation
Use `represent/model_interpreter_multi.py` to represent protein tasks, and modify `represent/config.yaml` to configure the downstream task paths. For example,
```python
python model_interpreter_multi.py --batch_size=32 --gpu=0 --ft=multi
```
The generated embeddings have the following format:
| |pro_id|pro_seq|pro_emb|
| :----- | :-----: | :-----: | -----: |
|0|fluo_train_0|SKGEELFT...|[-0.5087447166442871, -2.313387870788574, -0.1...|

Additionly, pretrained and finetuned weights used in genereating embeddings come from ours previous work [MASSA](https://github.com/SIAT-code/MASSA). And the hyperparameters of experiment are as follow:
|                | Pretrain          | Stability         | Fluorescence      | Remote Homology   | Secondary Structure | Pdbbind           | Kinase            | Skempi           |
| -------------- | :-----------------: | :-----------------: | :-----------------: | :------------------: | :--------------------: | :-----------------: | :-----------------: | :-----------------: |
| epoch          | 150               | 150               | 150               | 150               | 150                  | 150               | 150               | 150               |
| batch size     | 4                 | 8                 | 32                | 4                 | 8                    | 8                 | 4                 | 8                 |
| lr (learning rate) | 1e-4           | 1e-4              | 1e-4              | 1e-4              | 1e-4                 | 1e-4              | 1e-4              | 1e-4              |
| weight decay   | 1e-4              | 1e-4              | 1e-4              | 1e-4              | 1e-4                 | 1e-4              | 1e-4              | 1e-4              |
| gradient accumulation | 8             | 8                 | 8                 | 8                 | 8                    | 8                 | 8                 | 8                 |
| optimizer      | RAdam             | RAdam             | RAdam             | RAdam             | RAdam                | RAdam             | RAdam             | RAdam             |
| Loss           | CrossEntropy Loss | MSELoss           | MSELoss           | Equalized Focal Loss | CrossEntropy Loss   | MSELoss           | CrossEntropy Loss | MSELoss           |



## 3. OTMTD Calculation
Run `otmtd_cal.ipynb` to calculate the transferability metrics from multi-modal multi-task pre-training to downstream tasks.

## Acknowledgement
The SOFTWARE will be used for teaching or not-for-profit research purposes only. Permission is required for any commercial use of the Software.

