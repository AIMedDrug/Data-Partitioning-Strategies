# Evaluating Data Partitioning Strategies for Accurate Prediction of Protein-Ligand Binding Free Energy Changes in Mutated Proteins

## Overview

This repository provides the supplementary materials, code, and datasets for the study titled **"Evaluating Data Partitioning Strategies for Accurate Prediction of Protein-Ligand Binding Free Energy Changes in Mutated Proteins"**

## Abstract

Accurate prediction of the relative free energy of protein-ligand binding, especially regarding protein mutations, is vital for drug design and interpreting drug resistance. However, machine learning (ML) / deep learning (DL) methods often struggle with generalization due to dataset partitioning strategy. Random data partitioning potentially produces spuriously high correlations that inflate performance estimates. UniProt-based splitting preserves data independence but lacks high prediction accuracy. In this study, we first evaluate six distinct ML/DL models on the MdrDB database using two data partitioning methods. Protein sequences are embedded using the ESM-2 protein large language model, integrating wild-type and mutant features. Although all models show high predictive correlations (Pearson coefficients up to 0.70) under random partitioning, their performance declines with UniProt-based partitioning. To address this issue, we propose a query-anchor pairwise learning framework, utilizing known states as anchor points for predicting unknown query states. The proposed method is validated across three systems, revealing that even a small amount of reference data can significantly enhance prediction accuracy. This enhancement suggests that leveraging known states as anchor points allows for more precise predicting of unknown query states.

## Repository Structure

* data/: Datasets including protein sequences, mutation records, ligand structures, experimental ΔΔG datasets, and ESM-2 embedding data.
* notebook/: notebooks for exploratory analysis and visualization.
* plt/: Plotting scripts and figures.
* results/: Model performance evaluation results and indicators.
* script/: Scripts for model training and evaluation under different partitioning strategies.
* Supplementary_data/: Supplementary data and scripts.
* Supplementary_Data_4/: Supplementary data for extended analyses.
* README.md: Project documentation.

## Key Contributions

* **Advanced Protein and Ligand Representations:** Employed protein embeddings from ESM-2 (a protein language model) and ligand molecular fingerprints (ECFP).
* **Random partitioning:** significantly overestimates model accuracy.
* **UniProt-based partitioning:** more realistically evaluates generalization, highlighting lower but genuine predictive capabilities.
* **Anchor-query pairwise strategy:** substantially improves prediction accuracy when limited reference data is available.
* **Novel Anchor-Query Pairwise Learning Framework:** The anchor-query pairwise strategy consistently yields superior predictive performance compared to the unpaired approach, especially when only limited reference data is available.
* **The results were comparable to those of the physicochemical method (FEP):** The RMSE was 0.87 kcal/mol in the ABL kinase system (1.11 kcal/mol for FEP).

## Methods

* **Data Source:** MdrDB database with protein-ligand mutant binding affinity (ΔΔG) records.

## Feature Engineering
* **Protein Features:** ESM-2 embeddings, difference between wild-type and mutant protein embeddings.
* **Ligand Features:** ECFP molecular fingerprints (RDKit).

## Data Partitioning Schemes Evaluated
* **Random Partitioning:** Traditional 8:1:1 ratio, training/validation/testing.
* **UniProt-based Partitioning:** Proteins exclusively assigned to either training or test sets, eliminating data leakage.
* **Anchor-Query Partitioning:** Pairwise comparisons using known anchor mutations to predict unknown query mutations.

## Modeling Approaches 
* **Machine Learning:** Random Forest (RF), Support Vector Regression (SVR).
* **Deep Learning:** GRU, BiLSTM, DNN, Transformer encoder.
* **Graph Neural Network Model:** GCN.

## Evaluation Metrics
* **Mean Absolute Error (MAE)**
* **Root Mean Square Error (RMSE)**
* **Pearson correlation coefficients**
* **Spearman correlation coefficients**

## Results Summary
* **Random Partitioning:** Achieved correlations up to 0.70 (Pearson), showing high accuracy but limited practical applicability.
* **UniProt-based Partitioning:** Performance significantly dropped, highlighting real-world challenges in generalization.
* **Anchor-Query Method (Unpaired):** Introducing even small amounts of reference data markedly improved predictive accuracy, providing a practical route for performance enhancement.
* **Anchor-Query Method (paired):** Both unpaired and paired anchor-query strategies were compared, and the paired method provided consistently better performance, especially when reference data were limited.
* **Construct molecular graphs:** The results indicated that the graph convolutional network (GCN) using the molecular graph achieved a Pearson correlation of 0.64.

## Conclusion
This study demonstrates that the choice of data partitioning strategy plays a decisive role in the accuracy and generalizability of protein-ligand binding free energy predictions. While random partitioning often leads to overestimated performance due to data leakage, UniProt-based partitioning provides a more stringent and realistic assessment, closely mirroring real-world scenarios. Additionally, our anchor-query pairwise modeling framework enables substantial improvements in prediction accuracy, even when only a limited amount of reference data is available. These findings emphasize the importance of rigorous partitioning schemes and highlight anchor-query strategies as a promising avenue for robust and practical ΔΔG prediction.

## Usage
1.Environment Setup
* **This project requires Python 3.7.16 and the following main dependencies:** a.PyTorch 1.12.0  b.scikit-learn 1.0.2  c.RDKit 2023.03.2  d.ESM-2 Protein Language Model  e.CUML

2.Data Preparation
* **Download and place the original MdrDB dataset (containing protein sequences, mutation records, ligand SMILES, and experimental ΔΔG values) into the data/directory.**
* **The dataset is available at: https://quantum.tencent.com/mdrdb/**

3.Feature Extraction

a. Data preprocessing
* **Protein features:** Generate wild-type and mutant protein embeddings using ESM-2 by running: 
```bash
python ESM-2_Wild_embedding.py 
python ESM-2_mutation.py
```
* **Ligand features:** Generate ECFP fingerprints from ligand SMILES with RDKit:
```bash
python Script_ECFP.py
```
* **The scripts will automatically merge protein and ligand features for model input.**

b. Comparison of protein language models
* **Comparing ESMC and ProtT5:**
```bash
step1: python ESMC_embed_list.py   
       python ProtT5_embed_list.py
step2: python All_X-_4179_RF_best_ESMC.py  
       python All_X-_4179_RF_best_ProtT5.py
```

c. Sequence and SMILES similarity analysis
```bash
python blast_sequence.py 
python rdkit_smiles_random.py
...
```

4.Model Training and Evaluation

a. Machine learning and deep learning models
* **Three partitioning strategies are supported (random, uniprot, anchor_query), including both paired and unpaired anchor-query modes. Select the partitioning scheme via command-line arguments:** random, UniProt, anchor_query.

* **The training script will load features, perform data partitioning, train the selected model(s), and save the best checkpoints to the script/directory:** 
```bash
(1) python All_X-_4179_RF.py  
    python All_X-_4179_RF_best.py
    ...
(2) python All_X-_4179_UNIPORT_RF.py 
    python All_X-_4179_UNIPORT_RF_best.py
    ...
(3) python All_X-_4179_RF_Q13315_random.py
    python All_X-_4179_RF_Q13315_UNIPORT.py
    ...
(4) python All_X-_Data_pairing_UNIPROT_Q13315_RF_pred_0.1~0.9.py
    ...
```
b. Graph Convolutional Network (GCN)
* **A molecular graph between the protein pocket and the ligand was constructed to verify the performance:**
```bash
python Interaction_Pred_GCN.py
```

5.Prediction and Analysis
* **Predict ΔΔG values for new samples or the test set using a trained model.** 
* **Evaluation metrics (MAE, RMSE, Pearson, Spearman) and visualization outputs (correlation plots, error distributions) will be saved in the results/directory:**
```bash
(1) python All_X-_4179_RF_plot_pearson_random.py
    ...
(2) python All_X-_4179_RF_plot_pearson_UNIPORT.py
    ...
(3) python All_X-_RF_Q13315_P00533_P04637_plot_combined.py
    python 
    All_X-_RF_Q13315_P00533_P04637_plot_combined_UNIPROT.py
    ...
(4) python 
    All_X-_pairing_UNIPROT_Q13315_RF_pred_box_combined.py
    ...
(5) python All_esm_X-_model_plot_random_box.py
    python All_esm_X-_model_plot_uniport_box.py
```
* **For further analysis and customized visualization, refer to the notebooks in the notebooks/directory.**

6.Benchmark Comparison

a. ABL Kinase
* **Our method achieves comparable accuracy to physics-based approaches such as Free Energy Perturbation (FEP) on ABL kinase mutation datasets, with a binding free energy RMSE of 0.87 kcal/mol (vs. 1.11 kcal/mol for FEP).**
* **In mutation classification (susceptible vs. resistant, using a 1.36 kcal/mol threshold), our method achieved an accuracy of 0.83, slightly below the 0.89 of FEP.**
* **The Area Under the Curve (AUC) reached 0.62, outperforming the 0.56 obtained by thermodynamic integration with molecular dynamics simulations.**
```bash
python All_X-_Data_pairing_UNIPROT_P00520_P00519_RF_pred.py
python All_X-_4179_RF_P00520_P00519_Calculate_Pearson_RMSE.py
python All_X-_RF_P00520_P00519_Calculate_accuracy_auc_RMSE_plot.py
python plot_RF_P00520_P00519_bind_distal_mediate_0.1_0.3.py
```
b. Y1 receptor
* **We extended the benchmark to the human neuropeptide Y1 receptor (UniProt ID: P25929, PDB ID: 5ZBQ) — a G-protein-coupled receptor (GPCR) system that presents distinct conformational and physicochemical properties compared with kinases.**
```bash
python 5zbq_RF_pred_unpaired.py
python 5zbq_mut_FEP.py
```

c. EGFR
* **To further assess the generalization ability of the proposed model, we conducted validation experiments on the Epidermal Growth Factor Receptor (EGFR, UniProt ID: P00533) system — a clinically important tyrosine kinase frequently associated with drug resistance mutations.**
```bash
python All_X-_RF_P00533_random_pair_0.1_0.3_Analyze.py
python plot_RF_P00533_binding_distal_mediate_0.1_0.3.py
```
* **For more detailed analysis, please refer to the files in the Supplementary_data/ directory.**

7.Reproducibility
* **All experiments support random seed setting for reproducibility.**
* **Scripts are fully parameterized to facilitate batch runs and repeated trials.**

