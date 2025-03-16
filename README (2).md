# Protein Network Classification with GNN models and GNNExplainer (Protein GNN)

This project aims to leverage Graph Neural Networks (GNNs) to classify protein networks and, more importantly, identify and explain the unique regions or features that distinguish different classes of these networks. By treating proteins as nodes in a graph, with their spatial coordinates as node attributes and edge weights derived from inter-protein distances, GNNs enable the discovery of discriminatory patterns, such as variations in point density or other spatial features, that are characteristic of specific conditions. The ultimate goal is to help biologists better understand how conditions such as healthy versus diseased states affect protein distributions by highlighting the most influential regions or subsets of points within the network. This approach not only provides a powerful classification model but also incorporates explainability through tools like explainable AI and weak supervision, offering actionable insights into the underlying biological processes.


## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/r/personal/hamarneh_sfu_ca/Documents/TEACHING/CMPT340_FALL2024/FOR_STUDENTS/ProjectGroup_Timesheets/ProjectGroup_14_Timesheet.xlsx?d=w5bd9fa05109d40b49a697d0ebe9245ef&csf=1&web=1&e=p4QRU9) | [Slack channel](https://cmpt340fall2024.slack.com/archives/C07JBHED8S3) | [Project report](https://www.overleaf.com/project/66d0b0dd115523de9bef35db) |
|-----------|---------------|-------------------------|

## Video/demo/GIF
[![Watch the video](https://i.imgur.com/sx0NhNP.jpeg)](https://youtu.be/AXzvWi9ZU4Y )


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)


<a name="demo"></a>
## 1. Example demo

test_train.py trains 3 different GNN models: PointNet, GCT, and GGSNN. The models are first trained on synthetic data and then on real data. 
Please see below for where to find the csv files for the data and their respective classes.

```bash
repository
├── src                                     ## source code of the package itself
    ├── GNN                                 ## code, data and plots found here   
        ├── data                            ## synthetic and real data csv files
            ├── class_1_real                ## real data: HeLa DMSO control
	    ├── class_1_simulated           ## synthetic data: uniformly distributed
            ├── class_2_real                ## real data: HeLa Pitstop
	    ├── class_2_simulated           ## synthetic data: uniformly distributed with clusters
```

For our real SMLM data, we were shared a dataset from a biology collaborator that contained experiments with a protein clathrin. Clathrin is a key protein in endocytosis, which plays a crucial part in cellular transport mechanisms. These cellular transportation mechanisms can have potential implications for cancer, as cancerous cells exploit these pathways for rapid growth and survival. The dataset contains a replica of an experiment which compared HeLa cells (cervical cancer cells) with 4 different conditions. Out of the 4 different shared conditions, we focused on HeLa DMSO and HeLa Pitstop. HeLa DMSO is used as a control and Pitstop has been found to disrupt clathrin, helping stop cancer progression. We chose to analyze HeLa DMSO versus HeLa Pitstop, as this comparison demonstrated the greatest differences in molecular distributions.

Model Evaluation Results (on real data):

	•	PointNet: Accuracy = 0.8788, Precision = 0.8804, Recall = 0.8788, F1 Score = 0.8788, AUC = 0.9329
	•	GCNNet: Accuracy = 0.9545, Precision = 0.9549, Recall = 0.9545, F1 Score = 0.9545, AUC = 0.9954
	•	GGSNN: Accuracy = 0.9394, Precision = 0.9458, Recall = 0.9394, F1 Score = 0.9391, AUC = 1.0000

 As evident from the model evaluation results above, all models demonstrated strong performance, effectively distinguishing point clouds corresponding to different cell types.

### Confusion Matrices for our models on the real protein network data
![Confusion Matrix for GCN](https://github.com/sfu-cmpt340/2024_3_project_14/raw/main/src/GNN/Protein%20GNN%20project%20-%20Plots/sampled%20data/conf_matrix_GCNNet.png)
![Confusion Matrix for GCNNet](https://github.com/sfu-cmpt340/2024_3_project_14/raw/main/src/GNN/Protein%20GNN%20project%20-%20Plots/sampled%20data/conf_matrix_GGSNN.png)
![Confusion Matrix for GCNNet](https://github.com/sfu-cmpt340/2024_3_project_14/raw/main/src/GNN/Protein%20GNN%20project%20-%20Plots/sampled%20data/conf_matrix_PointNet.png)


### GNNExplainer plot examples on the real protein network data (HeLa Pitstop:

GCN

 
![Explanation for GCN](https://github.com/sfu-cmpt340/2024_3_project_14/blob/main/src/GNN/Protein%20GNN%20project%20-%20Plots/sampled%20data/explanation%20graphs/GCN/Class%202_Explanation/PC_34.png)


GGSNN
 
![Explanation for GGSNN](https://github.com/sfu-cmpt340/2024_3_project_14/blob/main/src/GNN/Protein%20GNN%20project%20-%20Plots/sampled%20data/explanation%20graphs/GGSNN/Class%202_Explanation/PC_34.png)

PointNet

![Explanation for PointNet](https://github.com/sfu-cmpt340/2024_3_project_14/blob/main/src/GNN/Protein%20GNN%20project%20-%20Plots/sampled%20data/explanation%20graphs/PointNet/Class%202_Explanation/PC_34.png)


As shown in the examples above, the nodes identified as most important by GNNExplainer are highlighted in yellow and displayed as larger points, indicating their significant influence on the model's classification decision.

 
### What to find where


```bash
repository
├── src                                     ## source code of the package itself
    ├── GNN                                 ## code, data and plots found here   
        ├── data                            ## synthetic and real data here
        ├── Protein GNN project - Plots     ## plots and confusion matrices here  
├── README.md                               ## You are here
├── requirements.yml                        ## dependency file
```

<a name="installation"></a>

## 2. Installation

Before running this project, please make sure the following are installed to ensure necessary dependencies are set up for running our project smoothly:
- Python 3.10
  - Strictly use this version for compatibility.
  - Download it from [*Python's official website*](https://www.python.org/).
- Anaconda Conda
  - Any version is acceptable.
  - Install from [*Conda's official website*](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
 
> **Notes for Windows:**  
> - During installation, select the option to add Anaconda to system PATH.  
> - Alternatively, use the `anaconda_prompt` in Anaconda Navigator to avoid PATH issues.  
> - Finally, run Conda commands either using `cmd` (if PATH is configured) or `anaconda_prompt` for seamless execution.  

> **Notes for Linux/Mac:**  
> - Conda typically works out of the box.  
> - PATH is configured automatically during installation.  

To get started with this project, clone the repository and create a new conda environment using the provided `requirements.yml` file. This will automatically install all the necessary dependencies:
```bash
git clone https://github.com/sfu-cmpt340/2024_3_project_14.git
cd 2024_3_project_14
conda env create -f requirements.yml
conda activate protein-gnn
```
If you need to deactivate the environment at any point, use:
```bash
conda deactivate
```

Alternative Library Installation Guide: 
Our project leverages several Python libraries for data processing, visualization, and graph neural network (GNN) model training. These include PyTorch, PyTorch Geometric, and supporting tools like Scikit-learn and Seaborn. If you face any issues with the requirements.yml file or prefer manual installation, you can install the required libraries individually by entering the following commands in your terminal or IDE command window:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn networkx                       #core libraries
pip install torch==2.0 torchvision                                                      #pytorch and supporting libraries                     
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric  #pytorch geometric libraries
pip install pybind11                                                                    #additional libraries
```

Notes:
- PyTorch Geometric Compatibility: Please ensure you have the correct CUDA version if using a GPU. Visit the [PyTorch Geometric installation page](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for guidance.
- System Requirements: Some libraries may require Microsoft C++ Build Tools (Windows) or `gcc` (Linux/macOS). Please install these if you encounter errors during installation.

> **Note:**  
> Regardless of your installation preference, you can use Jupyter Notebook or Google Colaboratory, which support most libraries by default.  
> If any libraries are missing, you can install them manually using the commands provided above.

<a name="repro"></a>
## 3. Reproduction
Data to train and test the models are provided in [2024_3_project_14/src/GNN/data](https://github.com/sfu-cmpt340/2024_3_project_14/tree/main/src/GNN/data).

Begin by activating the newly created `protein-gnn` environment:
```bash
conda activate protein-gnn
```

All model training, testing, and evaluations will be completed in the 'test_train.py'. 

**Make sure you are in the `src/GNN` directory**
```bash
cd src/GNN
python3 test_train.py
```

Evaluations for the models (PointNet, GCN, and GGSNN) will output to the console, along with a confusion matrix that can be accessed inside the `Protein GNN project - Plots` folder.


> ### ⚠️ **Important**
> Please ensure you are in the correct environment (`protein-gnn`) and directory (`src/GNN`) before running the script.  
> This is essential to avoid dependency issues and ensure that all required files are accessible.
