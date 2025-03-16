# Core libraries
import numpy as np  
import pandas as pd  
from matplotlib.pylab import plt 
import seaborn as sns
import itertools 
import os

# PyTorch Geometric modules
from torch_geometric.loader import DataLoader 
import torch  
from torch.nn import Linear  
import torch.nn.functional as F  
from torch_geometric.nn import global_mean_pool, global_max_pool, MessagePassing, GCNConv

# General PyTorch components
from torch import Tensor 
from torch.nn import Sequential, Linear, ReLU

# Custom utility modules
from graph_construction import prepare_graph_data
from preprocessing import preprocess_data 
from models import PointNet, GCNNet, GGSNN

# Scikit-learn modules for evaluation and data splitting
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold  
from sklearn.metrics import ( 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# For GNNExplainer
from torch_geometric.explain import Explainer, GNNExplainer
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from torch_geometric.utils import to_networkx

########### Import completed

# Set seed to ensure reproducible code
seed = 100
np.random.seed(seed)

# Data Preprocessing 
def load_dataset(path1,path2):
    class_1 = []
    class_2 = []
    for i in range(100):
        p1 = path1 + '/point_cloud_' + str(i+1) + '.csv'
        point_cloud1 = preprocess_data(p1)
        graph_data1 = prepare_graph_data(point_cloud1)
        graph_data1.y = torch.tensor(0)
        class_1.append(graph_data1)

        p2 = path2 + '/point_cloud_' + str(i+1) + '.csv'
        point_cloud2 = preprocess_data(p2)
        graph_data2 = prepare_graph_data(point_cloud2)
        graph_data2.y = torch.tensor(1)
        class_2.append(graph_data2)
    data_full = class_1 + class_2
    return data_full

def setting(model, lr, decay, step_size, gamma):
    """Set up optimizer, scheduler, and loss criterion for model training"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()
    return optimizer, scheduler, criterion

def plot_confusion_matrix(conf_matrix, direct, labels, model=None):
    """Plot and save a confusion matrix as images"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix of " + str(model))
    filename = direct + '/conf_matrix_' + str(model) + '.png'
    plt.savefig(filename)
    plt.close()


def full_run(path1,path2,direct):
    """A complete training and testing run.
    Inputs:
    path1: Path to folders containing csv files of Class 1 point clouds
    path2: Path to folders containing csv files of Class 2 point clouds
    direct: directory to save the models' confusion matrices
    Output: 
    - Performance metrics of 3 
    - training and testing data (to support GNNExplainer integration)"""

    data_full = load_dataset(path1,path2)
    # Hyperparameter grid for tuning
    param_grid = {
        'lr': [0.0001, 0.001, 0.01],  # Explore smaller and larger learning rates
        'decay': [1e-5, 1e-4, 5e-4],  # Wider range for weight decay (L2 regularization)
        'step_size': [5, 10, 15],  # Explore different learning rate decay step sizes
        'gamma': [0.3, 0.5, 0.7],  # Different values to control the learning rate decay
    }
    param_combinations = list(itertools.product(*param_grid.values()))
    
    # Training function to optimize model parameters using training data
    def train(model, optimizer, scheduler, criterion):
        model.train()
        total_loss = 0
        total_correct = 0
        for data in train_loader:
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index, data.batch)
            loss = criterion(logits, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Gradient clipping
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
            # Calculate accuracy
            pred = logits.argmax(dim=-1)
            total_correct += int((pred == data.y).sum())

        # Update the learning rate based on the scheduler
        scheduler.step()

        train_loss = total_loss / len(train_loader.dataset)
        train_accuracy = total_correct / len(train_loader.dataset)
        return (train_loss, train_accuracy)

    # Testing function to evaluate model performance on test data
    @torch.no_grad()
    def test(model, criterion):
        model.eval()  # Set the model to evaluation mode
        total_loss = 0
        total_correct = 0
        predictions = []
        for data in test_loader:
            logits = model(data.x, data.edge_index, data.batch)
            loss = criterion(logits, data.y)
            total_loss += float(loss) * data.num_graphs
                # Calculate accuracy
            pred = logits.argmax(dim=-1)
            predictions.extend(pred.cpu().numpy())  # Convert predictions to a list
            total_correct += int((pred == data.y).sum())
        valid_loss = total_loss / len(test_loader.dataset)
        valid_accuracy = total_correct / len(test_loader.dataset)
        return valid_loss, valid_accuracy, predictions

    # Initialize K-Fold Cross Validation
    kf = KFold(n_splits=3, shuffle=True, random_state=100)

    # Define models in a dictionary
    models_dict = {
        'PointNet': PointNet(),
        'GCNNet': GCNNet(num_classes=2),
        'GGSNN': GGSNN(hidden_dim=32, num_classes=2)
    }

    # Store results for all models
    all_results = []

    # Perform hyperparameter tuning with cross-validation
    for model_name, model in models_dict.items():
        print(f"\nTraining and evaluating model: {model_name}")

        fold_metrics = []
        model_results = {'Model': model_name}

        # Separate tracking of the best parameters and model for each model
        best_model = None
        best_params = None
        best_metrics = {'accuracy': 0}  # Reset for each model

        for params in param_combinations:
            lr, decay, step_size, gamma = params
            fold_metrics = []
            print(f"Testing params: lr={lr}, decay={decay}, step_size={step_size}, gamma={gamma}")

            for train_idx, test_idx in kf.split(data_full):
                train_data = [data_full[i] for i in train_idx]
                test_data = [data_full[i] for i in test_idx]

                train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
                test_loader = DataLoader(test_data, batch_size=4)

                # Initialize model and optimizer
                optimizer, scheduler, criterion = setting(model, lr, decay, step_size, gamma)

                # Train the model
                train(model, optimizer, scheduler, criterion)

                # Test the model
                test_loss, test_accuracy, predictions = test(model, criterion)
                fold_metrics.append(test_accuracy)

            # Average metrics across folds
            mean_accuracy = sum(fold_metrics) / len(fold_metrics)
            if mean_accuracy > best_metrics['accuracy']:
                best_model = model
                best_params = params
                best_metrics['accuracy'] = mean_accuracy

            model_results['Best Params'] = best_params

        # Evaluate the best model on the final test set
        final_predictions = []
        final_labels = []
        final_probs = []  # Store the prediction probabilities for ROC

        for data in test_loader:
            logits = best_model(data.x, data.edge_index, data.batch)
            probs = torch.nn.functional.softmax(logits, dim=-1)  # Get probabilities from logits
            pred = probs.argmax(dim=-1).cpu().numpy()  # Get predicted labels
            final_predictions.extend(pred)
            final_labels.extend(data.y.cpu().numpy())
            final_probs.extend(probs.detach().cpu().numpy())  # Detach and then convert to numpy for ROC AUC

        # Calculate detailed metrics
        accuracy = accuracy_score(final_labels, final_predictions)
        precision = precision_score(final_labels, final_predictions, average='weighted')
        recall = recall_score(final_labels, final_predictions, average='weighted')
        f1 = f1_score(final_labels, final_predictions, average='weighted')
        conf_matrix = confusion_matrix(final_labels, final_predictions)

        # Export confusion matrix as image 
        os.makedirs(direct, exist_ok=True)
        plot_confusion_matrix(conf_matrix, direct = direct, labels=["Class 1", "Class 2"], model=model_name) 

        # AUC ROC Curve
        fpr, tpr, thresholds = roc_curve(final_labels, [prob[1] for prob in final_probs])  # Use positive class probabilities
        auc = roc_auc_score(final_labels, [prob[1] for prob in final_probs])  # Calculate AUC score

        # Add metrics to the model's results
        model_results['Accuracy'] = accuracy
        model_results['Precision'] = precision
        model_results['Recall'] = recall
        model_results['F1 Score'] = f1
        model_results['AUC'] = auc
        model_results['Confusion Matrix'] = conf_matrix

        # Store the results for the current model
        all_results.append(model_results)

    # Create a DataFrame to display the results
    results_df = pd.DataFrame(all_results)

    # Print the results table
    print("\nModel Evaluation Results:\n")
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']])
    return train_loader, test_loader

############ Initialize folder paths
syn_1 = 'data/class_1_simulated' #Path to synthetic data
syn_2 = 'data/class_2_simulated'

real_1 = 'data/class_1_real' #Path to real data
real_2 = 'data/class_2_real'

syn_data_dir = "Protein GNN project - Plots/synthetic data" #Output directory
real_data_dir = "Protein GNN project - Plots/sampled data"

############ Complete train and test
syn_train, syn_test = full_run(syn_1,syn_2,syn_data_dir)
real_train, real_test = full_run(real_1,real_2,real_data_dir)

############ GNNExplainer
"""The following code will generate explanation graphs for PCs in Test set. 
Explanation graphs are 3D visualization of the nodes, colored based on their importance."""
explainer_PN = Explainer(
    model= PointNet(),
    algorithm=GNNExplainer(),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    ),
)

explainer_GCN = Explainer(
    model= GCNNet(num_classes=2),
    algorithm=GNNExplainer(),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    ),
)

explainer_GGSNN= Explainer(
    model= GGSNN(hidden_dim=32, num_classes=2),
    algorithm=GNNExplainer(),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    ),
)
def get_explanation(pt, exp, model, data_loader, folder):
    data = data_loader.dataset[pt]
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, batch=data.batch)  # Raw model outputs (logits)
        predicted_value = logits.argmax(dim=-1)
        
    explanation = exp(data.x, data.edge_index, batch=data.batch)
    node_mask = explanation.node_mask 
    edge_mask = explanation.edge_mask 
    node_importance = node_mask.mean(axis=1)
    
    # Normalize node_importance for transparency and size scaling
    min_val = node_importance.min()
    max_val = node_importance.max()
    normalized_importance = (node_importance - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero

    alphas = 1 - normalized_importance
    sizes = 50 + 200 * normalized_importance  # Adjust scale factors as needed
    
    # Plot settings
    G = to_networkx(data, to_undirected=True)
    pos_array = data.x.detach().cpu().numpy() 
    fig = plt.figure(figsize=(8, 6)) 
    ax = fig.add_subplot(111, projection='3d')
    colormap = plt.cm.viridis  
    mapped_colors = colormap(normalized_importance)
    
    if (data.y == 0):
        text = 'PC ' + str(pt) + ' with True Y = 0 (Class 1), Predicted Y = '  + str(int(predicted_value))
        filename = folder + 'Class 1_Explanation/PC_' + str(pt)
    else:
        text = 'PC ' + str(pt) + ' with True Y = 1 (Class 2), Predicted Y = '  + str(int(predicted_value))
        filename = folder + 'Class 2_Explanation/PC_' + str(pt)
        
    # Plot nodes in 3D with colors based on node mask
    sc = ax.scatter(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], 
                    c=mapped_colors, s=sizes, alpha=alphas, edgecolors='w', linewidth=0.5)
    
    plt.title(text)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label('Node Importance (Mean)')
    plt.savefig(filename)
    plt.close()
    
#Create directory for output
explanation_path_syn = 'Protein GNN project - Plots/synthetic data/explanation graphs/'
explanation_path_real = 'Protein GNN project - Plots/sampled data/explanation graphs/'
#PointNet
os.makedirs(explanation_path_syn + "PointNet/Class 1_Explanation", exist_ok=True) 
os.makedirs(explanation_path_real+ "PointNet/Class 1_Explanation", exist_ok=True) 
os.makedirs(explanation_path_syn + "PointNet/Class 2_Explanation", exist_ok=True) 
os.makedirs(explanation_path_real+ "PointNet/Class 2_Explanation", exist_ok=True) 
#GCN
os.makedirs(explanation_path_syn + "GCN/Class 1_Explanation", exist_ok=True) 
os.makedirs(explanation_path_real+ "GCN/Class 1_Explanation", exist_ok=True) 
os.makedirs(explanation_path_syn + "GCN/Class 2_Explanation", exist_ok=True) 
os.makedirs(explanation_path_real+ "GCN/Class 2_Explanation", exist_ok=True) 
#GGSNN
os.makedirs(explanation_path_syn + "GGSNN/Class 1_Explanation", exist_ok=True) 
os.makedirs(explanation_path_real+ "GGSNN/Class 1_Explanation", exist_ok=True) 
os.makedirs(explanation_path_syn + "GGSNN/Class 2_Explanation", exist_ok=True) 
os.makedirs(explanation_path_real+ "GGSNN/Class 2_Explanation", exist_ok=True) 

for i in range(len(syn_test.dataset)):
    #PointNet
    print('Generating PointNet Explanation...')
    get_explanation(i, explainer_PN, PointNet(), syn_test, explanation_path_syn + "PointNet/")
    get_explanation(i, explainer_PN, PointNet(), real_test, explanation_path_real+ "PointNet/")
    #GCN
    print('Generating GCN Explanation...')
    get_explanation(i, explainer_GCN, GCNNet(num_classes=2), syn_test, explanation_path_syn + "GCN/")
    get_explanation(i, explainer_GCN, GCNNet(num_classes=2), real_test, explanation_path_real+ "GCN/")
    #GGSNN
    print('Generating GGSNN Explanation...')
    get_explanation(i, explainer_GGSNN, GGSNN(hidden_dim=32, num_classes=2), syn_test, explanation_path_syn + "GGSNN/")
    get_explanation(i, explainer_GGSNN, GGSNN(hidden_dim=32, num_classes=2), real_test, explanation_path_real+ "GGSNN/")

print('All explanation graphs saved.')

print('All explanation graphs saved.')
