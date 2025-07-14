"""
A series of helper functions used throughout the course.
If a function gets defined once and could be used over and over, it'll go in here.
This functions are taken from a course, https://github.com/mrdbourke/pytorch-deep-learning. (with modifications)

MODIFICATIONS 
    - Added plot_functions()
    - Added plot_predictions()

"""
import torch
import matplotlib.pyplot as plt
import numpy as np

import os
import zipfile

from pathlib import Path
from typing import Optional, Callable, Type, Union
import requests

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7) # type: ignore
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu) # type: ignore
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# See creation: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function
from typing import List
import torchvision


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str],
    device : torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    transform=None
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def plot_functions(
    func: Union[Callable, torch.Tensor, np.ndarray],
    x_range: tuple = (-10, 10),
    num_points: int = 1000,
    title: str = 'Function',
    xlabel: str = "x",
    ylabel: str = "f(x)",
    color: str = 'blue',
    linewidth: int = 2,
    linestyle: str = '-',
    grid: bool = True,
    ax=None,
    **kwargs
):
    plt.style.use('seaborn-v0_8')
    
    # Generate x values (NumPy array)
    x_np = np.linspace(x_range[0], x_range[1], num_points)
    
    # Case 1: func is a callable (Python or PyTorch function)
    if callable(func):
        # First check if it's a PyTorch function by testing with a tensor
        test_tensor = torch.tensor([1.0], dtype=torch.float32)
        try:
            # Try calling with tensor (PyTorch function)
            _ = func(test_tensor)
            # If successful, process as PyTorch function
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            y_tensor = func(x_tensor,  **kwargs)
            y = y_tensor.detach().numpy()
        except (TypeError, AttributeError, RuntimeError):
            # If fails, treat as NumPy function
            y = func(x_np)
    # Case 2: func is precomputed y-values (Tensor or NumPy array)
    else:
        y = func.detach().numpy() if isinstance(func, torch.Tensor) else np.array(func)
        x_np = np.linspace(x_range[0], x_range[1], len(y))
    
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Main plot
    scatter = ax.scatter(
        x_np, y,
        c=y,
        cmap='viridis',
        s=50,
        alpha=0.7,
        edgecolor='white',
        linewidth=0.5,
    )
    
    ax.plot(x_np, y, 'k-', alpha=0.3, linewidth=linewidth)
    ax.set_title(title, fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(grid, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=y.min(), vmax=y.max())) # type: ignore
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Function Value', rotation=270, labelpad=15)
    
    plt.tight_layout()
    return ax


def plot_predictions(
    train_data: Union[np.ndarray, torch.Tensor],
    train_labels: Union[np.ndarray, torch.Tensor],
    test_data: Union[np.ndarray, torch.Tensor],
    test_labels: Union[np.ndarray, torch.Tensor],
    predictions: Optional[Union[np.ndarray, torch.Tensor]] = None,
    title: str = "Model Predictions",
    xlabel: str = "Input Data",
    ylabel: str = "Target Values",
    train_color: str = "#185b8b",  # Seaborn default blue
    test_color: str = "#205a20",   # Seaborn default green
    pred_color: str = "#891e1e",   # Seaborn default red
    alpha: float = 0.7,
    s: int = 50,
    grid: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs
):
    """
    Beautifully plots training data, test data, and model predictions with modern styling.
    
    Args:
        train_data: Input training data
        train_labels: Target training values
        test_data: Input test data
        test_labels: Target test values
        predictions: Model predictions on test data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        train_color: Color for training data points
        test_color: Color for test data points
        pred_color: Color for prediction points
        alpha: Transparency of points
        s: Size of points
        grid: Whether to show grid
        ax: Optional matplotlib axis to plot on
        **kwargs: Additional styling arguments
    """
    plt.style.use('seaborn-v0_8')
    
    # Convert torch tensors to numpy if needed
    if isinstance(train_data, torch.Tensor):
        train_data = train_data.detach().numpy()
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.detach().numpy()
    if isinstance(test_data, torch.Tensor):
        test_data = test_data.detach().numpy()
    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.detach().numpy()
    if predictions is not None and isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()
    
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training data
    train_scatter = ax.scatter(
        train_data, train_labels,
        c=train_color,
        s=s,
        alpha=alpha,
        edgecolor='white',
        linewidth=0.5,
        label="Training Data",
        zorder=3,
        **kwargs
    )
    
    # Plot test data
    test_scatter = ax.scatter(
        test_data, test_labels,
        c=test_color,
        s=s,
        alpha=alpha,
        edgecolor='white',
        linewidth=0.5,
        label="Testing Data",
        zorder=3,
        **kwargs
    )
    
    # Plot predictions if provided
    if predictions is not None:
        pred_scatter = ax.scatter(
            test_data, predictions,
            c=pred_color,
            s=s*1.2,  # Slightly larger for emphasis
            alpha=min(alpha + 0.1, 1.0),
            edgecolor='white',
            linewidth=0.7,
            label="Predictions",
            zorder=4,  # Draw on top
            **kwargs
        )
    
    # Styling
    ax.set_title(title, fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    if grid:
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Clean up spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Add legend
    ax.legend(
        loc='upper left',
        frameon=True,
        framealpha=0.9,
        edgecolor='white',
        facecolor='white'
    )
    
    plt.tight_layout()
    return ax