import numpy as np
from matplotlib.colors import to_hex
import matplotlib.cm as cm
import torch
from typing import Union

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.express import histogram

def compare_plot(datasets, titles=None, figsize=(1080, 600), colorscale='Viridis', 
                 line_width=3, marker_size=2, bgcolor='rgb(240, 240, 240)'):
    """
    Create beautiful interactive horizontal subplots for datasets of varying dimensions using Plotly.
    
    Args:
        datasets (list): List of numpy arrays (1D, 2D, or 3D)
        titles (list): Optional list of titles for each subplot
        figsize (tuple): Figure size (width, height)
        colorscale: Plotly colorscale name
        line_width: Width of plot lines
        marker_size: Size of start/end markers
        bgcolor: Background color
    """
    n = len(datasets)
    if titles is None:
        titles = [f'Dataset {i+1}' for i in range(n)]
    
    # Check if we need 3D subplots and create appropriate specs
    specs = []
    for data in datasets:
        if data.ndim > 1 and data.shape[1] == 3:
            specs.append({'type': 'scatter3d'})
        else:
            specs.append({'type': 'xy'})
    
    # Create horizontal subplots (1 row, n columns)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=titles,
        specs=[specs],  # Note: specs is now a list of dicts for columns
        horizontal_spacing=0.1 if n > 1 else 0.05
    )
    
    for i, data in enumerate(datasets):
        col = i+1
        
        if data.ndim == 1:
            # 1D data
            fig.add_trace(go.Scatter(
                y=data,
                mode='lines',
                line=dict(width=line_width),
                name=f'Dataset {i+1}',
                hoverinfo='y'
            ), row=1, col=col)
            
        elif data.shape[1] == 2:
            # 2D data
            fig.add_trace(go.Scatter(
                x=data[:,0],
                y=data[:,1],
                mode='lines',
                line=dict(width=line_width, color='green'),
                name=f'Dataset {i+1}',
                hoverinfo='x+y'
            ), row=1, col=col)
            
            # Add start/end markers
            fig.add_trace(go.Scatter(
                x=[data[0,0]],
                y=[data[0,1]],
                mode='markers',
                marker=dict(size=marker_size, color='limegreen'),
                name='Start',
                showlegend=False,
                hoverinfo='none'
            ), row=1, col=col)
            
            fig.add_trace(go.Scatter(
                x=[data[-1,0]],
                y=[data[-1,1]],
                mode='markers',
                marker=dict(size=marker_size, color='crimson'),
                name='End',
                showlegend=False,
                hoverinfo='none'
            ), row=1, col=col)
            
        elif data.shape[1] == 3:
            # 3D data
            fig.add_trace(go.Scatter3d(
                x=data[:,0],
                y=data[:,1],
                z=data[:,2],
                mode='lines',
                line=dict(width=line_width, color=data[:,2], colorscale=colorscale),
                name=f'Dataset {i+1}',
                hoverinfo='x+y+z'
            ), row=1, col=col)
            
            # Add start/end markers
            fig.add_trace(go.Scatter3d(
                x=[data[0,0]],
                y=[data[0,1]],
                z=[data[0,2]],
                mode='markers',
                marker=dict(size=marker_size, color='limegreen'),
                name='Start',
                showlegend=False,
                hoverinfo='none'
            ), row=1, col=col)
            
            fig.add_trace(go.Scatter3d(
                x=[data[-1,0]],
                y=[data[-1,1]],
                z=[data[-1,2]],
                mode='markers',
                marker=dict(size=marker_size, color='crimson'),
                name='End',
                showlegend=False,
                hoverinfo='none'
            ), row=1, col=col)
            
            # Update 3D scene settings
            fig.update_scenes(
                aspectmode='data',
                xaxis=dict(gridcolor='rgb(200, 200, 200)', showbackground=True),
                yaxis=dict(gridcolor='rgb(200, 200, 200)', showbackground=True),
                zaxis=dict(gridcolor='rgb(200, 200, 200)', showbackground=True),
                bgcolor=bgcolor,
                row=1, col=col
            )
    
    # Update layout
    fig.update_layout(
        height=figsize[1],
        width=max(figsize[0], 300 * n),  # Ensure enough width for all subplots
        margin=dict(l=50, r=50, b=50, t=80),
        paper_bgcolor='white',
        plot_bgcolor=bgcolor,
        showlegend=False,
        font=dict(family='Arial', size=12))
    
    fig.show()

def plot_components(trajectory, time=None, labels=None, title=None, 
                   figsize=(1080,600), colorscale='Viridis', line_width=2.5,
                   bgcolor='rgb(240, 240, 240)', title_fontsize=20):
    """
    Create beautiful interactive component plot using Plotly.
    
    Args:
        trajectory (np.ndarray): Shape (n_points, n_components)
        time (np.ndarray): Custom time values (default: indices)
        labels (list): Component names (e.g., ['x', 'y', 'z'])
        title (str): Overall title
        figsize (tuple): Figure size (width, height)
        colorscale: Plotly colorscale name
        line_width: Width of plot lines
        bgcolor: Background color
        title_fontsize: Font size for title
    """
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    n_components = trajectory.shape[1]
    time = np.arange(trajectory.shape[0]) if time is None else time
    
    if labels is None:
        labels = [f'Component {i+1}' for i in range(n_components)]
    
    # Create subplot figure
    fig = make_subplots(
        rows=n_components, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05, 
        subplot_titles="",
        specs=[[{'type': 'xy'}] for _ in range(n_components)]  # Proper 2D specs structure
    )
    
    # Get colors from colorscale
    colors = [to_hex(cm.get_cmap('viridis') (i/n_components)) for i in range(n_components)]
    
    for i in range(n_components):
        fig.add_trace(go.Scatter(
            x=time,
            y=trajectory[:,i],
            mode='lines',
            line=dict(width=line_width, color=colors[i]),
            #name=labels[i],
            hoverinfo='x+y',
            showlegend=False
        ), row=i+1, col=1)
        
        # Customize each subplot
        fig.update_yaxes(title_text=labels[i], row=i+1, col=1)
        fig.update_xaxes(showgrid=True, gridcolor='rgb(200, 200, 200)', row=i+1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='rgb(200, 200, 200)', row=i+1, col=1)
    
    # Update layout
    fig.update_layout(
        height=figsize[1],
        width=figsize[0],
        title={
            'text': f"<b>{title}</b>" if title else "",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=title_fontsize, family='Arial')
        },
        margin=dict(l=50, r=50, b=50, t=80 if title else 50),
        paper_bgcolor='white',
        plot_bgcolor=bgcolor,
        hovermode='x unified',
        font=dict(family='Arial', size=12))
    
    # Only show x-axis label on bottom plot
    #fig.update_xaxes(title_text='Time', col=1)   
    fig.show()


def plot_multidimensional(results, system_name, save_html=False):
    """
    Plot all dimensions of the system with interactive controls.
    
    Args:
        results: List of result dictionaries containing:
            - 'true_value': numpy array of true values
            - 'predictions': torch tensor of predictions
            - 'parameters': dictionary of parameters
            - 'metrics': dictionary containing 'RMSE'
        system_name: Name of the dynamical system
        save_html: Whether to save as interactive HTML file
    
    Returns:
        plotly.graph_objects.Figure: Interactive figure with dropdown
    """
    # Create figure
    fig = go.Figure()
    
    # Create visibility matrix and button definitions
    buttons = []
    visible_matrix = []
    param_strings = []  # To store formatted parameter strings
    
    for i, result in enumerate(results):
        true_vals = result['true_value']
        preds = result['predictions'].cpu().numpy() if hasattr(result['predictions'], 'cpu') else result['predictions']
        params = result['parameters']
        n_dims = true_vals.shape[1]
        
        # Format parameters for display
        param_str = "<br>".join([f"{k}: {v}" for k, v in params.items()])
        param_strings.append(param_str)
        
        # Create visibility array for this parameter set
        visible = [False] * (len(results) * n_dims * 2)  # (true + pred) * dims * results
        start_idx = i * n_dims * 2
        
        # Add traces for each dimension
        for dim in range(n_dims):
            # True values trace
            fig.add_trace(go.Scatter(
                x=np.arange(len(true_vals)),
                y=true_vals[:, dim],
                name=f'True Dim {dim+1}',
                line=dict(color=px.colors.qualitative.Plotly[dim]),
                visible=(i==0)  # Only show first set by default
            ))
            
            # Predictions trace
            fig.add_trace(go.Scatter(
                x=np.arange(len(preds)),
                y=preds[:, dim],
                name=f'Pred Dim {dim+1}',
                line=dict(color=px.colors.qualitative.Plotly[dim], dash='dash'),
                visible=(i==0)
            ))
            
            # Set visibility for this parameter set
            visible[start_idx + dim*2] = True      # True values
            visible[start_idx + dim*2 + 1] = True   # Predictions
        
        visible_matrix.append(visible)
        
        # Create button for this parameter set
        buttons.append(dict(
            label=f"Params {i+1} (RMSE: {result['metrics']['RMSE']:.3f})",
            method="update",
            args=[{"visible": visible_matrix[i]},
                  {"title": {
                      "text": f"{system_name} - Set {i+1} (RMSE: {result['metrics']['RMSE']:.3f})<br><span style='font-size: 12px;'>{param_str}</span>",
                      "x": 0.5,
                      "xanchor": "center"
                  }}]
        ))
    
    # Initial title with first parameter set
    initial_title = {
        "text": f"{system_name} - Set 1 (RMSE: {results[0]['metrics']['RMSE']:.3f})<br><span style='font-size: 12px;'>{param_strings[0]}</span>",
        "x": 0.5,
        "xanchor": "center"
    }
    
    # Update layout with dropdown menu
    fig.update_layout(
        title=initial_title,
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.1,
            "y": 1.2,  # Moved slightly higher to accommodate longer title
            "xanchor": "left",
            "yanchor": "top"
        }],
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=120)  # Increase top margin for longer title
    )
    
    if save_html:
        fig.write_html(f"{system_name}_predictions.html")
    else: fig.show()


def plot_multidimensional_3d(results, system_name, pp: int, save_html=False, path: str = "Examples/Input_Discretization/Plots/3DPlots/", show: bool = False):
    """
    Plot 3D trajectories of the system with interactive controls.
    
    Args:
        results: List of result dictionaries containing:
            - 'true_value': numpy array of true values (must be 3D)
            - 'predictions': torch tensor of predictions (must be 3D)
            - 'parameters': dictionary of parameters
            - 'metrics': dictionary containing 'RMSE'
        system_name: Name of the dynamical system
        save_html: Whether to save as interactive HTML file
    
    Returns:
        plotly.graph_objects.Figure: Interactive 3D figure with dropdown
    """
    # Create figure
    fig = go.Figure()
    
    # Create visibility matrix and button definitions
    buttons = []
    visible_matrix = []
    param_strings = []  # To store formatted parameter strings
    
    for i, result in enumerate(results):
        true_vals = result['true_value']
        preds = result['predictions'].cpu().numpy() if hasattr(result['predictions'], 'cpu') else result['predictions']
        params = result['parameters']
        
        # Verify data is 3D
        if true_vals.shape[1] != 3 or preds.shape[1] != 3:
            raise ValueError("Data must be 3-dimensional (shape: [n_points, 3])")
        
        # Format parameters for display
        param_str = "<br>".join([f"{k}: {v}" for k, v in params.items()])
        param_strings.append(param_str)
        
        # Create visibility array for this parameter set
        visible = [False] * (len(results) * 2)  # (true + pred) * results
        
        # True values trace
        fig.add_trace(go.Scatter3d(
            x=true_vals[:, 0],
            y=true_vals[:, 1],
            z=true_vals[:, 2],
            name=f'True Trajectory {i+1}',
            marker=dict(size=2),
            line=dict(color='grey', width=6),
            visible=(i==0)  # Only show first set by default
        ))
        
        # Predictions trace
        fig.add_trace(go.Scatter3d(
            x=preds[:, 0],
            y=preds[:, 1],
            z=preds[:, 2],
            name=f'Predicted Trajectory {i+1}',
            marker=dict(size=2),
            line=dict(color='darkorange', width=7),
            visible=(i==0)
        ))
        
        # Set visibility for this parameter set
        visible[i*2] = True    # True values
        visible[i*2+1] = True  # Predictions
        
        visible_matrix.append(visible)
        
        # Create button for this parameter set
        buttons.append(dict(
            label=f"Params {i+1} (RMSE: {result['metrics']['RMSE']:.3f})",
            method="update",
            args=[{"visible": visible_matrix[i]},
                  {"title": {
                      "text": f"{system_name} - Set {i+1} (RMSE: {result['metrics']['RMSE']:.3f})<br><span style='font-size: 12px;'>{param_str}</span>",
                      "x": 0.5,
                      "xanchor": "center"
                  },
                  "scene": {  # Reset camera view when switching
                      "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 0.5}},
                      # Apply axis settings here
                      "xaxis": dict(
                          showline=True,
                          linecolor="black",
                          linewidth=5,
                          showgrid=False,
                          zeroline=False,
                          showticklabels=False,
                          ticks="",
                          title=""
                      ),
                      "yaxis": dict(
                          showline=True,
                          linecolor="black",
                          linewidth=5,
                          showgrid=False,
                          zeroline=False,
                          showticklabels=False,
                          ticks="",
                          title=""
                      ),
                      "zaxis": dict(
                          showline=True,
                          linecolor="black",
                          linewidth=5,
                          showgrid=False,
                          zeroline=False,
                          showticklabels=False,
                          ticks="",
                          title=""
                      )
                  }}]
        ))
    
    # Initial title with first parameter set
    initial_title = {
        "text": f"{system_name} - Set 1 (RMSE: {results[0]['metrics']['RMSE']:.3f})<br><span style='font-size: 12px;'>{param_strings[0]}</span>",
        "x": 0.5,
        "xanchor": "center"
    }
    
    # Update layout with dropdown menu
    fig.update_layout(
        title=initial_title,
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.1,
            "y": 1.2,
            "xanchor": "left",
            "yanchor": "top"
        }],
        scene=dict(
            xaxis=dict(
                showline=True,
                linecolor="black",
                linewidth=5,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                ticks="",
                title=""
            ),
            yaxis=dict(
                showline=True,
                linecolor="black",
                linewidth=5,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                ticks="",
                title=""
            ),
            zaxis=dict(
                showline=True,
                linecolor="black",
                linewidth=5,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                ticks="",
                title=""
            ),
            aspectmode='data',  # Preserve aspect ratio
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.5)  # Initial camera position
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        margin=dict(t=120)
    )

    if save_html:
        fig.write_html(f"{path}{system_name}/{pp}.html")
        print(f"saved File at {path}{system_name}/{pp}.html")
    if show:
        fig.show()
