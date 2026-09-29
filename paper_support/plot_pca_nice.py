import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import numpy as np
from pathlib import Path

def plot_pca_clusters_3cm(x_data, y_data, cluster_labels, 
                         axis_labels=None, legend_labels=None, 
                         title=None, output_dir=None, filename=None,
                         show_legend=False, colors=None, dpi=600):
    """
    Create a publication-ready 3cm x 3cm PCA cluster plot.
    
    Parameters:
    -----------
    x_data : array-like
        X coordinates (e.g., PC1 values)
    y_data : array-like
        Y coordinates (e.g., PC2 values)
    cluster_labels : array-like
        Cluster assignments for each point (1-indexed)
    axis_labels : tuple of str, optional
        (xlabel, ylabel). Default: ('PC1', 'PC2')
    legend_labels : list of str, optional
        Custom labels for each cluster. If None, uses 'Cluster N'
    title : str, optional
        Plot title. If None, shows 'k=N_clusters'
    output_dir : str or Path, optional
        Directory to save files. If None, doesn't save
    filename : str, optional
        Base filename (without extension). Default: 'pca_clusters_3cm'
    show_legend : bool, optional
        Whether to show legend. Default: False (recommended for 3cm)
    colors : list of str, optional
        Custom colors for clusters. If None, uses default palette
    dpi : int, optional
        Resolution for saved files. Default: 600
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Store original rcParams to restore later
    original_rcParams = rcParams.copy()
    
    # Set publication-ready matplotlib parameters for 3cm x 3cm panels
    rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 5,
        'axes.titlesize': 6,
        'axes.labelsize': 5,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'legend.fontsize': 5,
        'figure.titlesize': 6,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })
    
    try:
        # Convert inputs to numpy arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        cluster_labels = np.array(cluster_labels)
        
        # Get number of clusters
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        # Set default colors if not provided
        if colors is None:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Map cluster labels to colors
        cluster_to_color = {cluster: colors[(i) % len(colors)] 
                           for i, cluster in enumerate(unique_clusters)}
        point_colors = [cluster_to_color[label] for label in cluster_labels]
        
        # Create figure with exact 3cm x 3cm size (1.18 x 1.18 inches)
        fig, ax = plt.subplots(figsize=(1.18, 1.18), dpi=dpi)
        
        # Plot scatter points
        scatter_plot = ax.scatter(x_data, y_data, 
                                 c=point_colors, alpha=0.8, s=4, 
                                 linewidth=0.1, edgecolors='white')
        
        # Set axis labels
        if axis_labels is None:
            ax.set_xlabel('PC1', fontweight='normal')
            ax.set_ylabel('PC2', fontweight='normal')
        else:
            ax.set_xlabel(axis_labels[0], fontweight='normal')
            ax.set_ylabel(axis_labels[1], fontweight='normal')
        
        # Set title
        if title is None:
            ax.set_title(f'k={n_clusters}', fontweight='normal', pad=2)
        else:
            ax.set_title(title, fontweight='normal', pad=2)
        
        # Format ticks
        ax.tick_params(axis='both', which='major', length=1.5, width=0.4, 
                       direction='out', top=False, right=False)
        ax.tick_params(axis='both', which='minor', length=0.8, width=0.3, 
                       direction='out', top=False, right=False)
        
        # Set axis limits with minimal padding
        x_margin = (x_data.max() - x_data.min()) * 0.02
        y_margin = (y_data.max() - y_data.min()) * 0.02
        ax.set_xlim(x_data.min() - x_margin, x_data.max() + x_margin)
        ax.set_ylim(y_data.min() - y_margin, y_data.max() + y_margin)
        
        # Add legend if requested (not recommended for 3cm panels)
        if show_legend:
            legend_elements = []
            for cluster in unique_clusters:
                cluster_size = np.sum(cluster_labels == cluster)
                if legend_labels is not None:
                    label = legend_labels[int(cluster) - 1]
                else:
                    label = f'C{cluster} (n={cluster_size})'
                
                legend_elements.append(
                    patches.Circle((0, 0), 1, facecolor=cluster_to_color[cluster], 
                                 edgecolor='white', linewidth=0.1, label=label)
                )
            
            legend = ax.legend(handles=legend_elements, loc='upper right', 
                              bbox_to_anchor=(1.0, 1.0), frameon=True, 
                              fancybox=False, shadow=False, framealpha=0.9)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('gray')
            legend.get_frame().set_linewidth(0.3)
        
        # Style spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.4)
        ax.spines['bottom'].set_linewidth(0.4)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save files if output directory provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if filename is None:
                filename = 'pca_clusters_3cm'
            
            # Save PDF
            pdf_file = output_dir / f"{filename}.pdf"
            plt.savefig(pdf_file, dpi=dpi, bbox_inches='tight', 
                       format='pdf', facecolor='white', edgecolor='none')
            
            # Save PNG
            png_file = output_dir / f"{filename}.png"
            plt.savefig(png_file, dpi=dpi, bbox_inches='tight', 
                       format='png', facecolor='white', edgecolor='none')
            
            print(f"✓ 3cm x 3cm panel saved ({dpi} DPI):")
            print(f"  • PDF: {pdf_file}")
            print(f"  • PNG: {png_file}")
        
        return fig, ax
        
    finally:
        # Restore original rcParams
        rcParams.update(original_rcParams)


def plot_pca_with_variance_labels(x_data, y_data, cluster_labels, 
                                 pc1_variance, pc2_variance,
                                 title=None, output_dir=None, filename=None,
                                 show_legend=False, colors=None, dpi=600):
    """
    Convenience function for PCA plots with variance-explained labels.
    
    Parameters:
    -----------
    x_data : array-like
        PC1 coordinates
    y_data : array-like  
        PC2 coordinates
    cluster_labels : array-like
        Cluster assignments for each point
    pc1_variance : float
        Variance explained by PC1 (as decimal, e.g., 0.45)
    pc2_variance : float
        Variance explained by PC2 (as decimal, e.g., 0.23)
    colors : array-like or None
        Custom colors for clusters. If None, uses Set2 colormap.
    Other parameters same as plot_pca_clusters_3cm()
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Use Set2 colormap by default if no colors provided
    if colors is None:
        from matplotlib.cm import Set2
        import numpy as np
        n_clusters = len(np.unique(cluster_labels))
        colors = Set2(np.linspace(0, 1, n_clusters))
    
    axis_labels = (f'PC1 ({pc1_variance:.0%})', f'PC2 ({pc2_variance:.0%})')
    
    return plot_pca_clusters_3cm(
        x_data, y_data, cluster_labels,
        axis_labels=axis_labels,
        title=title,
        output_dir=output_dir,
        filename=filename,
        show_legend=show_legend,
        colors=colors,
        dpi=dpi
    )

# Example usage:
if __name__ == "__main__":
    # Example with your original data structure
    # Assuming you have: pca_coordinates, cluster_assignments, PCA_RESULTS
    
    # Basic usage
    fig, ax = plot_pca_clusters_3cm(
        x_data=pca_coordinates[:, 0],
        y_data=pca_coordinates[:, 1], 
        cluster_labels=cluster_assignments,
        output_dir=OUTPUT_DIR,
        filename="hierarchical_clustering_3cm"
    )
    plt.show()
    
    # With variance labels
    fig, ax = plot_pca_with_variance_labels(
        x_data=pca_coordinates[:, 0],
        y_data=pca_coordinates[:, 1],
        cluster_labels=cluster_assignments,
        pc1_variance=PCA_RESULTS["explained_variance"][0],
        pc2_variance=PCA_RESULTS["explained_variance"][1],
        output_dir=OUTPUT_DIR,
        filename="hierarchical_clustering_with_variance"
    )
    plt.show()
    
    # Custom styling
    custom_colors = ['#E31A1C', '#1F78B4', '#33A02C', '#FF7F00']
    custom_legend = ['Group A', 'Group B', 'Group C', 'Group D']
    
    fig, ax = plot_pca_clusters_3cm(
        x_data=pca_coordinates[:, 0],
        y_data=pca_coordinates[:, 1],
        cluster_labels=cluster_assignments,
        axis_labels=('First Component', 'Second Component'),
        legend_labels=custom_legend,
        title='Custom Analysis',
        colors=custom_colors,
        show_legend=True,  # Only if you have space
        output_dir=OUTPUT_DIR,
        filename="custom_pca_plot"
    )
    plt.show()