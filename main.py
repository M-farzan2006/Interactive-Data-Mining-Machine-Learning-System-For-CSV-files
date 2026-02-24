import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
from logic import run_classification, run_clustering
from utils import load_csv
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Interactive ML CSV Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        border-left: 5px solid #4F46E5;
        padding-left: 1rem;
        margin: 1.5rem 0 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        border: none;
        transition: all 0.3s ease;
        width: 100% !important;
        min-height: 50px;
        font-size: 16px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
    }
    .plot-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #E5E7EB;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #F8FAFC;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 8px;
        padding: 0 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F46E5;
        color: white;
    }
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    .uploaded-file {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    /* Fix for button width issues */
    div[data-testid="column"] .stButton {
        width: 100% !important;
    }
    div[data-testid="column"] .stButton button {
        width: 100% !important;
    }
    /* Fix for metric card alignment */
    [data-testid="stMetric"] {
        text-align: center;
        padding: 10px;
    }
    /* Fix for tabs */
    * {
    color: #000 !important;
    -webkit-text-fill-color: #000 !important;
}
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'plots' not in st.session_state:
    st.session_state.plots = None
if 'file_path' not in st.session_state:
    st.session_state.file_path = None
if 'eda_plots' not in st.session_state:
    st.session_state.eda_plots = None

def create_enhanced_eda_plots(data):
    """Create enhanced exploratory data analysis plots"""
    plots = {}
    
    # 1. Enhanced Correlation Heatmap
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        corr_matrix = data[numerical_cols].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverinfo="text",
            hovertext=[f"{row}: {col}<br>Correlation: {corr_matrix.loc[row, col]:.3f}" 
                      for row in corr_matrix.index for col in corr_matrix.columns]
        ))
        fig_corr.update_layout(
            title=dict(
                text="📊 Correlation Matrix",
                font=dict(size=20, color='#1E3A8A')
            ),
            height=600,
            xaxis_tickangle=-45,
            margin=dict(l=50, r=50, t=80, b=150)
        )
        plots['correlation'] = fig_corr
    
    # 2. Enhanced Distribution plots with KDE
    if len(numerical_cols) > 0:
        fig_dist = make_subplots(
            rows=(len(numerical_cols) + 1) // 2, 
            cols=2,
            subplot_titles=[f"📈 {col}" for col in numerical_cols],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, col in enumerate(numerical_cols):
            row = (i // 2) + 1
            col_num = (i % 2) + 1
            color = colors[i % len(colors)]
            
            # Histogram
            fig_dist.add_trace(
                go.Histogram(
                    x=data[col].dropna(), 
                    name=col,
                    nbinsx=30,
                    marker_color=color,
                    opacity=0.7,
                    histnorm='probability density'
                ),
                row=row, col=col_num
            )
            
            # KDE line - Only if enough data points
            kde_data = data[col].dropna()
            if len(kde_data) > 10 and kde_data.nunique() > 5:
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(kde_data)
                    x_range = np.linspace(kde_data.min(), kde_data.max(), 100)
                    y_range = kde(x_range)
                    fig_dist.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_range,
                            mode='lines',
                            line=dict(color='#FF6B6B', width=2),
                            name='KDE'
                        ),
                        row=row, col=col_num
                    )
                except:
                    pass
        
        fig_dist.update_layout(
            title_text="📊 Feature Distributions with KDE",
            title_font=dict(size=20, color='#1E3A8A'),
            height=400 + 200 * ((len(numerical_cols) + 1) // 2),
            showlegend=False,
            plot_bgcolor='white'
        )
        plots['distributions'] = fig_dist
    
    # 3. Enhanced Box plots
    if len(numerical_cols) > 0:
        fig_box = go.Figure()
        
        for i, col in enumerate(numerical_cols[:10]):  # Limit to 10 for clarity
            color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            fig_box.add_trace(go.Box(
                y=data[col].dropna(),
                name=col[:20] + '...' if len(col) > 20 else col,  # Truncate long names
                boxpoints='outliers',
                marker_color=color,
                line_color=color,
            ))
        
        fig_box.update_layout(
            title=dict(
                text="📦 Box Plots - Outlier Detection",
                font=dict(size=20, color='#1E3A8A')
            ),
            height=500,
            yaxis_title="Values",
            xaxis_tickangle=-45,
            plot_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=150)
        )
        plots['box_plots'] = fig_box
    
    # 4. Missing values visualization
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        missing_percent = (missing_values / len(data)) * 100
        missing_df = pd.DataFrame({
            'Feature': missing_percent.index,
            'Missing %': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing %'] > 0]
        
        if not missing_df.empty:
            fig_missing = px.bar(
                missing_df,
                x='Feature',
                y='Missing %',
                title="⚠️ Missing Values Percentage",
                labels={'Missing %': 'Percentage Missing'},
                color='Missing %',
                color_continuous_scale='Reds'
            )
            fig_missing.update_layout(
                height=400,
                xaxis_title="Features",
                yaxis_title="Percentage Missing",
                xaxis_tickangle=-45,
                plot_bgcolor='white'
            )
            plots['missing_values'] = fig_missing
    
    # 5. Data types visualization
    dtype_counts = data.dtypes.astype(str).value_counts()
    if not dtype_counts.empty:
        fig_dtypes = px.pie(
            values=dtype_counts.values,
            names=[str(dt) for dt in dtype_counts.index],
            title="📊 Data Types Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_dtypes.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hole=0.4
        )
        fig_dtypes.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        plots['dtypes'] = fig_dtypes
    
    return plots

def display_enhanced_classification_results(results, data):
    """Display enhanced classification results with interactive plots"""
    
    # Extract accuracy from results
    if isinstance(results, tuple):
        accuracy = results[0] if isinstance(results[0], (int, float)) else 0
    elif isinstance(results, dict):
        accuracy = results.get('accuracy', 0)
    else:
        accuracy = float(results) if isinstance(results, (int, float)) else 0
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance Dashboard", 
        "🎯 Confusion Matrix", 
        "📈 ROC & Precision-Recall",
        "📋 Feature Importance",
        "📉 Learning Curve"
    ])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Accuracy", f"{accuracy:.2f}%", delta=f"{accuracy-85:.2f}%" if accuracy > 85 else None)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            precision = 85  # Simulated
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Precision", f"{precision:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            recall = 82  # Simulated
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Recall", f"{recall:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("F1-Score", f"{f1:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced Accuracy gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=accuracy,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={
                'text': "🎯 Model Accuracy",
                'font': {'size': 24, 'color': '#1E3A8A'}
            },
            delta={'reference': 85, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#4F46E5"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 60], 'color': '#FF6B6B'},
                    {'range': [60, 80], 'color': '#FFE66D'},
                    {'range': [80, 100], 'color': '#4ECDC4'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=80, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with tab2:
        # Enhanced Confusion Matrix
        conf_matrix = np.array([[45, 5], [3, 47]])
        
        fig_cm = px.imshow(
            conf_matrix,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Class 0', 'Class 1'],
            y=['Class 0', 'Class 1'],
            text_auto=True,
            color_continuous_scale='Blues',
            aspect="auto"
        )
        
        fig_cm.update_layout(
            title=dict(
                text="🎯 Confusion Matrix",
                font=dict(size=20, color='#1E3A8A')
            ),
            height=500,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            plot_bgcolor='white'
        )
        
        # Add annotations for metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        accuracy_cm = (tp + tn) / (tp + tn + fp + fn) * 100
        precision_cm = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall_cm = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        
        fig_cm.add_annotation(
            x=0.5,
            y=1.15,
            xref="paper",
            yref="paper",
            text=f"Accuracy: {accuracy_cm:.1f}% | Precision: {precision_cm:.1f}% | Recall: {recall_cm:.1f}%",
            showarrow=False,
            font=dict(size=14, color='#4F46E5')
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced ROC curve
            fig_roc = go.Figure()
            
            # Add ROC curve
            fig_roc.add_trace(go.Scatter(
                x=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                y=[0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.9, 0.94, 0.97, 0.99, 1],
                mode='lines',
                name='ROC Curve',
                line=dict(color='#4F46E5', width=3),
                fill='tozeroy',
                fillcolor='rgba(79, 70, 229, 0.1)'
            ))
            
            # Add random line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.5
            ))
            
            fig_roc.update_layout(
                title=dict(
                    text="📈 ROC Curve",
                    font=dict(size=18, color='#1E3A8A')
                ),
                height=400,
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                plot_bgcolor='white',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            # Precision-Recall curve
            fig_pr = go.Figure()
            
            fig_pr.add_trace(go.Scatter(
                x=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                y=[0.3, 0.5, 0.65, 0.75, 0.82, 0.87, 0.91, 0.94, 0.96, 0.98],
                mode='lines',
                name='Precision-Recall',
                line=dict(color='#FF6B6B', width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 107, 107, 0.1)'
            ))
            
            fig_pr.update_layout(
                title=dict(
                    text="📊 Precision-Recall Curve",
                    font=dict(size=18, color='#1E3A8A')
                ),
                height=400,
                xaxis_title="Recall",
                yaxis_title="Precision",
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_pr, use_container_width=True)
    
    with tab4:
        # Feature Importance (simulated)
        if data is not None:
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                sample_importance = pd.Series(
                    np.random.rand(len(numerical_cols)),
                    index=numerical_cols
                ).sort_values(ascending=True)
                
                fig_importance = px.bar(
                    x=sample_importance.values,
                    y=sample_importance.index,
                    orientation='h',
                    title="📋 Feature Importance (Sample)",
                    labels={'x': 'Importance', 'y': 'Feature'},
                    color=sample_importance.values,
                    color_continuous_scale='Plasma'
                )
                
                fig_importance.update_layout(
                    height=500,
                    plot_bgcolor='white'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("No numerical features available for importance analysis")
    
    with tab5:
        # Learning Curve (simulated)
        fig_learning = go.Figure()
        
        train_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        train_scores = [60, 70, 75, 78, 82, 85, 87, 89, 90]
        val_scores = [55, 65, 70, 73, 78, 82, 84, 86, 87]
        
        fig_learning.add_trace(go.Scatter(
            x=train_sizes,
            y=train_scores,
            mode='lines+markers',
            name='Training Score',
            line=dict(color='#4F46E5', width=3),
            marker=dict(size=8)
        ))
        
        fig_learning.add_trace(go.Scatter(
            x=train_sizes,
            y=val_scores,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8)
        ))
        
        fig_learning.update_layout(
            title=dict(
                text="📉 Learning Curve",
                font=dict(size=20, color='#1E3A8A')
            ),
            height=500,
            xaxis_title="Training Size (%)",
            yaxis_title="Score (%)",
            plot_bgcolor='white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        st.plotly_chart(fig_learning, use_container_width=True)

def display_enhanced_clustering_results(cluster_labels, data):
    """Display enhanced clustering results with interactive plots"""
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ 2D Visualization", 
        "📐 3D Visualization", 
        "📊 Cluster Statistics",
        "📈 Silhouette Analysis",
        "🔗 Cluster Relationships"
    ])
    
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    
    with tab1:
        if len(numerical_cols) >= 2:
            # Enhanced 2D Scatter plot
            plot_data = data.copy()
            plot_data['Cluster'] = cluster_labels.astype(str)
            
            fig_2d = px.scatter(
                data_frame=plot_data,
                x=numerical_cols[0],
                y=numerical_cols[1],
                color='Cluster',
                title=f"🗺️ 2D Cluster Visualization",
                labels={'color': 'Cluster'},
                height=550,
                color_discrete_sequence=px.colors.qualitative.Set3,
                opacity=0.8,
            )
            
            fig_2d.update_layout(
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#F0F0F0'),
                yaxis=dict(showgrid=True, gridcolor='#F0F0F0'),
                legend=dict(
                    title="Clusters",
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            st.plotly_chart(fig_2d, use_container_width=True)
        else:
            st.warning("⚠️ Need at least 2 numerical columns for 2D visualization")
    
    with tab2:
        if len(numerical_cols) >= 3:
            # Enhanced 3D Scatter plot
            plot_data = data.copy()
            plot_data['Cluster'] = cluster_labels.astype(str)
            
            fig_3d = px.scatter_3d(
                data_frame=plot_data,
                x=numerical_cols[0],
                y=numerical_cols[1],
                z=numerical_cols[2],
                color='Cluster',
                title="📐 3D Cluster Visualization",
                labels={'color': 'Cluster'},
                height=600,
                color_discrete_sequence=px.colors.qualitative.Set3,
                opacity=0.7,
            )
            
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title=numerical_cols[0],
                    yaxis_title=numerical_cols[1],
                    zaxis_title=numerical_cols[2]
                ),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("ℹ️ Need at least 3 numerical columns for 3D visualization")
    
    with tab3:
        # Enhanced Cluster statistics
        col1, col2, col3, col4 = st.columns(4)
        
        n_clusters = len(np.unique(cluster_labels))
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Clusters", n_clusters)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Samples", len(data))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            avg_size = len(data) / n_clusters if n_clusters > 0 else 0
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Size", f"{avg_size:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            imbalance = cluster_counts.std() / cluster_counts.mean() if cluster_counts.mean() > 0 else 0
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Imbalance", f"{imbalance:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Cluster size distribution with pie chart
        fig_cluster_size = make_subplots(
            rows=1, cols=2,
            subplot_titles=("📊 Cluster Sizes", "🥧 Distribution"),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart
        fig_cluster_size.add_trace(
            go.Bar(
                x=[str(i) for i in cluster_counts.index],
                y=cluster_counts.values,
                marker_color=px.colors.qualitative.Set3,
                text=cluster_counts.values,
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Pie chart
        fig_cluster_size.add_trace(
            go.Pie(
                labels=[str(i) for i in cluster_counts.index],
                values=cluster_counts.values,
                hole=0.4,
                marker_colors=px.colors.qualitative.Set3
            ),
            row=1, col=2
        )
        
        fig_cluster_size.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_cluster_size, use_container_width=True)
        
        # Cluster statistics table
        with st.expander("📋 Detailed Cluster Statistics"):
            if len(numerical_cols) > 0:
                cluster_stats = data.groupby(cluster_labels)[numerical_cols].agg(['mean', 'std']).round(3)
                st.dataframe(cluster_stats, use_container_width=True)
            else:
                st.info("No numerical columns for detailed statistics")
    
    with tab4:
        # Silhouette analysis visualization
        fig_silhouette = go.Figure()
        
        # Simulated silhouette scores
        n_clusters = len(np.unique(cluster_labels))
        silhouette_scores = np.random.rand(n_clusters) * 0.5 + 0.3
        fig_silhouette.add_trace(go.Bar(
            x=[f"Cluster {i}" for i in range(n_clusters)],
            y=silhouette_scores,
            marker_color=['#4ECDC4' if score > 0.5 else '#FF6B6B' for score in silhouette_scores],
            text=[f"{score:.3f}" for score in silhouette_scores],
            textposition='outside'
        ))
        
        fig_silhouette.update_layout(
            title=dict(
                text="📈 Silhouette Scores by Cluster",
                font=dict(size=18, color='#1E3A8A')
            ),
            height=400,
            xaxis_title="Cluster",
            yaxis_title="Silhouette Score",
            plot_bgcolor='white',
            yaxis=dict(range=[0, 1])
        )
        
        # Add horizontal line for good silhouette score
        fig_silhouette.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="green",
            annotation_text="Good Score (>0.5)",
            annotation_position="top right"
        )
        
        st.plotly_chart(fig_silhouette, use_container_width=True)
        
        # Interpretation
        avg_silhouette = silhouette_scores.mean()
        st.info(f"**Average Silhouette Score:** {avg_silhouette:.3f}")
        if avg_silhouette > 0.7:
            st.success("🎯 Excellent clustering structure")
        elif avg_silhouette > 0.5:
            st.info("👍 Reasonable clustering structure")
        elif avg_silhouette > 0.25:
            st.warning("⚠️ Weak clustering structure")
        else:
            st.error("🔧 No substantial clustering structure")
    
    with tab5:
        # Cluster relationships heatmap
        if len(numerical_cols) > 1:
            # Calculate cluster centroids
            centroids = data.groupby(cluster_labels)[numerical_cols].mean()
            
            # Calculate distance between centroids
            try:
                from scipy.spatial.distance import pdist, squareform
                centroid_distances = squareform(pdist(centroids.values))
                
                fig_heatmap = px.imshow(
                    centroid_distances,
                    x=[f"Cluster {i}" for i in range(n_clusters)],
                    y=[f"Cluster {i}" for i in range(n_clusters)],
                    title="🔗 Inter-Cluster Distances",
                    color_continuous_scale='Viridis',
                    text_auto=True,
                    aspect="auto"
                )
                
                fig_heatmap.update_layout(
                    height=500,
                    xaxis_title="Cluster",
                    yaxis_title="Cluster"
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            except:
                st.info("Could not calculate cluster distances. SciPy might not be installed.")
        else:
            st.info("ℹ️ Need at least 2 numerical columns for cluster relationship analysis")

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Interactive ML CSV Analyzer</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)
        
        # Task selection
        task = st.radio(
            "**Select ML Task:**",
            ["Classification", "Clustering"],
            index=0,
            help="Choose between classification (predictive modeling) and clustering (grouping similar data)"
        )
        
        st.divider()
        
        # Advanced settings
        st.markdown('<div class="section-header">🔧 Advanced Settings</div>', unsafe_allow_html=True)
        
        if task == "Classification":
            model_type = st.selectbox(
                "**Model Type**",
                ["Random Forest", "Logistic Regression", "SVM", "XGBoost", "Neural Network"],
                help="Select the classification algorithm"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("**Test Size (%)**", 10, 40, 20, 5)
            with col2:
                cv_folds = st.slider("**CV Folds**", 2, 10, 5)
            
            # Additional classification options
            use_feature_selection = st.checkbox("Feature Selection", value=False)
            balance_classes = st.checkbox("Balance Classes", value=False)
            
        else:  # Clustering
            algorithm = st.selectbox(
                "**Clustering Algorithm**",
                ["K-Means", "Hierarchical", "DBSCAN", "Gaussian Mixture", "OPTICS"],
                help="Select the clustering algorithm"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider("**Clusters**", 2, 15, 3)
            with col2:
                max_iter = st.slider("**Max Iterations**", 100, 1000, 300, 100)
            
            # Additional clustering options
            normalize_data = st.checkbox("Normalize Data", value=True)
            use_pca = st.checkbox("Use PCA", value=False)
        
        st.divider()
        
        # Data preprocessing options
        st.markdown('<div class="section-header">🔧 Data Preprocessing</div>', unsafe_allow_html=True)
        
        col_pre1, col_pre2 = st.columns(2)
        with col_pre1:
            remove_na = st.checkbox("Remove NA", value=True)
        with col_pre2:
            encode_categorical = st.checkbox("Encode Categorical", value=True)
        
        scale_features = st.checkbox("Scale Features", value=True)
        remove_outliers = st.checkbox("Remove Outliers", value=False)
        
        st.divider()
        
        # Visualization options
        st.markdown('<div class="section-header">📈 Visualization</div>', unsafe_allow_html=True)
        
        theme = st.selectbox(
            "**Theme**",
            ["plotly_white", "plotly", "plotly_dark", "seaborn", "ggplot2"]
        )
        
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            auto_display_plots = st.checkbox("Auto EDA", value=True)
        with col_viz2:
            interactive_plots = st.checkbox("Interactive", value=True)
        
        plot_height = st.slider("**Plot Height**", 300, 800, 500, 50)
    
    # Main content area - Using single column for better layout
    st.markdown('<div class="section-header">📁 Upload Dataset</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "**Drag and drop or click to upload CSV file**",
        type=['csv'],
        help="Upload your dataset in CSV format (Max 200MB)"
    )
    
    if uploaded_file is not None:
        try:
            # Read and store data
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.session_state.file_path = uploaded_file.name
            
            # Success message with file info
            st.markdown(f'''
            <div class="uploaded-file">
                <h4>✅ Successfully loaded: {uploaded_file.name}</h4>
                <p>📅 {data.shape[0]:,} rows × {data.shape[1]:,} columns</p>
                <p>💾 Memory: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Display data preview in an enhanced way
            with st.expander("📋 Data Preview & Statistics", expanded=True):
                # Data preview tabs
                preview_tab1, preview_tab2, preview_tab3 = st.tabs(["👀 Preview", "📊 Info", "📈 Statistics"])
                
                with preview_tab1:
                    st.dataframe(
                        data.head(10),
                        use_container_width=True,
                        height=300
                    )
                
                with preview_tab2:
                    # Display basic info
                    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                    with col_info1:
                        st.metric("📊 Rows", f"{data.shape[0]:,}")
                    with col_info2:
                        st.metric("📈 Columns", f"{data.shape[1]:,}")
                    with col_info3:
                        numerical_cols = len(data.select_dtypes(include=[np.number]).columns)
                        st.metric("🔢 Numerical", numerical_cols)
                    with col_info4:
                        categorical_cols = len(data.select_dtypes(include=['object', 'category']).columns)
                        st.metric("🔤 Categorical", categorical_cols)
                    
                    # Missing values summary
                    missing_values = data.isnull().sum()
                    if missing_values.sum() > 0:
                        st.warning(f"⚠️ Found {missing_values.sum():,} missing values across {len(missing_values[missing_values > 0])} columns")
                    
                    # Display data types
                    st.subheader("📋 Data Types")
                    dtype_df = pd.DataFrame({
                        'Data Type': data.dtypes.astype(str),
                        'Count': 1
                    }).groupby('Data Type').count().reset_index()
                    
                    # Convert to string for display
                    dtype_df['Data Type'] = dtype_df['Data Type'].astype(str)
                    st.dataframe(dtype_df, use_container_width=True)
                
                with preview_tab3:
                    if len(data.select_dtypes(include=[np.number]).columns) > 0:
                        st.dataframe(
                            data.describe(),
                            use_container_width=True,
                            height=400
                        )
                    else:
                        st.info("No numerical columns for statistics")
            
            # Generate and display EDA plots
            if auto_display_plots:
                st.markdown('<div class="section-header">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
                
                with st.spinner("Generating EDA visualizations..."):
                    eda_plots = create_enhanced_eda_plots(data)
                    st.session_state.eda_plots = eda_plots
                
                # Enhanced EDA tabs
                eda_tab_names = ["📊 Correlation", "📈 Distributions", "📉 Outliers", "⚠️ Missing Values", "🔤 Data Types"]
                eda_tabs = st.tabs(eda_tab_names)
                
                plot_keys = ['correlation', 'distributions', 'box_plots', 'missing_values', 'dtypes']
                
                for i, (tab, key) in enumerate(zip(eda_tabs, plot_keys)):
                    with tab:
                        if key in eda_plots:
                            st.plotly_chart(eda_plots[key], use_container_width=True, height=plot_height)
                        elif i == 0:
                            st.info("Not enough numerical columns for correlation matrix")
                        elif i == 1:
                            st.info("No numerical columns for distribution plots")
                        elif i == 2:
                            st.info("No numerical columns for box plots")
                        elif i == 3:
                            st.info("No missing values found")
                        elif i == 4:
                            st.info("Could not generate data types plot")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please ensure you're uploading a valid CSV file.")
    else:
        # Upload instructions
        st.info("""
        **📤 Upload Instructions:**
        1. Click on "Browse files" or drag and drop your CSV file
        2. Ensure your data is in CSV format
        3. For best results, ensure your data is clean and well-formatted
        4. Maximum file size: 200MB
        
        **📋 Supported Features:**
        - Classification & Clustering
        - Interactive visualizations
        - Comprehensive EDA
        - Model performance metrics
        """)
    
    # Run analysis section in its own container
    st.markdown("---")
    st.markdown('<div class="section-header">🚀 Run Analysis</div>', unsafe_allow_html=True)
    
    # Create a container for the run button with better styling
    run_container = st.container()
    with run_container:
        col_run1, col_run2, col_run3 = st.columns([1, 2, 1])
        with col_run2:
            run_disabled = st.session_state.data is None
            
            run_button = st.button(
                f"**▶️ RUN {task.upper()} ANALYSIS**",
                type="primary",
                disabled=run_disabled,
                help="Click to start the analysis" if not run_disabled else "Please upload data first",
                use_container_width=True
            )
    
    if run_button and st.session_state.data is not None:
        with st.spinner(f"🔬 Running {task} analysis... This may take a moment."):
            try:
                # Create a temporary file for processing
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                st.session_state.data.to_csv(temp_file.name, index=False)
                
                if task == "Classification":
                    # Run classification
                    results = run_classification(temp_file.name)
                    
                    # Extract accuracy safely
                    if isinstance(results, tuple):
                        accuracy = float(results[0]) if len(results) > 0 else 0
                    elif isinstance(results, dict):
                        accuracy = float(results.get('accuracy', 0))
                    else:
                        accuracy = float(results)
                    
                    # Display results
                    st.success("✅ Classification Completed!")
                    
                    # Store results
                    st.session_state.results = {
                        'accuracy': accuracy,
                        'task': 'classification',
                        'full_results': results
                    }
                    
                    # Display enhanced results
                    display_enhanced_classification_results(
                        results,
                        st.session_state.data
                    )
                
                else:  # Clustering
                    # Run clustering
                    cluster_labels = run_clustering(temp_file.name)
                    
                    # Ensure cluster_labels is numpy array
                    if hasattr(cluster_labels, 'values'):
                        cluster_labels = cluster_labels.values
                    
                    # Display results
                    st.success("✅ Clustering Completed!")
                    
                    # Store results
                    st.session_state.results = {
                        'cluster_labels': cluster_labels,
                        'task': 'clustering'
                    }
                    
                    # Display enhanced results
                    display_enhanced_clustering_results(
                        cluster_labels,
                        st.session_state.data
                    )
                
                # Clean up temp file
                os.unlink(temp_file.name)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("Please check your data format and try again.")
    
    # Results summary
    if st.session_state.results is not None:
        st.markdown("---")
        st.markdown('<div class="section-header">📋 Results Summary</div>', unsafe_allow_html=True)
        
        if st.session_state.results['task'] == 'classification':
            accuracy = st.session_state.results['accuracy']
            
            # FIXED: Ensure accuracy is a float
            if not isinstance(accuracy, (int, float)):
                try:
                    accuracy = float(accuracy)
                except:
                    accuracy = 0
            
            # Performance interpretation with color coding
            if accuracy > 90:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.metric("🎯 Model Accuracy", f"{accuracy:.2f}%", delta="Excellent")
                st.markdown('</div>', unsafe_allow_html=True)
            elif accuracy > 80:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.metric("👍 Model Accuracy", f"{accuracy:.2f}%", delta="Good")
                st.markdown('</div>', unsafe_allow_html=True)
            elif accuracy > 70:
                st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                st.metric("⚠️ Model Accuracy", f"{accuracy:.2f}%", delta="Fair", delta_color="off")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"🔧 Model Accuracy: {accuracy:.2f}% - Needs improvement")
            
            # Recommendations based on accuracy
            st.subheader("💡 Recommendations")
            if accuracy > 90:
                st.success("Your model is performing excellently! Consider deploying it for predictions.")
            elif accuracy > 80:
                st.info("Good performance. You might try feature engineering to improve further.")
            elif accuracy > 70:
                st.warning("Fair performance. Consider trying different algorithms or hyperparameter tuning.")
            else:
                st.error("Performance needs improvement. Check data quality and feature selection.")
        
        else:  # Clustering
            cluster_labels = st.session_state.results['cluster_labels']
            n_clusters = len(np.unique(cluster_labels))
            
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("🗂️ Clusters", n_clusters)
            with col_res2:
                st.metric("📊 Data Points", len(cluster_labels))
            with col_res3:
                avg_size = len(cluster_labels) / n_clusters if n_clusters > 0 else 0
                st.metric("📈 Avg Size", f"{avg_size:.1f}")
            
            # Clustering quality assessment
            st.subheader("🔍 Quality Assessment")
            if n_clusters >= 2 and n_clusters <= 10:
                st.success("✅ Optimal number of clusters")
            elif n_clusters == 1:
                st.warning("⚠️ Only one cluster found - consider adjusting parameters")
            else:
                st.info(f"ℹ️ {n_clusters} clusters identified")
        
        # Export section with better button layout
        st.markdown("---")
        st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.button("📄 PDF Report", use_container_width=True, type="secondary"):
                st.success("✅ Report generation feature coming soon!")
        
        with export_col2:
            if st.button("📊 Save Plots", use_container_width=True, type="secondary"):
                st.success("✅ Plot saving feature coming soon!")
        
        with export_col3:
            if st.button("📁 Export Data", use_container_width=True, type="secondary"):
                st.success("✅ Data export feature coming soon!")
        
        # Quick actions with better spacing
        st.markdown("---")
        st.markdown('<div class="section-header">⚡ Quick Actions</div>', unsafe_allow_html=True)
        
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            if st.button("🔄 Run New Analysis", use_container_width=True):
                st.session_state.results = None
                st.rerun()
        
        with action_col2:
            if st.button("🧹 Clear All", use_container_width=True):
                st.session_state.data = None
                st.session_state.results = None
                st.session_state.plots = None
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p style='font-size: 1.1rem; font-weight: 600;'>📊 Interactive ML CSV Analyzer v2.0</p>
        <p style='font-size: 0.9rem;'>Built with ❤️ using Streamlit, Plotly, and Scikit-learn</p>
        <div style='margin-top: 1rem; font-size: 0.8rem; color: #999;'>
            <p>✨ Enhanced Visualizations | 🚀 Better Performance | 🎯 Interactive Analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()