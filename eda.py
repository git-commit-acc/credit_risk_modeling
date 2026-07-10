# evaluation/eda.py
"""
Exploratory Data Analysis for Freddie Mac SFLLD data.
Provides comprehensive visualizations to understand data characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, avg, stddev, min as spark_min, max as spark_max

logger = logging.getLogger(__name__)


class SFLLDEDA:
    """
    Exploratory Data Analysis for Freddie Mac SFLLD data.
    Creates visualizations to understand data characteristics.
    """
    
    def __init__(self, save_dir: str = None):
        """
        Initialize EDA visualizer.
        
        Args:
            save_dir: Directory to save plots. If None, display plots.
        """
        self.save_dir = save_dir
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    # ========================================================================
    # DATA QUALITY VISUALIZATIONS
    # ========================================================================
    
    def plot_missing_values(self, df: pd.DataFrame, title: str = "Missing Values Analysis") -> plt.Figure:
        """
        Plot missing value percentages for each column.
        
        Args:
            df: Pandas DataFrame with data
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Calculate missing percentages
        missing_pct = (df.isnull().sum() / len(df)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
        
        if len(missing_pct) == 0:
            ax.text(0.5, 0.5, "No Missing Values Found", 
                   ha='center', va='center', fontsize=16)
            ax.set_title(title)
            return fig
        
        # Create bar plot
        bars = ax.barh(missing_pct.index, missing_pct.values, color='coral')
        
        # Add value labels
        for i, (idx, val) in enumerate(missing_pct.items()):
            ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)
        
        ax.set_xlabel('Missing Percentage (%)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(0, min(100, missing_pct.max() * 1.15))
        
        plt.tight_layout()
        return fig
    
    def plot_data_types(self, df: pd.DataFrame) -> plt.Figure:
        """
        Plot data type distribution.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Data type distribution
        dtype_counts = df.dtypes.astype(str).value_counts()
        colors = sns.color_palette("Set2", len(dtype_counts))
        
        ax1.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title('Data Type Distribution', fontsize=14)
        
        # Number of columns by type
        ax2.bar(dtype_counts.index, dtype_counts.values, color=colors)
        ax2.set_xlabel('Data Type', fontsize=12)
        ax2.set_ylabel('Number of Columns', fontsize=12)
        ax2.set_title('Columns by Data Type', fontsize=14)
        
        # Add value labels
        for i, (idx, val) in enumerate(dtype_counts.items()):
            ax2.text(i, val + 0.5, str(val), ha='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    # ========================================================================
    # UNIVARIATE ANALYSIS - ORIGINATION
    # ========================================================================
    
    def plot_origination_distributions(self, df: pd.DataFrame) -> plt.Figure:
        """
        Plot distributions of key origination variables.
        
        Args:
            df: Pandas DataFrame with origination data
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Credit Score Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_histogram(df, 'CREDIT_SCORE', ax1, 
                            title='Credit Score Distribution',
                            xlabel='Credit Score')
        
        # 2. Original LTV Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_histogram(df, 'ORIGINAL_LTV', ax2,
                            title='Original LTV Distribution',
                            xlabel='LTV (%)')
        
        # 3. Original DTI Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_histogram(df, 'ORIGINAL_DTI', ax3,
                            title='Original DTI Distribution',
                            xlabel='DTI (%)')
        
        # 4. Original Interest Rate Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_histogram(df, 'ORIGINAL_INTEREST_RATE', ax4,
                            title='Original Interest Rate Distribution',
                            xlabel='Interest Rate (%)')
        
        # 5. Original UPB Distribution (log scale)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_histogram(df, 'ORIGINAL_UPB', ax5,
                            title='Original UPB Distribution',
                            xlabel='UPB ($)', log_scale=True)
        
        # 6. Original Loan Term Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_histogram(df, 'ORIGINAL_LOAN_TERM', ax6,
                            title='Original Loan Term Distribution',
                            xlabel='Loan Term (months)')
        
        # 7. First Time Homebuyer Flag
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_categorical(df, 'FIRST_TIME_HOMEBUYER_FLAG', ax7,
                              title='First Time Homebuyer Flag')
        
        # 8. Property Type
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_categorical(df, 'PROPERTY_TYPE', ax8,
                              title='Property Type')
        
        # 9. Occupancy Status
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_categorical(df, 'OCCUPANCY_STATUS', ax9,
                              title='Occupancy Status')
        
        # 10. Loan Purpose
        ax10 = fig.add_subplot(gs[3, 0])
        self._plot_categorical(df, 'LOAN_PURPOSE', ax10,
                              title='Loan Purpose')
        
        # 11. Channel
        ax11 = fig.add_subplot(gs[3, 1])
        self._plot_categorical(df, 'CHANNEL', ax11,
                              title='Channel')
        
        # 12. Super Conforming Flag
        ax12 = fig.add_subplot(gs[3, 2])
        self._plot_categorical(df, 'SUPER_CONFORMING_FLAG', ax12,
                              title='Super Conforming Flag')
        
        plt.suptitle('Origination Data Distributions', fontsize=16, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_origination_by_vintage(self, df: pd.DataFrame) -> plt.Figure:
        """
        Plot origination trends by vintage year.
        
        Args:
            df: Pandas DataFrame with origination data
            
        Returns:
            Matplotlib figure
        """
        # Extract vintage year from loan sequence number
        df['vintage_year'] = df['LOAN_SEQUENCE_NUMBER'].str[1:3].astype(int) + 2000
        
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Number of Loans by Vintage
        ax1 = fig.add_subplot(gs[0, 0])
        vintage_counts = df['vintage_year'].value_counts().sort_index()
        ax1.bar(vintage_counts.index, vintage_counts.values, color='steelblue')
        ax1.set_xlabel('Vintage Year', fontsize=12)
        ax1.set_ylabel('Number of Loans', fontsize=12)
        ax1.set_title('Loans by Vintage Year', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Average Credit Score by Vintage
        ax2 = fig.add_subplot(gs[0, 1])
        avg_score = df.groupby('vintage_year')['CREDIT_SCORE'].mean()
        ax2.plot(avg_score.index, avg_score.values, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('Vintage Year', fontsize=12)
        ax2.set_ylabel('Average Credit Score', fontsize=12)
        ax2.set_title('Average Credit Score by Vintage', fontsize=14)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Average LTV by Vintage
        ax3 = fig.add_subplot(gs[0, 2])
        avg_ltv = df.groupby('vintage_year')['ORIGINAL_LTV'].mean()
        ax3.plot(avg_ltv.index, avg_ltv.values, 'o-', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Vintage Year', fontsize=12)
        ax3.set_ylabel('Average LTV (%)', fontsize=12)
        ax3.set_title('Average LTV by Vintage', fontsize=14)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Average DTI by Vintage
        ax4 = fig.add_subplot(gs[1, 0])
        avg_dti = df.groupby('vintage_year')['ORIGINAL_DTI'].mean()
        ax4.plot(avg_dti.index, avg_dti.values, 'o-', linewidth=2, markersize=8, color='red')
        ax4.set_xlabel('Vintage Year', fontsize=12)
        ax4.set_ylabel('Average DTI (%)', fontsize=12)
        ax4.set_title('Average DTI by Vintage', fontsize=14)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 5. Average Interest Rate by Vintage
        ax5 = fig.add_subplot(gs[1, 1])
        avg_rate = df.groupby('vintage_year')['ORIGINAL_INTEREST_RATE'].mean()
        ax5.plot(avg_rate.index, avg_rate.values, 'o-', linewidth=2, markersize=8, color='orange')
        ax5.set_xlabel('Vintage Year', fontsize=12)
        ax5.set_ylabel('Average Interest Rate (%)', fontsize=12)
        ax5.set_title('Average Interest Rate by Vintage', fontsize=14)
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # 6. Default Rate by Vintage
        ax6 = fig.add_subplot(gs[1, 2])
        if 'target' in df.columns:
            default_rate = df.groupby('vintage_year')['target'].mean()
            ax6.plot(default_rate.index, default_rate.values * 100, 'o-', 
                    linewidth=2, markersize=8, color='purple')
            ax6.set_xlabel('Vintage Year', fontsize=12)
            ax6.set_ylabel('Default Rate (%)', fontsize=12)
            ax6.set_title('Default Rate by Vintage', fontsize=14)
            ax6.tick_params(axis='x', rotation=45)
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, "Target not available\n(Feature engineering not yet run)", 
                    ha='center', va='center', fontsize=12)
            ax6.set_title('Default Rate by Vintage (Not Available)', fontsize=14)
        
        plt.suptitle('Vintage Year Trends', fontsize=16, y=1.02)
        plt.tight_layout()
        return fig
    
    # ========================================================================
    # UNIVARIATE ANALYSIS - PERFORMANCE
    # ========================================================================
    
    def plot_performance_distributions(self, df: pd.DataFrame) -> plt.Figure:
        """
        Plot distributions of key performance variables.
        
        Args:
            df: Pandas DataFrame with performance data
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Delinquency Status Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        status_counts = df['CURRENT_LOAN_DELINQUENCY_STATUS'].value_counts().head(10)
        ax1.bar(status_counts.index, status_counts.values, color='coral')
        ax1.set_xlabel('Delinquency Status', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Delinquency Status Distribution', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Current UPB Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_histogram(df, 'CURRENT_ACTUAL_UPB', ax2,
                            title='Current UPB Distribution',
                            xlabel='Current UPB ($)', log_scale=True)
        
        # 3. Loan Age Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_histogram(df, 'LOAN_AGE', ax3,
                            title='Loan Age Distribution',
                            xlabel='Loan Age (months)')
        
        # 4. Current Interest Rate Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_histogram(df, 'CURRENT_INTEREST_RATE', ax4,
                            title='Current Interest Rate Distribution',
                            xlabel='Interest Rate (%)')
        
        # 5. Remaining Months Distribution
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_histogram(df, 'REMAINING_MONTHS_TO_LEGAL_MATURITY', ax5,
                            title='Remaining Months to Maturity',
                            xlabel='Remaining Months')
        
        # 6. Modification Flag
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_categorical(df, 'MODIFICATION_FLAG', ax6,
                              title='Modification Flag')
        
        # 7. Zero Balance Code (if available)
        ax7 = fig.add_subplot(gs[2, 0])
        if 'ZERO_BALANCE_CODE' in df.columns:
            zb_counts = df['ZERO_BALANCE_CODE'].value_counts().dropna()
            ax7.bar(zb_counts.index, zb_counts.values, color='lightgreen')
            ax7.set_xlabel('Zero Balance Code', fontsize=12)
            ax7.set_ylabel('Count', fontsize=12)
            ax7.set_title('Zero Balance Code Distribution', fontsize=14)
            ax7.tick_params(axis='x', rotation=45)
        else:
            ax7.text(0.5, 0.5, "Zero Balance Code not available", 
                    ha='center', va='center', fontsize=12)
            ax7.set_title('Zero Balance Code Distribution', fontsize=14)
        
        # 8. Payment Deferral Flag
        ax8 = fig.add_subplot(gs[2, 1])
        if 'PAYMENT_DEFERRAL_FLAG' in df.columns:
            self._plot_categorical(df, 'PAYMENT_DEFERRAL_FLAG', ax8,
                                  title='Payment Deferral Flag')
        else:
            ax8.text(0.5, 0.5, "Payment Deferral not available", 
                    ha='center', va='center', fontsize=12)
            ax8.set_title('Payment Deferral Flag', fontsize=14)
        
        # 9. ELTV Distribution
        ax9 = fig.add_subplot(gs[2, 2])
        if 'ELTV' in df.columns:
            self._plot_histogram(df, 'ELTV', ax9,
                                title='ELTV Distribution',
                                xlabel='ELTV (%)')
        else:
            ax9.text(0.5, 0.5, "ELTV not available", 
                    ha='center', va='center', fontsize=12)
            ax9.set_title('ELTV Distribution', fontsize=14)
        
        plt.suptitle('Performance Data Distributions', fontsize=16, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_delinquency_timeline(self, df: pd.DataFrame) -> plt.Figure:
        """
        Plot delinquency rates over time.
        
        Args:
            df: Pandas DataFrame with performance data
            
        Returns:
            Matplotlib figure
        """
        # Parse reporting period
        if 'MONTHLY_REPORTING_PERIOD' in df.columns:
            df['reporting_date'] = pd.to_datetime(
                df['MONTHLY_REPORTING_PERIOD'].astype(str) + '01', 
                format='%Y%m%d'
            )
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # 1. Delinquency rate over time
            monthly_stats = df.groupby('reporting_date').agg({
                'CURRENT_LOAN_DELINQUENCY_STATUS': lambda x: (x != '0').mean()
            }).reset_index()
            monthly_stats.columns = ['date', 'delinquency_rate']
            
            ax1.plot(monthly_stats['date'], monthly_stats['delinquency_rate'] * 100, 
                    linewidth=2, color='darkred')
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Delinquency Rate (%)', fontsize=12)
            ax1.set_title('Delinquency Rate Over Time', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Severity of delinquency
            severity = df.groupby('reporting_date')['CURRENT_LOAN_DELINQUENCY_STATUS'].value_counts().unstack().fillna(0)
            severity_pct = severity.div(severity.sum(axis=1), axis=0) * 100
            
            # Plot only top delinquency levels
            top_levels = ['0', '1', '2', '3']
            available_levels = [l for l in top_levels if l in severity_pct.columns]
            
            for level in available_levels:
                label_map = {'0': 'Current', '1': '30 Days', '2': '60 Days', '3': '90+ Days'}
                ax2.plot(severity_pct.index, severity_pct[level], 
                        label=label_map.get(level, f'{level} Days'),
                        linewidth=2)
            
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Percentage of Loans (%)', fontsize=12)
            ax2.set_title('Delinquency Severity Over Time', fontsize=14)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            plt.suptitle('Delinquency Trends Over Time', fontsize=16, y=1.02)
            plt.tight_layout()
            return fig
        
        return None
    
    # ========================================================================
    # BIVARIATE ANALYSIS
    # ========================================================================
    
    def plot_target_analysis(self, df: pd.DataFrame) -> plt.Figure:
        """
        Plot target variable analysis with key predictors.
        
        Args:
            df: Pandas DataFrame with features and target
            
        Returns:
            Matplotlib figure
        """
        if 'target' not in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Target variable not available\n(Run feature engineering first)",
                   ha='center', va='center', fontsize=14)
            ax.set_title('Target Analysis (Not Available)')
            return fig
        
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Target Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        target_counts = df['target'].value_counts()
        colors = ['green', 'red']
        ax1.pie(target_counts.values, labels=['Non-Default', 'Default'], 
                autopct='%1.2f%%', colors=colors, startangle=90)
        ax1.set_title('Target Distribution', fontsize=14)
        
        # 2. Credit Score by Target
        ax2 = fig.add_subplot(gs[0, 1])
        data = [df[df['target'] == 0]['CREDIT_SCORE'].dropna(),
                df[df['target'] == 1]['CREDIT_SCORE'].dropna()]
        ax2.boxplot(data, labels=['Non-Default', 'Default'])
        ax2.set_ylabel('Credit Score', fontsize=12)
        ax2.set_title('Credit Score by Default Status', fontsize=14)
        
        # 3. LTV by Target
        ax3 = fig.add_subplot(gs[0, 2])
        data = [df[df['target'] == 0]['ORIGINAL_LTV'].dropna(),
                df[df['target'] == 1]['ORIGINAL_LTV'].dropna()]
        ax3.boxplot(data, labels=['Non-Default', 'Default'])
        ax3.set_ylabel('Original LTV (%)', fontsize=12)
        ax3.set_title('LTV by Default Status', fontsize=14)
        
        # 4. DTI by Target
        ax4 = fig.add_subplot(gs[1, 0])
        data = [df[df['target'] == 0]['ORIGINAL_DTI'].dropna(),
                df[df['target'] == 1]['ORIGINAL_DTI'].dropna()]
        ax4.boxplot(data, labels=['Non-Default', 'Default'])
        ax4.set_ylabel('Original DTI (%)', fontsize=12)
        ax4.set_title('DTI by Default Status', fontsize=14)
        
        # 5. Default Rate by Delinquency Status
        ax5 = fig.add_subplot(gs[1, 1])
        if 'delinquency_numeric' in df.columns:
            default_by_delinq = df.groupby('delinquency_numeric')['target'].mean() * 100
            ax5.bar(default_by_delinq.index, default_by_delinq.values, color='coral')
            ax5.set_xlabel('Current Delinquency Status', fontsize=12)
            ax5.set_ylabel('Future Default Rate (%)', fontsize=12)
            ax5.set_title('Future Default by Current Delinquency', fontsize=14)
        else:
            ax5.text(0.5, 0.5, "Delinquency status not available", 
                    ha='center', va='center', fontsize=12)
        
        # 6. Default Rate by Vintage
        ax6 = fig.add_subplot(gs[1, 2])
        if 'LOAN_SEQUENCE_NUMBER' in df.columns:
            df['vintage_year'] = df['LOAN_SEQUENCE_NUMBER'].str[1:3].astype(int) + 2000
            default_by_vintage = df.groupby('vintage_year')['target'].mean() * 100
            ax6.plot(default_by_vintage.index, default_by_vintage.values, 'o-', 
                    linewidth=2, markersize=8, color='purple')
            ax6.set_xlabel('Vintage Year', fontsize=12)
            ax6.set_ylabel('Default Rate (%)', fontsize=12)
            ax6.set_title('Default Rate by Vintage', fontsize=14)
            ax6.grid(True, alpha=0.3)
        
        # 7. Feature Importance Preview (Correlation)
        ax7 = fig.add_subplot(gs[2, 0:3])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'target' and c != 'delinquency_numeric']
        
        if len(numeric_cols) > 0:
            # Calculate correlations with target
            correlations = df[numeric_cols + ['target']].corr()['target'].drop('target')
            correlations = correlations.abs().sort_values(ascending=True).tail(15)
            
            ax7.barh(correlations.index, correlations.values, color='steelblue')
            ax7.set_xlabel('Absolute Correlation with Target', fontsize=12)
            ax7.set_title('Top 15 Features by Correlation with Default', fontsize=14)
            ax7.grid(True, alpha=0.3)
        
        plt.suptitle('Target Analysis: Default Prediction', fontsize=16, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, max_vars: int = 20) -> plt.Figure:
        """
        Plot correlation matrix for numeric variables.
        
        Args:
            df: Pandas DataFrame
            max_vars: Maximum number of variables to show
            
        Returns:
            Matplotlib figure
        """
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Limit to top variables by variance if too many
        if len(numeric_cols) > max_vars:
            variances = df[numeric_cols].var()
            numeric_cols = variances.nlargest(max_vars).index
        
        if len(numeric_cols) < 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Insufficient numeric variables for correlation matrix",
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='RdYlBu_r', center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title('Correlation Matrix', fontsize=16)
        plt.tight_layout()
        return fig
    
    # ========================================================================
    # TIME SERIES ANALYSIS
    # ========================================================================
    
    def plot_monthly_trends(self, df: pd.DataFrame) -> plt.Figure:
        """
        Plot monthly trends in key performance metrics.
        
        Args:
            df: Pandas DataFrame with performance data
            
        Returns:
            Matplotlib figure
        """
        if 'MONTHLY_REPORTING_PERIOD' not in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Monthly reporting period not available",
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Parse reporting period
        df['reporting_date'] = pd.to_datetime(
            df['MONTHLY_REPORTING_PERIOD'].astype(str) + '01',
            format='%Y%m%d'
        )
        
        # Aggregate by month
        monthly_agg = df.groupby('reporting_date').agg({
            'LOAN_AGE': 'mean',
            'CURRENT_ACTUAL_UPB': 'mean',
            'CURRENT_INTEREST_RATE': 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 1. Average Loan Age
        axes[0].plot(monthly_agg['reporting_date'], monthly_agg['LOAN_AGE'],
                    linewidth=2, color='blue')
        axes[0].set_xlabel('Date', fontsize=12)
        axes[0].set_ylabel('Average Loan Age (months)', fontsize=12)
        axes[0].set_title('Average Loan Age Over Time', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. Average Current UPB
        axes[1].plot(monthly_agg['reporting_date'], monthly_agg['CURRENT_ACTUAL_UPB'] / 1000,
                    linewidth=2, color='green')
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Average Current UPB ($000)', fontsize=12)
        axes[1].set_title('Average Current UPB Over Time', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        # 3. Average Current Interest Rate
        axes[2].plot(monthly_agg['reporting_date'], monthly_agg['CURRENT_INTEREST_RATE'],
                    linewidth=2, color='red')
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].set_ylabel('Average Interest Rate (%)', fontsize=12)
        axes[2].set_title('Average Current Interest Rate Over Time', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Monthly Trends in Performance Metrics', fontsize=16, y=1.02)
        plt.tight_layout()
        return fig
    
    # ========================================================================
    # GEOGRAPHIC ANALYSIS
    # ========================================================================
    
    def plot_geographic_distribution(self, df: pd.DataFrame) -> plt.Figure:
        """
        Plot geographic distribution of loans by state.
        
        Args:
            df: Pandas DataFrame with property state
            
        Returns:
            Matplotlib figure
        """
        if 'PROPERTY_STATE' not in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Property state not available",
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Count by state
        state_counts = df['PROPERTY_STATE'].value_counts().head(20)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Top states by loan count
        ax1.bar(state_counts.index, state_counts.values, color='steelblue')
        ax1.set_xlabel('State', fontsize=12)
        ax1.set_ylabel('Number of Loans', fontsize=12)
        ax1.set_title('Top 20 States by Loan Volume', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Default rate by state (if target available)
        if 'target' in df.columns:
            state_default = df.groupby('PROPERTY_STATE')['target'].mean() * 100
            state_default = state_default.sort_values(ascending=False).head(20)
            
            ax2.bar(state_default.index, state_default.values, color='coral')
            ax2.set_xlabel('State', fontsize=12)
            ax2.set_ylabel('Default Rate (%)', fontsize=12)
            ax2.set_title('Top 20 States by Default Rate', fontsize=14)
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, "Target not available\n(Run feature engineering first)",
                    ha='center', va='center', fontsize=12)
            ax2.set_title('Default Rate by State (Not Available)', fontsize=14)
        
        plt.suptitle('Geographic Distribution of Loans', fontsize=16, y=1.02)
        plt.tight_layout()
        return fig
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _plot_histogram(self, df: pd.DataFrame, col_name: str, ax: plt.Axes,
                       title: str, xlabel: str, log_scale: bool = False):
        """Helper to plot histogram with statistics."""
        data = df[col_name].dropna()
        if len(data) == 0:
            ax.text(0.5, 0.5, f"No data for {col_name}", 
                   ha='center', va='center')
            return
        
        ax.hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        
        if log_scale:
            ax.set_xscale('log')
        
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(title, fontsize=12)
        
        # Add statistics
        stats = f'n={len(data):,}\nμ={data.mean():.2f}\nσ={data.std():.2f}\nmin={data.min():.2f}\nmax={data.max():.2f}'
        ax.text(0.95, 0.95, stats, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_categorical(self, df: pd.DataFrame, col_name: str, ax: plt.Axes,
                         title: str, top_n: int = 10):
        """Helper to plot categorical distribution."""
        data = df[col_name].value_counts().head(top_n)
        
        if len(data) == 0:
            ax.text(0.5, 0.5, f"No data for {col_name}",
                   ha='center', va='center')
            return
        
        bars = ax.bar(data.index, data.values, color='lightgreen')
        ax.set_xlabel(col_name, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        total = data.sum()
        for i, (idx, val) in enumerate(data.items()):
            pct = (val / total) * 100
            ax.text(i, val + total * 0.01, f'{pct:.1f}%',
                   ha='center', fontsize=8)
    
    # ========================================================================
    # COMPREHENSIVE EDA REPORT
    # ========================================================================
    
    def create_eda_report(
        self,
        orig_df: pd.DataFrame,
        perf_df: pd.DataFrame,
        feature_df: pd.DataFrame = None
    ) -> None:
        """
        Create comprehensive EDA report with all visualizations.
        
        Args:
            orig_df: Origination DataFrame
            perf_df: Performance DataFrame
            feature_df: Feature DataFrame (optional)
        """
        logger.info("Creating EDA Report...")
        
        # Data Quality
        logger.info("  - Data Quality Analysis")
        fig1 = self.plot_missing_values(orig_df, "Missing Values - Origination")
        fig2 = self.plot_missing_values(perf_df, "Missing Values - Performance")
        fig3 = self.plot_data_types(orig_df)
        
        # Origination Analysis
        logger.info("  - Origination Analysis")
        fig4 = self.plot_origination_distributions(orig_df)
        fig5 = self.plot_origination_by_vintage(orig_df)
        
        # Performance Analysis
        logger.info("  - Performance Analysis")
        fig6 = self.plot_performance_distributions(perf_df)
        fig7 = self.plot_delinquency_timeline(perf_df)
        fig8 = self.plot_monthly_trends(perf_df)
        
        # Geographic Analysis
        logger.info("  - Geographic Analysis")
        fig9 = self.plot_geographic_distribution(orig_df)
        
        # Feature Analysis (if available)
        if feature_df is not None:
            logger.info("  - Feature Analysis")
            fig10 = self.plot_target_analysis(feature_df)
            fig11 = self.plot_correlation_matrix(feature_df)
        
        # Save or show
        if self.save_dir:
            import os
            logger.info(f"  - Saving figures to {self.save_dir}")
            figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]
            names = ['missing_orig', 'missing_perf', 'data_types', 'orig_distributions',
                    'orig_vintage', 'perf_distributions', 'delinquency_timeline',
                    'monthly_trends', 'geographic']
            
            if feature_df is not None:
                figs.extend([fig10, fig11])
                names.extend(['target_analysis', 'correlation_matrix'])
            
            for fig, name in zip(figs, names):
                fig.savefig(f"{self.save_dir}/{name}.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
            
            logger.info("  - EDA Report saved successfully")
        else:
            plt.show()
        
        logger.info("EDA Report completed!")