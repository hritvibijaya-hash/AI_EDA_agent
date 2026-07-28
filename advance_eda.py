
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def eda_by_ai(df):
    """
    Performs an exhaustive, end-to-end Advanced Data Analysis on the provided DataFrame `df`.
    All code is self-contained within this function as requested.
    """
    
    # ==========================================
    # Phase 1: Environment Setup & Data Loading
    # ==========================================
    # Note: df is already loaded and passed as an argument.
    print("=" * 60)
    print("PHASE 1: ENVIRONMENT SETUP & DATA METADATA")
    print("=" * 60)
    
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nData Types and Non-Null Counts:")
    print(df.info())
    
    print("\nMemory Usage:")
    print(df.memory_usage(deep=True))
    
    print("\nFirst 5 Rows:")
    display_func = print if not hasattr(pd.DataFrame, '_repr_html_') else display
    display_func(df.head())

    # ==========================================
    # Phase 2: Automated Dataset Overview (`describe`)
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 2: AUTOMATED DATASET OVERVIEW")
    print("=" * 60)
    
    # Numerical descriptive statistics including skewness and kurtosis
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        print("\nNumerical Columns Descriptive Statistics:")
        num_desc = df[num_cols].describe().T
        num_desc['skew'] = df[num_cols].skew()
        num_desc['kurtosis'] = df[num_cols].kurtosis()
        # Calculate IQR
        Q1 = df[num_cols].quantile(0.25)
        Q3 = df[num_cols].quantile(0.75)
        num_desc['iqr'] = Q3 - Q1
        print(num_desc)
    else:
        print("\nNo numerical columns found.")

    # Categorical descriptive statistics
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    if len(cat_cols) > 0:
        print("\nCategorical Columns Descriptive Statistics:")
        print(df[cat_cols].describe(include='all').T)
    else:
        print("\nNo categorical columns found.")

    # Missing values and duplicates
    print("\nMissing Value Percentages:")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': df.isnull().sum(), 'Missing Percentage (%)': missing_pct})
    print(missing_df[missing_df['Missing Count'] > 0])
    
    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate Row Count: {duplicate_count}")

    # ==========================================
    # Phase 3: Correlation Analysis (`corr`)
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 3: CORRELATION ANALYSIS")
    print("=" * 60)
    
    if len(num_cols) > 1:
        # Compute Pearson correlation matrix
        corr_matrix = df[num_cols].corr(method='pearson')
        
        # Visualize using Seaborn heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True)
        plt.title('Pearson Correlation Matrix Heatmap')
        plt.tight_layout()
        plt.show()
        
        # Unstack correlation matrix to find top strongest positive and negative correlations (excluding self-correlation)
        corr_unstacked = corr_matrix.unstack()
        corr_unstacked = corr_unstacked[corr_unstacked.index.get_level_values(0) != corr_unstacked.index.get_level_values(1)]
        # Drop duplicate pairs
        corr_unstacked = corr_unstacked.drop_duplicates()
        
        print("\nTop 5 Strongest Positive Correlations:")
        print(corr_unstacked.nlargest(5))
        
        print("\nTop 5 Strongest Negative Correlations:")
        print(corr_unstacked.nsmallest(5))
    else:
        print("\nInsufficient numerical columns for correlation analysis.")

    # ==========================================
    # Phase 4: Univariate Analysis
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 4: UNIVARIATE ANALYSIS")
    print("=" * 60)
    
    # Numerical: Histograms with KDE and Boxplots
    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram with KDE
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f'Distribution & KDE of {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')
        
        # Boxplot
        sns.boxplot(x=df[col], ax=axes[1], color='lightgreen')
        axes[1].set_title(f'Boxplot of {col} (Outlier Detection)')
        axes[1].set_xlabel(col)
        
        plt.tight_layout()
        plt.show()
        
    # Categorical: Count plots (Top 10 if high cardinality)
    for col in cat_cols:
        plt.figure(figsize=(10, 5))
        top_cats = df[col].value_counts().nlargest(10)
        sns.barplot(x=top_cats.index, y=top_cats.values, palette='viridis')
        plt.title(f'Frequency Distribution of {col} (Top 10)')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # ==========================================
    # Phase 5: Bivariate Analysis
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 5: BIVARIATE ANALYSIS")
    print("=" * 60)
    
    if len(cat_cols) > 0 and len(num_cols) > 0:
        # Pick the first categorical and first numerical column as default demonstration drivers
        sample_cat = cat_cols[0]
        sample_num = num_cols[0]
        print(f"\nGrouped Summary Statistics for '{sample_num}' grouped by '{sample_cat}':")
        
        grouped_stats = df.groupby(sample_cat)[sample_num].agg(['mean', 'median', 'sum', 'count']).reset_index()
        print(grouped_stats)
        
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x=sample_cat, y=sample_num, estimator=np.mean, ci=None, palette='muted')
        plt.title(f'Mean of {sample_num} by {sample_cat}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("\nSkipping bivariate categorical-numerical analysis due to missing column types.")

    # ==========================================
    # Phase 6: Time Series Analysis (Conditional)
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 6: TIME SERIES ANALYSIS")
    print("=" * 60)
    
    date_col = None
    # Programmatically check for potential date/time columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col = col
            break
        else:
            # Try parsing a sample of object/string columns as dates
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].dropna().iloc[:5])
                    date_col = col
                    break
                except (ValueError, TypeError):
                    continue

    if date_col and len(num_cols) > 0:
        print(f"\nDetected datetime-parseable column: '{date_col}'")
        # Create a working copy for time series processing
        ts_df = df.copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col])
        ts_df = ts_df.sort_values(by=date_col)
        ts_df.set_index(date_col, inplace=True)
        
        target_num = num_cols[0]
        # Resample monthly
        resampled_ts = ts_df[target_num].resample('M').sum().fillna(method='ffill')
        
        plt.figure(figsize=(12, 4))
        resampled_ts.plot(title=f'Monthly Resampled Trend of {target_num}')
        plt.xlabel('Date')
        plt.ylabel(target_num)
        plt.tight_layout()
        plt.show()
        
        # Decompose time series if data points are sufficient (>= 24 periods for yearly seasonality)
        if len(resampled_ts) >= 24:
            try:
                decomposition = seasonal_decompose(resampled_ts, model='additive', period=12)
                fig = decomposition.plot()
                fig.set_size_inches(12, 8)
                plt.tight_layout()
                plt.show()
                print("\nTime Series successfully decomposed into Trend, Seasonality, and Residuals.")
            except Exception as e:
                print(f"\nCould not perform time series decomposition: {e}")
        else:
            print("\nNot enough data points for seasonal decomposition (requires >= 24 periods).")
    else:
        print("\nNo valid datetime column found for Time Series Analysis.")

    # ==========================================
    # Phase 7: Multivariate Analysis & Grouped Visualizations
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 7: MULTIVARIATE ANALYSIS & GROUPED VISUALIZATIONS")
    print("=" * 60)
    
    if len(cat_cols) >= 2 and len(num_cols) >= 1:
        primary_cat = cat_cols[0]
        hue_cat = cat_cols[1]
        target_num = num_cols[0]
        
        print(f"\nGenerating multivariate bar plot: X={primary_cat}, Hue={hue_cat}, Metric={target_num}")
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df, x=primary_cat, y=target_num, hue=hue_cat, estimator=np.mean, ci=None, palette='Set2')
        plt.title(f'Multivariate Analysis: {target_num} by {primary_cat} and {hue_cat}')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=hue_cat, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    else:
        print("\nInsufficient categorical columns (need at least 2) for advanced grouped multivariate bar plots.")

    # ==========================================
    # Phase 8: Automated Business Insights & Recommendations
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 8: AUTOMATED BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("=" * 60)
    
    insights = []
    
    # Insight on missing values
    high_missing = missing_df[missing_df['Missing Percentage (%)'] > 20]
    if not high_missing.empty:
        insights.append(f"- Data Quality Risk: The following columns have >20% missing values and require imputation or dropping: {list(high_missing.index)}")
    else:
        insights.append("- Data Quality: Missing values are minimal or non-existent across high-impact columns.")
        
    # Insight on duplicates
    if duplicate_count > 0:
        insights.append(f"- Data Hygiene: Found {duplicate_count} duplicate rows. Consider deduplicating prior to machine learning modeling.")
        
    # Insight on numerical distributions
    if len(num_cols) > 0:
        skewed_cols = num_cols[df[num_cols].skew().abs() > 1.5]
        if not skewed_cols.empty:
            insights.append(f"- Feature Distribution: Highly skewed numerical features detected ({list(skewed_cols)}). Apply transformations (e.g., log/Box-Cox) for parametric models.")
            
    # Print final summary block
    print("\nActionable Insights Summary:")
    for insight in insights:
        print(insight)
    print("\nEDA by AI execution complete.")
