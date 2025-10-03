# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "seaborn",
#     "matplotlib",
#     "httpx",
#     "tenacity",
#     "numpy",
#     "scipy",
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import json
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
API_URL = "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"
MODEL = "gpt-4o-mini"
MAX_RETRIES = 3


class DataAnalyzer:
    """Main class for automated data analysis"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.analysis_results = {}
        self.token = os.environ.get("AIPROXY_TOKEN")
        
        if not self.token:
            raise ValueError("AIPROXY_TOKEN environment variable not set")
    
    def load_data(self):
        """Load CSV file with encoding detection"""
        try:
            self.df = pd.read_csv(self.csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.csv_path, encoding='latin-1')
        
        print(f"Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def basic_analysis(self):
        """Perform basic statistical analysis"""
        analysis = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "missing_percentage": (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
        }
        
        # Summary statistics for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            analysis["numeric_summary"] = self.df[numeric_cols].describe().to_dict()
        
        # Value counts for categorical columns (top 5)
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        analysis["categorical_info"] = {}
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            analysis["categorical_info"][col] = {
                "unique_count": self.df[col].nunique(),
                "top_values": self.df[col].value_counts().head(5).to_dict()
            }
        
        self.analysis_results["basic"] = analysis
        return analysis
    
    def correlation_analysis(self):
        """Calculate correlation matrix for numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return None
        
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find strong correlations (excluding diagonal)
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": round(corr_val, 3)
                    })
        
        self.analysis_results["correlation"] = {
            "matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_corr
        }
        return corr_matrix
    
    def outlier_detection(self):
        """Detect outliers using IQR method"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        outliers_info = {}
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outliers_info[col] = {
                    "count": outlier_count,
                    "percentage": round(outlier_count / len(self.df) * 100, 2),
                    "lower_bound": round(lower_bound, 2),
                    "upper_bound": round(upper_bound, 2)
                }
        
        self.analysis_results["outliers"] = outliers_info
        return outliers_info
    
    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
    def call_llm(self, messages, functions=None):
        """Call LLM API with retry logic"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
        }
        
        if functions:
            payload["functions"] = functions
            payload["function_call"] = "auto"
        
        response = httpx.post(API_URL, headers=headers, json=payload, timeout=60.0)
        response.raise_for_status()
        return response.json()
    
    def get_analysis_suggestions(self):
        """Ask LLM for analysis suggestions based on data structure"""
        summary = {
            "columns": self.analysis_results["basic"]["columns"],
            "dtypes": self.analysis_results["basic"]["dtypes"],
            "shape": self.analysis_results["basic"]["shape"],
            "missing_percentage": {k: v for k, v in self.analysis_results["basic"]["missing_percentage"].items() if v > 0},
            "numeric_summary": self.analysis_results["basic"].get("numeric_summary", {}),
        }
        
        prompt = f"""You are a data analyst. Given this dataset summary:
{json.dumps(summary, indent=2)}

Suggest 2-3 specific and insightful analyses that would be valuable for this dataset.
Be specific about which columns to analyze and what insights to look for.
Keep suggestions practical and implementable."""

        messages = [
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.call_llm(messages)
            suggestions = response["choices"][0]["message"]["content"]
            self.analysis_results["llm_suggestions"] = suggestions
            return suggestions
        except Exception as e:
            print(f"Error getting LLM suggestions: {e}")
            return "Could not generate suggestions"
    
    def create_visualizations(self):
        """Create insightful visualizations"""
        fig_paths = []
        
        # 1. Correlation Heatmap (if numeric columns exist)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(10, 8))
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1)
            plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('correlation_matrix.png', dpi=100, bbox_inches='tight')
            plt.close()
            fig_paths.append('correlation_matrix.png')
            print("Created: correlation_matrix.png")
        
        # 2. Missing Data Visualization
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            plt.figure(figsize=(10, 6))
            missing_data.plot(kind='bar', color='salmon')
            plt.title('Missing Values by Column', fontsize=16, fontweight='bold')
            plt.xlabel('Columns', fontsize=12)
            plt.ylabel('Number of Missing Values', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('missing_values.png', dpi=100, bbox_inches='tight')
            plt.close()
            fig_paths.append('missing_values.png')
            print("Created: missing_values.png")
        
        # 3. Distribution of first numeric column or most interesting insight
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            plt.figure(figsize=(10, 6))
            
            # Create subplot with histogram and boxplot
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Histogram
            axes[0].hist(self.df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
            axes[0].set_title(f'Distribution of {col}', fontsize=16, fontweight='bold')
            axes[0].set_xlabel(col, fontsize=12)
            axes[0].set_ylabel('Frequency', fontsize=12)
            axes[0].grid(alpha=0.3)
            
            # Boxplot
            axes[1].boxplot(self.df[col].dropna(), vert=False, widths=0.7)
            axes[1].set_xlabel(col, fontsize=12)
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('distribution_analysis.png', dpi=100, bbox_inches='tight')
            plt.close()
            fig_paths.append('distribution_analysis.png')
            print("Created: distribution_analysis.png")
        
        return fig_paths
    
    def generate_story(self, image_paths):
        """Generate narrative story using LLM"""
        # Prepare analysis summary for LLM
        analysis_summary = {
            "dataset_info": {
                "rows": self.analysis_results["basic"]["shape"][0],
                "columns": self.analysis_results["basic"]["shape"][1],
                "column_names": self.analysis_results["basic"]["columns"][:10],  # Limit columns
            },
            "missing_data": {k: v for k, v in self.analysis_results["basic"]["missing_percentage"].items() if v > 5},
            "key_insights": {
                "correlations": self.analysis_results.get("correlation", {}).get("strong_correlations", [])[:5],
                "outliers": self.analysis_results.get("outliers", {}),
            }
        }
        
        prompt = f"""You are a data storyteller. Write an engaging and professional analysis report in Markdown format.

Dataset Analysis Summary:
{json.dumps(analysis_summary, indent=2)}

Write a comprehensive README.md that includes:

1. **# Dataset Overview**: Brief description of the data (2-3 sentences)
2. **## Analysis Performed**: What analyses were conducted (bullet points)
3. **## Key Insights**: Most important findings (3-5 insights with specific numbers)
4. **## Visualizations**: Describe what each chart shows
5. **## Implications & Recommendations**: What actions should be taken based on these insights

Make it engaging, use specific numbers from the analysis, and focus on actionable insights.
Use proper Markdown formatting with headers, bullet points, and emphasis.
Keep it concise but insightful (400-600 words).

Include image references like: ![Description](image_name.png)"""

        messages = [
            {"role": "system", "content": "You are an expert data analyst and storyteller who creates clear, insightful narratives from data."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.call_llm(messages)
            story = response["choices"][0]["message"]["content"]
            
            # Add image references if not already included
            for img_path in image_paths:
                if img_path not in story:
                    story += f"\n\n![Analysis Chart]({img_path})\n"
            
            return story
        except Exception as e:
            print(f"Error generating story: {e}")
            # Fallback story
            return self.create_fallback_story(image_paths)
    
    def create_fallback_story(self, image_paths):
        """Create a basic story if LLM fails"""
        story = f"""# Dataset Analysis Report

## Dataset Overview
This analysis examines a dataset with {self.df.shape[0]} rows and {self.df.shape[1]} columns.

## Key Statistics
- Total records: {self.df.shape[0]}
- Total features: {self.df.shape[1]}
- Missing data found in {len([k for k, v in self.analysis_results['basic']['missing_percentage'].items() if v > 0])} columns

## Visualizations
"""
        for img in image_paths:
            story += f"\n![Analysis]({img})\n"
        
        return story
    
    def save_story(self, story):
        """Save the narrative to README.md"""
        with open('README.md', 'w', encoding='utf-8') as f:
            f.write(story)
        print("Created: README.md")
    
    def run_analysis(self):
        """Main analysis pipeline"""
        print("=" * 50)
        print("Starting Automated Analysis")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Perform analyses
        print("\n[1/6] Running basic analysis...")
        self.basic_analysis()
        
        print("[2/6] Analyzing correlations...")
        self.correlation_analysis()
        
        print("[3/6] Detecting outliers...")
        self.outlier_detection()
        
        print("[4/6] Getting LLM suggestions...")
        self.get_analysis_suggestions()
        
        print("[5/6] Creating visualizations...")
        image_paths = self.create_visualizations()
        
        print("[6/6] Generating story...")
        story = self.generate_story(image_paths)
        self.save_story(story)
        
        print("\n" + "=" * 50)
        print("Analysis Complete!")
        print(f"Generated: README.md and {len(image_paths)} PNG files")
        print("=" * 50)


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found")
        sys.exit(1)
    
    try:
        analyzer = DataAnalyzer(csv_path)
        analyzer.run_analysis()
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()