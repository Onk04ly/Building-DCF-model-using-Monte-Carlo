import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Optional
import seaborn as sns # type: ignore # For advanced visualizations 
import pandas as pd

# Set style for seaborn for better visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")



class DCFVisualizer:
    """
    A comprehensive visualization class for Monte Carlo DCF analysis.
    
    This class provides all the visualization tools needed for professional
    DCF analysis, including distribution plots, risk analysis, and sensitivity charts.
    """
    
    def __init__(self, mc_results: Dict, company_name: str = "Company"):
        """
        Initialize the DCF Visualizer.
        
        Parameters:
        -----------
        mc_results : Dict
            Results from monte_carlo_dcf() function
        company_name : str
            Name of the company being analyzed
        """
        self.mc_results = mc_results
        self.company_name = company_name
        self.valuations = mc_results['valuations']
        self.valuations_billions = self.valuations / 1000  # Convert to billions
        
    def plot_valuation_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive distribution plot of valuations.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, plot is displayed.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.company_name} - Monte Carlo DCF Valuation Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Histogram with density curve
        ax1.hist(self.valuations_billions, bins=50, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black', linewidth=0.5)
        
        # Add density curve
        x = np.linspace(self.valuations_billions.min(), self.valuations_billions.max(), 100)
        kde = stats.gaussian_kde(self.valuations_billions)
        ax1.plot(x, kde(x), 'r-', linewidth=2, label='Density Curve')
        
        # Add mean and median lines
        ax1.axvline(self.mc_results['mean']/1000, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: ${self.mc_results["mean"]/1000:.1f}B')
        ax1.axvline(self.mc_results['median']/1000, color='green', linestyle='--', 
                   linewidth=2, label=f'Median: ${self.mc_results["median"]/1000:.1f}B')
        
        ax1.set_xlabel('Valuation ($ Billions)')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Valuations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot with outliers
        ax2.boxplot(self.valuations_billions, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_ylabel('Valuation ($ Billions)')
        ax2.set_title('Box Plot - Outliers & Quartiles')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative Distribution Function (CDF)
        sorted_vals = np.sort(self.valuations_billions)
        cumulative_prob = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax3.plot(sorted_vals, cumulative_prob, 'b-', linewidth=2)
        ax3.set_xlabel('Valuation ($ Billions)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Distribution Function')
        ax3.grid(True, alpha=0.3)
        
        # Add percentile lines
        percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        for p in percentiles:
            val = np.percentile(self.valuations_billions, p*100)
            ax3.axvline(val, color='red', linestyle=':', alpha=0.7)
            ax3.text(val, p, f'{p*100:.0f}%', rotation=90, va='bottom')
        
        # 4. Q-Q Plot (Quantile-Quantile) for normality check
        stats.probplot(self.valuations_billions, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot - Normality Check')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_risk_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive risk analysis visualizations.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, plot is displayed.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.company_name} - Risk Analysis Dashboard', 
                     fontsize=16, fontweight='bold')
        
        # 1. Value at Risk (VaR) Analysis
        var_levels = [0.01, 0.05, 0.1, 0.25]
        var_values = [np.percentile(self.valuations_billions, p*100) for p in var_levels]
        
        ax1.bar(range(len(var_levels)), var_values, color=['red', 'orange', 'yellow', 'lightgreen'])
        ax1.set_xlabel('Risk Level')
        ax1.set_ylabel('Value at Risk ($ Billions)')
        ax1.set_title('Value at Risk Analysis')
        ax1.set_xticks(range(len(var_levels)))
        ax1.set_xticklabels([f'{p*100:.0f}%' for p in var_levels])
        
        # Add value labels on bars
        for i, v in enumerate(var_values):
            ax1.text(i, v + 0.5, f'${v:.1f}B', ha='center', va='bottom', fontweight='bold')
        
        ax1.grid(True, alpha=0.3)
        
        # 2. Probability of Different Valuation Ranges
        ranges = [
            (0, 10, "< $10B"),
            (10, 20, "$10B-$20B"),
            (20, 30, "$20B-$30B"),
            (30, 40, "$30B-$40B"),
            (40, float('inf'), "> $40B")
        ]
        
        probabilities = []
        labels = []
        
        for low, high, label in ranges:
            if high == float('inf'):
                prob = (self.valuations_billions >= low).sum() / len(self.valuations_billions) * 100
            else:
                prob = ((self.valuations_billions >= low) & (self.valuations_billions < high)).sum() / len(self.valuations_billions) * 100
            probabilities.append(prob)
            labels.append(label)
        
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        wedges, texts, autotexts = ax2.pie(probabilities, labels=labels, colors=colors, autopct='%1.1f%%')
        ax2.set_title('Probability Distribution by Valuation Range')
        
        # 3. Downside Risk Analysis
        mean_val = self.mc_results['mean'] / 1000
        downside_thresholds = np.arange(0, mean_val, mean_val/10)
        downside_probs = [(self.valuations_billions < threshold).sum() / len(self.valuations_billions) * 100 
                         for threshold in downside_thresholds]
        
        ax3.plot(downside_thresholds, downside_probs, 'ro-', linewidth=2, markersize=6)
        ax3.set_xlabel('Valuation Threshold ($ Billions)')
        ax3.set_ylabel('Probability of Downside (%)')
        ax3.set_title('Downside Risk Curve')
        ax3.grid(True, alpha=0.3)
        
        # 4. Confidence Intervals Visualization
        confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
        intervals = []
        
        for conf in confidence_levels:
            alpha = 1 - conf
            lower = np.percentile(self.valuations_billions, (alpha/2) * 100)
            upper = np.percentile(self.valuations_billions, (1 - alpha/2) * 100)
            intervals.append((lower, upper))
        
        y_pos = np.arange(len(confidence_levels))
        
        for i, (lower, upper) in enumerate(intervals):
            ax4.barh(y_pos[i], upper - lower, left=lower, height=0.6, 
                    alpha=0.7, color=plt.get_cmap('viridis')(i/len(confidence_levels)))
            ax4.text(lower + (upper - lower)/2, y_pos[i], f'${lower:.1f}B - ${upper:.1f}B', 
                    ha='center', va='center', fontweight='bold', color='white')
        
        ax4.set_xlabel('Valuation ($ Billions)')
        ax4.set_ylabel('Confidence Level')
        ax4.set_title('Confidence Intervals')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f'{conf*100:.0f}%' for conf in confidence_levels])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Risk analysis plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_sensitivity_analysis(self, base_cash_flows: List[float], save_path: Optional[str] = None) -> None:
        """
        Create sensitivity analysis for key DCF parameters.
        
        Parameters:
        -----------
        base_cash_flows : List[float]
            Base cash flows for sensitivity analysis
        save_path : str, optional
            Path to save the plot. If None, plot is displayed.
        """
        from monte_carlo_dcf import calculate_dcf  # Import our DCF function
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.company_name} - Sensitivity Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Sensitivity to Discount Rate
        discount_rates = np.linspace(0.08, 0.20, 25)
        base_terminal_growth = 0.03
        
        valuations_dr = []
        for dr in discount_rates:
            try:
                val = calculate_dcf(base_cash_flows, dr, base_terminal_growth)
                valuations_dr.append(val['total_valuation'] / 1000)
            except:
                valuations_dr.append(np.nan)
        
        ax1.plot(discount_rates * 100, valuations_dr, 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Discount Rate (%)')
        ax1.set_ylabel('Valuation ($ Billions)')
        ax1.set_title('Sensitivity to Discount Rate')
        ax1.grid(True, alpha=0.3)
        
        # 2. Sensitivity to Terminal Growth Rate
        terminal_growth_rates = np.linspace(0.01, 0.06, 25)
        base_discount_rate = 0.12
        
        valuations_tg = []
        for tg in terminal_growth_rates:
            try:
                val = calculate_dcf(base_cash_flows, base_discount_rate, tg)
                valuations_tg.append(val['total_valuation'] / 1000)
            except:
                valuations_tg.append(np.nan)
        
        ax2.plot(terminal_growth_rates * 100, valuations_tg, 'r-', linewidth=2, marker='o')
        ax2.set_xlabel('Terminal Growth Rate (%)')
        ax2.set_ylabel('Valuation ($ Billions)')
        ax2.set_title('Sensitivity to Terminal Growth Rate')
        ax2.grid(True, alpha=0.3)
        
        # 3. Two-way sensitivity heatmap (Discount Rate vs Terminal Growth)
        dr_range = np.linspace(0.08, 0.18, 10)
        tg_range = np.linspace(0.01, 0.05, 10)
        
        sensitivity_matrix = np.zeros((len(dr_range), len(tg_range)))
        
        for i, dr in enumerate(dr_range):
            for j, tg in enumerate(tg_range):
                try:
                    val = calculate_dcf(base_cash_flows, dr, tg)
                    sensitivity_matrix[i, j] = val['total_valuation'] / 1000
                except:
                    sensitivity_matrix[i, j] = np.nan
        
        im = ax3.imshow(sensitivity_matrix, cmap='RdYlGn', aspect='auto')
        ax3.set_xlabel('Terminal Growth Rate (%)')
        ax3.set_ylabel('Discount Rate (%)')
        ax3.set_title('Two-way Sensitivity Heatmap')
        
        # Set tick labels
        ax3.set_xticks(range(len(tg_range)))
        ax3.set_xticklabels([f'{tg*100:.1f}' for tg in tg_range])
        ax3.set_yticks(range(len(dr_range)))
        ax3.set_yticklabels([f'{dr*100:.1f}' for dr in dr_range])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Valuation ($ Billions)')
        
        # 4. Cash Flow Growth Sensitivity
        growth_multipliers = np.linspace(0.7, 1.3, 25)
        
        valuations_cf = []
        for mult in growth_multipliers:
            adjusted_cf = [cf * mult for cf in base_cash_flows]
            try:
                val = calculate_dcf(adjusted_cf, base_discount_rate, base_terminal_growth)
                valuations_cf.append(val['total_valuation'] / 1000)
            except:
                valuations_cf.append(np.nan)
        
        ax4.plot(growth_multipliers * 100, valuations_cf, 'g-', linewidth=2, marker='o')
        ax4.set_xlabel('Cash Flow Growth Multiplier (%)')
        ax4.set_ylabel('Valuation ($ Billions)')
        ax4.set_title('Sensitivity to Cash Flow Growth')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sensitivity analysis plot saved to {save_path}")
        else:
            plt.show()
    
    def create_executive_summary(self, save_path: Optional[str] = None) -> None:
        """
        Create a professional executive summary dashboard.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, plot is displayed.
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create a complex layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle(f'{self.company_name} - Monte Carlo DCF Executive Summary', 
                     fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Key Statistics (Top left)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis('off')
        
        stats_text = f"""
        KEY VALUATION STATISTICS
        
        Mean Valuation:           ${self.mc_results['mean']/1000:.1f}B
        Median Valuation:         ${self.mc_results['median']/1000:.1f}B
        Standard Deviation:       ${self.mc_results['std_dev']/1000:.1f}B
        
        Range:                    ${self.mc_results['min']/1000:.1f}B - ${self.mc_results['max']/1000:.1f}B
        
        90% Confidence Interval:  ${self.mc_results['percentiles']['5th']/1000:.1f}B - ${self.mc_results['percentiles']['95th']/1000:.1f}B
        50% Confidence Interval:  ${self.mc_results['percentiles']['25th']/1000:.1f}B - ${self.mc_results['percentiles']['75th']/1000:.1f}B
        
        Number of Simulations:    {self.mc_results['num_simulations']:,}
        Success Rate:             {(self.mc_results['num_simulations']/(self.mc_results['num_simulations']+self.mc_results['failed_simulations'])*100):.1f}%
        """
        
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 2. Distribution histogram (Top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.hist(self.valuations_billions, bins=40, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black')
        ax2.axvline(self.mc_results['mean']/1000, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Valuation ($ Billions)')
        ax2.set_ylabel('Density')
        ax2.set_title('Valuation Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Risk analysis pie chart (Middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        ranges = [(0, 10, "< $10B"), (10, 20, "$10B-$20B"), (20, 30, "$20B-$30B"), (30, float('inf'), "> $30B")]
        probabilities = []
        labels = []
        
        for low, high, label in ranges:
            if high == float('inf'):
                prob = (self.valuations_billions >= low).sum() / len(self.valuations_billions) * 100
            else:
                prob = ((self.valuations_billions >= low) & (self.valuations_billions < high)).sum() / len(self.valuations_billions) * 100
            probabilities.append(prob)
            labels.append(label)
        
        colors = ['red', 'orange', 'yellow', 'green']
        ax3.pie(probabilities, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Risk Distribution')
        
        # 4. Confidence intervals (Middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        confidence_levels = [0.5, 0.8, 0.9, 0.95]
        intervals = []
        
        for conf in confidence_levels:
            alpha = 1 - conf
            lower = np.percentile(self.valuations_billions, (alpha/2) * 100)
            upper = np.percentile(self.valuations_billions, (1 - alpha/2) * 100)
            intervals.append((lower, upper))
        
        y_pos = np.arange(len(confidence_levels))
        
        for i, (lower, upper) in enumerate(intervals):
            ax4.barh(y_pos[i], upper - lower, left=lower, height=0.6, 
                    alpha=0.7, color=plt.get_cmap('viridis')(i/len(confidence_levels)))
        
        ax4.set_xlabel('Valuation ($ Billions)')
        ax4.set_ylabel('Confidence Level')
        ax4.set_title('Confidence Intervals')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f'{conf*100:.0f}%' for conf in confidence_levels])
        ax4.grid(True, alpha=0.3)
        
        # 5. Investment recommendation (Bottom)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Generate investment recommendation based on analysis
        mean_val = self.mc_results['mean'] / 1000
        std_val = self.mc_results['std_dev'] / 1000
        cv = std_val / mean_val  # Coefficient of variation
        
        if cv < 0.3:
            risk_level = "LOW"
            recommendation = "STRONG BUY"
            color = "green"
        elif cv < 0.6:
            risk_level = "MODERATE"
            recommendation = "BUY"
            color = "orange"
        else:
            risk_level = "HIGH"
            recommendation = "HOLD/CAUTIOUS"
            color = "red"
        
        downside_risk = (self.valuations_billions < mean_val * 0.8).sum() / len(self.valuations_billions) * 100
        upside_potential = (self.valuations_billions > mean_val * 1.2).sum() / len(self.valuations_billions) * 100
        
        recommendation_text = f"""
        INVESTMENT RECOMMENDATION: {recommendation}
        
        Risk Level: {risk_level} (Coefficient of Variation: {cv:.2f})
        Downside Risk (>20% loss): {downside_risk:.1f}%
        Upside Potential (>20% gain): {upside_potential:.1f}%
        
        Key Insights:
        • Monte Carlo simulation based on {self.mc_results['num_simulations']:,} scenarios
        • 90% confidence that valuation will be between ${self.mc_results['percentiles']['5th']/1000:.1f}B and ${self.mc_results['percentiles']['95th']/1000:.1f}B
        • Expected return varies significantly due to market uncertainties
        • Consider position sizing based on risk tolerance
        """
        
        ax5.text(0.5, 0.5, recommendation_text, transform=ax5.transAxes, fontsize=14,
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(boxstyle="round,pad=1", facecolor=color, alpha=0.2))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Executive summary saved to {save_path}")
        else:
            plt.show()
    
    def generate_all_reports(self, base_cash_flows: List[float], output_dir: str = "dcf_reports") -> None:
        """
        Generate all visualization reports and save them to a directory.
        
        Parameters:
        -----------
        base_cash_flows : List[float]
            Base cash flows for analysis
        output_dir : str
            Directory to save all reports
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        company_clean = self.company_name.replace(" ", "_").lower()
        
        print(f"Generating comprehensive DCF reports for {self.company_name}...")
        print(f"Output directory: {output_dir}")
        print("-" * 50)
        
        # Generate all reports
        self.plot_valuation_distribution(f"{output_dir}/{company_clean}_distribution_analysis.png")
        self.plot_risk_analysis(f"{output_dir}/{company_clean}_risk_analysis.png")
        self.plot_sensitivity_analysis(base_cash_flows, f"{output_dir}/{company_clean}_sensitivity_analysis.png")
        self.create_executive_summary(f"{output_dir}/{company_clean}_executive_summary.png")
        
        print("-" * 50)
        print("All reports generated successfully!")
        print(f"Check the '{output_dir}' directory for all visualization files.")

# =============================================================================
# CONVENIENCE FUNCTIONS FOR QUICK ANALYSIS
# =============================================================================

def quick_visualize(mc_results: Dict, company_name: str = "Company") -> None:
    """
    Quick visualization function for immediate analysis.
    
    Parameters:
    -----------
    mc_results : Dict
        Results from monte_carlo_dcf() function
    company_name : str
        Name of the company being analyzed
    """
    visualizer = DCFVisualizer(mc_results, company_name)
    visualizer.plot_valuation_distribution()

def create_risk_dashboard(mc_results: Dict, company_name: str = "Company") -> None:
    """
    Create a comprehensive risk analysis dashboard.
    
    Parameters:
    -----------
    mc_results : Dict
        Results from monte_carlo_dcf() function
    company_name : str
        Name of the company being analyzed
    """
    visualizer = DCFVisualizer(mc_results, company_name)
    visualizer.plot_risk_analysis()

def full_analysis_suite(mc_results: Dict, base_cash_flows: List[float], 
                       company_name: str = "Company") -> None:
    """
    Run the complete analysis suite with all visualizations.
    
    Parameters:
    -----------
    mc_results : Dict
        Results from monte_carlo_dcf() function
    base_cash_flows : List[float]
        Base cash flows for sensitivity analysis
    company_name : str
        Name of the company being analyzed
    """
    visualizer = DCFVisualizer(mc_results, company_name)
    
    print(f"Running full analysis suite for {company_name}...")
    print("=" * 60)
    
    # Show all visualizations
    visualizer.plot_valuation_distribution()
    visualizer.plot_risk_analysis()
    visualizer.plot_sensitivity_analysis(base_cash_flows)
    visualizer.create_executive_summary()
    
    print("Full analysis suite completed!")
