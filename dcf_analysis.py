import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
try:
    from monte_carlo_dcf import monte_carlo_dcf, calculate_dcf
    from visualization import DCFVisualizer, quick_visualize, create_risk_dashboard, full_analysis_suite
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required files are in the same directory:")
    print("- monte_carlo_dcf.py")
    print("- visualization.py")
    sys.exit(1)

def analyze_spotify():
    """
    Comprehensive analysis of Spotify using Monte Carlo DCF.
    """
    print("=" * 80)
    print("SPOTIFY - MONTE CARLO DCF ANALYSIS")
    print("=" * 80)
    
    # Spotify's projected cash flows (in millions)
    spotify_cf = [1000, 1200, 1400, 1600, 1800]
    
    print("Running Monte Carlo simulation for Spotify...")
    print(f"Base cash flows: {spotify_cf}")
    print("This may take a moment...")
    print()
    
    # Run Monte Carlo simulation
    mc_results = monte_carlo_dcf(spotify_cf, num_simulations=15000)
    
    if mc_results is None:
        print("❌ Monte Carlo simulation failed!")
        return
    
    print("✅ Monte Carlo simulation completed successfully!")
    print()
    
    # Create visualizer
    visualizer = DCFVisualizer(mc_results, "Spotify")
    
    # Generate all analysis
    print("Generating comprehensive analysis...")
    print("-" * 50)
    
    # 1. Distribution Analysis
    print(" Creating distribution analysis...")
    visualizer.plot_valuation_distribution()
    
    # 2. Risk Analysis
    print("  Creating risk analysis dashboard...")
    visualizer.plot_risk_analysis()
    
    # 3. Sensitivity Analysis
    print(" Creating sensitivity analysis...")
    visualizer.plot_sensitivity_analysis(spotify_cf)
    
    # 4. Executive Summary
    print("Creating executive summary...")
    visualizer.create_executive_summary()
    
    # 5. Generate all reports to files
    print(" Generating all reports to files...")
    visualizer.generate_all_reports(spotify_cf, "spotify_dcf_reports")
    
    print("\n" + "=" * 80)
    print("SPOTIFY ANALYSIS COMPLETE!")
    print("=" * 80)
    
    return mc_results

def analyze_netflix():
    """
    Comprehensive analysis of Netflix using Monte Carlo DCF.
    """
    print("=" * 80)
    print("NETFLIX - MONTE CARLO DCF ANALYSIS")
    print("=" * 80)
    
    # Netflix's projected cash flows (in millions) - larger company
    netflix_cf = [2500.0, 3000.0, 3500.0, 4000.0, 4500.0]
    
    print("Running Monte Carlo simulation for Netflix...")
    print(f"Base cash flows: {netflix_cf}")
    print("This may take a moment...")
    print()
    
    # Run Monte Carlo simulation
    mc_results = monte_carlo_dcf(netflix_cf, num_simulations=15000)
    
    if mc_results is None:
        print("❌ Monte Carlo simulation failed!")
        return
    
    print(" Monte Carlo simulation completed successfully!")
    print()
    
    # Create visualizer
    visualizer = DCFVisualizer(mc_results, "Netflix")
    
    # Generate all analysis
    print("Generating comprehensive analysis...")
    print("-" * 50)
    
    # Quick analysis suite
    full_analysis_suite(mc_results, netflix_cf, "Netflix")
    
    # Generate all reports to files
    print(" Generating all reports to files...")
    visualizer.generate_all_reports(netflix_cf, "netflix_dcf_reports")
    
    print("\n" + "=" * 80)
    print("NETFLIX ANALYSIS COMPLETE!")
    print("=" * 80)
    
    return mc_results

def compare_companies():
    """
    Compare multiple companies using Monte Carlo DCF analysis.
    """
    print("=" * 80)
    print("COMPARATIVE ANALYSIS - SPOTIFY vs NETFLIX")
    print("=" * 80)
    
    # Company data
    companies = {
        "Spotify": [1000, 1200, 1400, 1600, 1800],
        "Netflix": [2500, 3000, 3500, 4000, 4500],
        "Apple": [15000, 16000, 17000, 18000, 19000],  # Much larger company
        "Tesla": [3000, 4000, 5000, 6000, 7000]       # High growth company
    }
    
    results = {}
    
    print("Running Monte Carlo simulations for all companies...")
    print("This will take a few minutes...")
    print()
    
    for company, cash_flows in companies.items():
        print(f"Analyzing {company}...")
        mc_results = monte_carlo_dcf(cash_flows, num_simulations=10000)
        
        if mc_results is not None:
            results[company] = mc_results
            print(f" {company} analysis completed")
        else:
            print(f" {company} analysis failed")
        print()
    
    # Create comparison visualization
    if len(results) > 1:
        create_comparison_chart(results)
        create_comparison_table(results)
    
    return results

def create_comparison_chart(results):
    """
    Create a comparison chart for multiple companies.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Company DCF Comparison', fontsize=16, fontweight='bold')
    
    companies = list(results.keys())
    colors = plt.get_cmap('Set3')(np.linspace(0, 1, len(companies)))
    
    # 1. Mean valuations comparison
    mean_vals = [results[company]['mean']/1000 for company in companies]
    ax1.bar(companies, mean_vals, color=colors)
    ax1.set_ylabel('Mean Valuation ($ Billions)')
    ax1.set_title('Mean Valuation Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(mean_vals):
        ax1.text(i, v + max(mean_vals)*0.01, f'${v:.1f}B', ha='center', va='bottom')
    
    # 2. Risk comparison (Standard deviation)
    std_vals = [results[company]['std_dev']/1000 for company in companies]
    ax2.bar(companies, std_vals, color=colors)
    ax2.set_ylabel('Standard Deviation ($ Billions)')
    ax2.set_title('Risk Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Distribution overlays
    for i, company in enumerate(companies):
        valuations = results[company]['valuations'] / 1000
        ax3.hist(valuations, bins=30, alpha=0.5, label=company, color=colors[i])
    
    ax3.set_xlabel('Valuation ($ Billions)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Valuation Distributions')
    ax3.legend()
    
    # 4. Risk-Return scatter plot
    for i, company in enumerate(companies):
        mean_val = results[company]['mean'] / 1000
        std_val = results[company]['std_dev'] / 1000
        ax4.scatter(std_val, mean_val, s=200, alpha=0.7, color=colors[i], label=company)
        ax4.annotate(company, (std_val, mean_val), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
    
    ax4.set_xlabel('Risk (Standard Deviation - $ Billions)')
    ax4.set_ylabel('Expected Return (Mean - $ Billions)')
    ax4.set_title('Risk vs Return')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('company_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comparison_table(results):
    """
    Create a detailed comparison table for multiple companies.
    """
    print("\n" + "=" * 120)
    print("DETAILED COMPARISON TABLE")
    print("=" * 120)
    
    # Table header
    header = f"{'Company':<15} {'Mean ($B)':<12} {'Median ($B)':<12} {'Std Dev ($B)':<12} {'Min ($B)':<10} {'Max ($B)':<10} {'Risk Level':<12}"
    print(header)
    print("-" * 120)
    
    # Sort companies by mean valuation
    sorted_companies = sorted(results.keys(), key=lambda x: results[x]['mean'], reverse=True)
    
    for company in sorted_companies:
        data = results[company]
        mean_val = data['mean'] / 1000
        median_val = data['median'] / 1000
        std_val = data['std_dev'] / 1000
        min_val = data['min'] / 1000
        max_val = data['max'] / 1000
        
        # Calculate risk level
        cv = std_val / mean_val  # Coefficient of variation
        if cv < 0.3:
            risk_level = "LOW"
        elif cv < 0.6:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"
        
        row = f"{company:<15} {mean_val:<12.1f} {median_val:<12.1f} {std_val:<12.1f} {min_val:<10.1f} {max_val:<10.1f} {risk_level:<12}"
        print(row)
    
    print("=" * 120)
    print("\nKey Insights:")
    print("- Risk Level based on Coefficient of Variation (Std Dev / Mean)")
    print("- LOW: CV < 0.3, MODERATE: 0.3 ≤ CV < 0.6, HIGH: CV ≥ 0.6")
    print("- Higher standard deviation indicates more uncertainty in valuation")

def stress_test_analysis():
    """
    Perform stress testing on DCF models under extreme scenarios.
    """
    print("=" * 80)
    print("STRESS TEST ANALYSIS")
    print("=" * 80)
    
    # Base case
    base_cf = [1000, 1200, 1400, 1600, 1800]
    
    # Define stress scenarios
    scenarios = {
        "Base Case": {"cf_multiplier": 1.0, "dr_adjustment": 0.0, "tg_adjustment": 0.0},
        "Recession": {"cf_multiplier": 0.7, "dr_adjustment": 0.03, "tg_adjustment": -0.01},
        "Economic Boom": {"cf_multiplier": 1.3, "dr_adjustment": -0.02, "tg_adjustment": 0.01},
        "Interest Rate Shock": {"cf_multiplier": 1.0, "dr_adjustment": 0.05, "tg_adjustment": 0.0},
        "Pandemic Scenario": {"cf_multiplier": 0.5, "dr_adjustment": 0.04, "tg_adjustment": -0.015}
    }
    
    print("Running stress test scenarios...")
    print("-" * 50)
    
    stress_results = {}
    
    for scenario_name, params in scenarios.items():
        print(f"Testing {scenario_name}...")
        
        # Adjust cash flows
        adjusted_cf = [cf * params["cf_multiplier"] for cf in base_cf]
        
        # For stress testing, we'll use a simpler approach with fixed parameters
        # In a real implementation, you'd modify the monte_carlo_dcf function
        try:
            # Use base DCF with adjusted parameters
            base_dr = 0.12 + params["dr_adjustment"]
            base_tg = 0.03 + params["tg_adjustment"]
            
            # Ensure valid parameters
            base_dr = max(0.05, min(0.25, base_dr))
            base_tg = max(0.01, min(base_dr - 0.02, base_tg))
            
            valuation = calculate_dcf(adjusted_cf, base_dr, base_tg)
            stress_results[scenario_name] = valuation['total_valuation'] / 1000
            
        except Exception as e:
            print(f"Error in {scenario_name}: {e}")
            stress_results[scenario_name] = 0
    
    # Display results
    print("\n" + "=" * 60)
    print("STRESS TEST RESULTS")
    print("=" * 60)
    
    base_valuation = stress_results.get("Base Case", 0)
    
    print(f"{'Scenario':<20} {'Valuation ($B)':<15} {'vs Base Case':<15}")
    print("-" * 60)
    
    for scenario, valuation in stress_results.items():
        if base_valuation > 0:
            change = ((valuation - base_valuation) / base_valuation) * 100
            change_str = f"{change:+.1f}%"
        else:
            change_str = "N/A"
        
        print(f"{scenario:<20} {valuation:<15.1f} {change_str:<15}")
    
    print("=" * 60)
    
    # Create stress test visualization
    create_stress_test_chart(stress_results)

def create_stress_test_chart(stress_results):
    """
    Create visualization for stress test results.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    scenarios = list(stress_results.keys())
    valuations = list(stress_results.values())
    
    # Bar chart
    colors = ['green' if s == 'Base Case' else 'red' if v < stress_results['Base Case'] else 'blue' 
              for s, v in stress_results.items()]
    
    ax1.bar(scenarios, valuations, color=colors)
    ax1.set_ylabel('Valuation ($ Billions)')
    ax1.set_title('Stress Test Results')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(valuations):
        ax1.text(i, v + max(valuations)*0.01, f'${v:.1f}B', ha='center', va='bottom')
    
    # Percentage change from base case
    base_val = stress_results['Base Case']
    changes = [((v - base_val) / base_val) * 100 if base_val > 0 else 0 for v in valuations]
    
    ax2.bar(scenarios, changes, color=colors)
    ax2.set_ylabel('Change from Base Case (%)')
    ax2.set_title('Percentage Change from Base Case')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stress_test_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run comprehensive DCF analysis suite.
    """
    print(" MONTE CARLO DCF ANALYSIS SUITE")
    print("=" * 80)
    print("Welcome to the comprehensive DCF analysis platform!")
    print("This suite will run multiple analyses:")
    print("1. Individual company analysis (Spotify)")
    print("2. Individual company analysis (Netflix)")
    print("3. Comparative analysis")
    print("4. Stress testing")
    print("=" * 80)
    print()
    
    # Ask user what analysis to run
    while True:
        print("Select analysis to run:")
        print("1. Spotify Analysis")
        print("2. Netflix Analysis")
        print("3. Company Comparison")
        print("4. Stress Testing")
        print("5. Run All Analyses")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            analyze_spotify()
        elif choice == '2':
            analyze_netflix()
        elif choice == '3':
            compare_companies()
        elif choice == '4':
            stress_test_analysis()
        elif choice == '5':
            print("Running comprehensive analysis suite...")
            print("This will take several minutes...")
            print()
            
            # Run all analyses
            analyze_spotify()
            analyze_netflix()
            compare_companies()
            stress_test_analysis()
            
            print("\n" + "🎉" * 20)
            print("ALL ANALYSES COMPLETED SUCCESSFULLY!")
            print("🎉" * 20)
            
        elif choice == '6':
            print("Exiting analysis suite. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-6.")
        
        print("\n" + "-" * 50)
        continue_analysis = input("Continue with more analysis? (y/n): ").strip().lower()
        if continue_analysis != 'y':
            break
    
    print("\nThank you for using the Monte Carlo DCF Analysis Suite!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check your input files and try again.")