import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns # type: ignore
from typing import List, Dict, Optional, Tuple
from scipy import stats


# Set style for seaborn for better visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")




def calculate_dcf(cash_flows, discount_rate,terminal_growth_rate):
    """
    Calculate company's intrinsic value using Discounted Cash Flow (DCF) method.

    Parameters:
    cash_flows: List of projected cash flows for explicit forecast period.
    discount_rate: required rate of return (as a decimal).  
    terminal_growth_rate: Growth rate after forecast period (as a decimal). 

    Returns:
    dict: containing present value of cash flows,and total valuation.

    """
    # Validate inputs
    if discount_rate <= terminal_growth_rate:
        raise ValueError(f"Discount rate ({discount_rate:.3f}) must be greater than terminal growth rate ({terminal_growth_rate:.3f})")
                         
     # Calculate present value of explicit forecast cash flows
    present_values = []
    for year, cf in enumerate(cash_flows, 1):
        pv = cf / (1 + discount_rate) ** year
        present_values.append(pv)
    
    # Calculate terminal value
    terminal_cf = cash_flows[-1] * (1 + terminal_growth_rate)
    terminal_value = terminal_cf / (discount_rate - terminal_growth_rate)
    
    # Present value of terminal value
    terminal_pv = terminal_value / (1 + discount_rate) ** len(cash_flows)
    
    # Total valuation
    total_valuation = sum(present_values) + terminal_pv
    
    return {
        'explicit_period_pv': present_values,
        'terminal_value': terminal_value,
        'terminal_pv': terminal_pv,
        'total_valuation': total_valuation,
        'explicit_period_total': sum(present_values)
    } 



"""
# Example usage
print("===  DCF Valuation Calculator  ===")
print()

# Sportify's projected cash flows for the next 5 years (in millions)
spotify_cf = [1000, 1200, 1400, 1600, 1800]  # Example cash flows
discount_rate = 0.12  
terminal_growth_rate = 0.03  # Terminal growth rate (3%)

# Calculate valuation
valuation = calculate_dcf(spotify_cf, discount_rate, terminal_growth_rate)

print("Projected Free Cash Flows ($ millions):")
for i, cf in enumerate(spotify_cf, 1):
    print(f"Year {i}: ${cf:,}")

print(f"\nDiscount Rate: {discount_rate:.1%}")
print(f"Terminal Growth Rate: {terminal_growth_rate:.1%}")
print()

print("VALUATION BREAKDOWN:")
print("-" * 40)
print("Present Value of Explicit Period Cash Flows:")
for i, pv in enumerate(valuation['explicit_period_pv'], 1):
    print(f"Year {i}: ${pv:,.0f} million")

print(f"\nExplicit Period Total: ${valuation['explicit_period_total']:,.0f} million")
print(f"Terminal Value: ${valuation['terminal_value']:,.0f} million")
print(f"Present Value of Terminal Value: ${valuation['terminal_pv']:,.0f} million")
print()
print(f"TOTAL COMPANY VALUATION: ${valuation['total_valuation']:,.0f} million")
print(f"TOTAL COMPANY VALUATION: ${valuation['total_valuation']/1000:.1f} billion")

# Show the impact of different assumptions
print("\n" + "="*50)
print("SENSITIVITY ANALYSIS - HOW ASSUMPTIONS AFFECT VALUATION")
print("="*50)

scenarios = [
    ("Conservative", 0.20, 0.02),
    ("Base Case", 0.17, 0.03),
    ("Optimistic", 0.13, 0.04)
]

print(f"{'Scenario':<12} {'Discount Rate':<13} {'Terminal Growth':<15} {'Valuation ($B)':<15}")
print("-" * 55)

for name, dr, tg in scenarios:
    val = calculate_dcf(spotify_cf, dr, tg)
    print(f"{name:<12} {dr:<13.1%} {tg:<15.1%} {val['total_valuation']/1000:<15.1f}")

print("\nNotice how small changes in assumptions create large valuation differences!")
print("This is exactly why we need Monte Carlo simulation...")

"""




def generate_random_cash_flows(base_cf, growth_params, num_simulations=1):

    """
    Generate random cash flows based on a probability distribution.
    
    parameters:
    base_cf: Base cash flow amount.
    growth_params: dictionary with growth parameters (mean, std_dev).
    num_simulations: Number of cash flow simulations to generate.

    Returns:
    numpy array of simulated cash flows.

    """

    #Generate random growth rates based for each year
    growth_rates = np.random.triangular(
        growth_params['min'],
        growth_params['mode'],
        growth_params['max'],
        size=(num_simulations, len(base_cf))
    )

    # Calculate cash flows for each simulation
    cash_flows = []
    for sim in range(num_simulations):
        cf_scenario = [base_cf[0]]  # Start with the first year's cash flow

        for year in range(1, len(base_cf)):
            # apply growth rate to previous year's cash flow
            next_cf = cf_scenario[-1] * (1 + growth_rates[sim, year-1])
            cf_scenario.append(next_cf)

        cash_flows.append(cf_scenario)
    return np.array(cash_flows)

def monte_carlo_dcf(base_cash_flows, num_simulations=20000):
    """
    Run Monte Carlo simulation for DCF valuation.
    
    Parameters:
    base_cash_flows: starting cash flows for starting year.
    num_simulations: Number of Monte carlo simulations to run.

    Returns:
    dict: containing valuation results from all simulations and statistics.
"""
    # Define probability distribution for uncertain variables
    # These ranges are based on typical tech company volatility

    # Cash flow growth parameters (triangular distribution)
    growth_params = {
        'min': 0.05,  # 5% Minimum growth rate
        'mode': 0.15,  # 15% Most likely growth rate
        'max': 0.35   # 35% Maximum growth rate
    }   

    # Discount rate distribution (normal distribution)
    discount_rate_mean = 0.12  # 12% Mean discount rate
    discount_rate_std = 0.02  # 2% Standard deviation

    # Terminal growth rate distribution (triangular distribution)
    terminal_growth_params = {
        'min': 0.015,  # 1.5% Minimum terminal growth rate
        'mode': 0.03,  # 3% Most likely terminal growth rate
        'max': 0.06   # 6% Maximum terminal growth rate
    }

    print("Running Monte Carlo DCF simulation...")
    print(f"Running {num_simulations} simulations... ")
    print()

    # Storage for results
    valuations = []
    failed_simulations = 0
    failure_reasons = []


    # Run monte carlo simulations

    for sim in range(num_simulations):
        try:
            #Generate random cash flows
            random_cf = generate_random_cash_flows(base_cash_flows,growth_params,1)[0]

            # Generate random discount rate
            discount_rate = max(0.06, min(0.25, np.random.normal(discount_rate_mean, discount_rate_std)))

            # Generate random terminal growth rate
            terminal_growth = np.random.triangular(
                terminal_growth_params['min'],
                terminal_growth_params['mode'],
                terminal_growth_params['max']
            )

            # Ensure terminal growth is less than discount rate
            terminal_growth_rate = min(terminal_growth, discount_rate - 0.02)

            # Addional validation
            if terminal_growth_rate <=0:
                terminal_growth = 0.01

            # Calculat DCF valuation for this scenario
            valuation = calculate_dcf(random_cf, discount_rate, terminal_growth_rate)

            # Sanity check for valid valuation
            if valuation['total_valuation'] > 0 and valuation['total_valuation'] < 1000000:
                valuations.append(valuation['total_valuation'])
            else:
                failed_simulations += 1
                failure_reasons.append(f"Invalid valuation: {valuation['total_valuation']}")
                continue

        except Exception as e:
            failed_simulations += 1
            failure_reasons.append(str(e))
            # skip invalid scenarios
            continue

    

    # Check if we have enough valid simulations
    if len(valuations) < 100:
        print(f"Error: Only {len(valuations)} valid simulations out of {num_simulations}.")
        print(f"failed simulations: {failed_simulations}")
        print("Common failure reasons:")
        for reason in set(failure_reasons[:5]): # show ony first 5 unique reasons
            print(f"- {reason}")
        return None

    # Convert to numpy array for easier analysis
    valuations = np.array(valuations)


    # Calculate statistics
    results = {
        'valuations': valuations,
        'mean': np.mean(valuations),
        'median': np.median(valuations),
        'std_dev': np.std(valuations),
        'min': np.min(valuations),
        'max': np.max(valuations),
        'percentiles': {
            '5th': np.percentile(valuations, 5),
            '25th': np.percentile(valuations, 25),
            '75th': np.percentile(valuations, 75),
            '95th': np.percentile(valuations, 95),
            'num_simulations':len(valuations),
            'failed_simulations': failed_simulations
            
        }

        
     }

    return results

    print("Testing monte carlo dcf valuation...")
    print()

# Spotify base cash flows (in millions)
spotify_base_cf = [1000, 1200, 1400, 1600, 1800]

# Run Monte Carlo simulation
mc_results = monte_carlo_dcf(spotify_base_cf, num_simulations=20000)


# Check if results are valid
if mc_results is None:
    print("Simulation failed. Please check the input parameters.")
    exit(1)

# Display results
print("Monte Carlo DCF Valuation Results:")
print("=" * 50)
print(f"Number of successful simulations: {mc_results['percentiles']['num_simulations']}")
print()

print("VALUATION STATISTICS ($ millions):")
print("-" * 40)
print(f"Mean Valuation:           ${mc_results['mean']:,.0f}")
print(f"Median Valuation:         ${mc_results['median']:,.0f}")
print(f"Standard Deviation:       ${mc_results['std_dev']:,.0f}")
print(f"Minimum Valuation:        ${mc_results['min']:,.0f}")
print(f"Maximum Valuation:        ${mc_results['max']:,.0f}")
print()

print("CONFIDENCE INTERVALS:")
print("-" * 40)
print(f"90% Confidence Interval:  ${mc_results['percentiles']['5th']:,.0f} - ${mc_results['percentiles']['95th']:,.0f}")
print(f"50% Confidence Interval:  ${mc_results['percentiles']['25th']:,.0f} - ${mc_results['percentiles']['75th']:,.0f}")
print()

print("RISK ANALYSIS:")
print("-" * 40)
downside_risk = (mc_results['valuations'] < 10000).sum() / len(mc_results['valuations']) * 100
upside_potential = (mc_results['valuations'] > 25000).sum() / len(mc_results['valuations']) * 100

print(f"Probability of valuation < $10B:  {downside_risk:.1f}%")
print(f"Probability of valuation > $25B:  {upside_potential:.1f}%")

# Convert to billions for easier reading
valuations_billions = mc_results['valuations'] / 1000

print(f"\nIn billions: Mean = ${mc_results['mean']/1000:.1f}B, "
      f"Range = ${mc_results['min']/1000:.1f}B - ${mc_results['max']/1000:.1f}B")

print("\n" + "="*50)
print("COMPARISON: Traditional DCF vs Monte Carlo")
print("="*50)

# Traditional DCF (single point estimate)
traditional_dcf = calculate_dcf(spotify_base_cf, 0.12, 0.03)
print(f"Traditional DCF:     ${traditional_dcf['total_valuation']:,.0f} million (${traditional_dcf['total_valuation']/1000:.1f}B)")
print(f"Monte Carlo Mean:    ${mc_results['mean']:,.0f} million (${mc_results['mean']/1000:.1f}B)")
print(f"Monte Carlo Range:   ${mc_results['min']:,.0f} - ${mc_results['max']:,.0f} million")
print(f"                     (${mc_results['min']/1000:.1f}B - ${mc_results['max']/1000:.1f}B)")

print("Notice how monte carlo simulations provide a range of possible valuations!")




# Example usage

if __name__ == "__main__":
    print("DCF Visualization Moduele")
    print("=" * 50)
    print("This module provides comprehensive visualization tools for Monte Carlo DCF analysis.")
    print("\nExample usage:")
    print("from visualization import DCFVisualizer")
    print("visualizer = DCFVisualizer(mc_results, 'Spotify')")
    print("visualizer.plot_valuation_distribution()")
    print("visualizer.create_executive_summary()")