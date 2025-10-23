"""
CAT Data Scraper for ReRisk AI
Generates catastrophe-specific insurance data for property risk assessment
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import time
import os

def scrape_cat_insurance_data():
    """Generate catastrophe-specific insurance data for property risk assessment"""
    print("Generating catastrophe insurance data...")
    
    # Generate realistic CAT insurance data
    np.random.seed(42)
    n_samples = 10000
    
    # Property age (years since construction) - affects vulnerability
    property_ages = np.random.normal(25, 15, n_samples)
    property_ages = np.clip(property_ages, 1, 100).astype(int)
    
    # Building type (affects vulnerability to different perils)
    building_types = np.random.choice(['single_family', 'multi_family', 'commercial', 'industrial'], 
                                    n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # Construction quality (0-1, affects damage susceptibility)
    construction_quality = np.random.normal(0.7, 0.2, n_samples)
    construction_quality = np.clip(construction_quality, 0.1, 1.0)
    
    # Number of stories (affects wind damage)
    stories = np.random.poisson(2, n_samples)
    stories = np.clip(stories, 1, 20)
    
    # Distance to coast (miles) - affects hurricane risk
    distance_to_coast = np.random.exponential(50, n_samples)
    distance_to_coast = np.clip(distance_to_coast, 0, 200)
    
    # Elevation (feet above sea level) - affects flood risk
    elevation = np.random.normal(100, 200, n_samples)
    elevation = np.clip(elevation, 0, 2000)
    
    # Regional distribution
    regions = np.random.choice(
        ['northeast', 'southeast', 'northwest', 'southwest', 'south'],
        n_samples,
        p=[0.18, 0.25, 0.12, 0.15, 0.30]
    )
    
    # Calculate CAT insurance charges based on catastrophe risk factors
    charges = []
    for i in range(n_samples):
        base_charge = 5000  # Base CAT insurance premium
        
        # Property age factor (older buildings more vulnerable)
        age_factor = 1 + (property_ages[i] / 100) * 0.5
        
        # Construction quality factor (poorer quality = higher risk)
        quality_factor = 1 + (1 - construction_quality[i]) * 0.8
        
        # Stories factor (taller buildings more vulnerable to wind)
        stories_factor = 1 + (stories[i] / 10) * 0.3
        
        # Distance to coast factor (closer = higher hurricane risk)
        coast_factor = 1 + max(0, (100 - distance_to_coast[i]) / 100) * 0.6
        
        # Elevation factor (lower elevation = higher flood risk)
        elevation_factor = 1 + max(0, (100 - elevation[i]) / 100) * 0.4
        
        # Building type factor
        type_factors = {
            'single_family': 1.0, 'multi_family': 1.2, 
            'commercial': 1.5, 'industrial': 1.8
        }
        type_factor = type_factors[building_types[i]]
        
        # Regional factor (some regions have higher CAT risk)
        regional_factors = {
            'northeast': 1.0, 'southeast': 1.4, 'northwest': 1.1, 
            'southwest': 1.2, 'south': 1.5
        }
        regional_factor = regional_factors[regions[i]]
        
        total_charge = (base_charge * age_factor * quality_factor * stories_factor * 
                       coast_factor * elevation_factor * type_factor * regional_factor)
        charges.append(max(total_charge, 1000))  # Minimum charge
    
    # Create DataFrame
    df = pd.DataFrame({
        'property_age': property_ages,
        'building_type': building_types,
        'construction_quality': construction_quality,
        'stories': stories,
        'distance_to_coast': distance_to_coast,
        'elevation': elevation,
        'region': regions,
        'charges': charges
    })
    
    # Add risk score based on CAT factors
    df['risk_score'] = (
        (df['property_age'] / 100) + 
        (1 - df['construction_quality']) + 
        (df['stories'] / 20) + 
        (1 / (df['distance_to_coast'] + 1)) + 
        (1 / (df['elevation'] + 1))
    ) / 5
    
    # Add time-series features for better realism
    df['policy_year'] = np.random.choice(range(2015, 2025), n_samples)
    df['years_since_inception'] = 2024 - df['policy_year']
    df['inflation_adjusted_charges'] = df['charges'] * (1 + 0.03 * df['years_since_inception'])
    
    # Add seasonal effects
    df['season'] = np.random.choice(['spring', 'summer', 'fall', 'winter'], n_samples)
    seasonal_multipliers = {'spring': 1.0, 'summer': 1.1, 'fall': 1.05, 'winter': 1.15}
    df['seasonal_charges'] = df['charges'] * df['season'].map(seasonal_multipliers)
    
    # Add claims history
    df['prior_claims'] = np.random.poisson(0.3, n_samples)
    df['claims_frequency'] = df['prior_claims'] / (df['years_since_inception'] + 1)
    
    # Add property value proxy
    df['property_value'] = np.random.normal(300000, 150000, n_samples)
    df['property_value'] = np.clip(df['property_value'], 50000, 2000000)
    
    print(f"Generated {len(df)} CAT insurance records with enhanced features")
    return df

def scrape_hurricane_data():
    """Scrape and generate comprehensive hurricane/catastrophe data"""
    print("Scraping hurricane/catastrophe data...")
    
    # Enhanced catastrophe exposure data with multiple perils
    hurricane_data = {
        'region': ['northeast', 'southeast', 'northwest', 'southwest', 'south'],
        'cat_exposure': [1.0, 1.4, 1.1, 1.2, 1.5],  # South has highest exposure
        'hurricane_frequency': [0.1, 0.7, 0.2, 0.3, 0.8],  # Annual frequency
        'avg_intensity': [2.5, 3.6, 2.8, 3.0, 3.8],  # Average category
        'flood_risk': [0.3, 0.8, 0.4, 0.5, 0.9],  # Flood risk factor
        'wind_risk': [0.2, 0.9, 0.3, 0.4, 0.95],  # Wind risk factor
        'earthquake_risk': [0.1, 0.2, 0.8, 0.6, 0.3],  # Earthquake risk (West Coast)
        'seismic_hazard': [0.05, 0.1, 0.9, 0.7, 0.2],  # Seismic hazard score
        'tornado_risk': [0.3, 0.6, 0.1, 0.4, 0.5],  # Tornado risk factor
        'wildfire_risk': [0.1, 0.3, 0.2, 0.8, 0.4],  # Wildfire risk factor
        'storm_surge_risk': [0.2, 0.9, 0.1, 0.3, 0.95]  # Storm surge risk
    }
    
    df_hurricanes = pd.DataFrame(hurricane_data)
    
    print(f"Generated hurricane data for {len(df_hurricanes)} regions")
    return df_hurricanes

def scrape_historical_events():
    """Generate historical catastrophe events data"""
    print("Generating historical claims data...")
    
    # Historical catastrophe events
    events = []
    event_types = ['hurricane', 'earthquake', 'flood', 'tornado', 'wildfire']
    
    for year in range(2015, 2024):
        for event_type in event_types:
            # Generate 1-3 events per type per year
            n_events = np.random.poisson(1.5)
            for _ in range(n_events):
                event = {
                    'year': year,
                    'event_type': event_type,
                    'severity': np.random.uniform(0.1, 1.0),
                    'affected_region': np.random.choice(['northeast', 'southeast', 'northwest', 'southwest', 'south']),
                    'estimated_loss': np.random.lognormal(10, 2) * 1000000,  # Loss in millions
                    'frequency': np.random.uniform(0.1, 1.0)
                }
                events.append(event)
    
    df_events = pd.DataFrame(events)
    
    print(f"Generated {len(df_events)} historical catastrophe events")
    return df_events

def scrape_economic_data():
    """Generate economic indicators that affect insurance pricing"""
    print("Generating economic data...")
    
    # Enhanced economic indicators by region
    economic_data = {
        'region': ['northeast', 'southeast', 'northwest', 'southwest', 'south'],
        'gdp_per_capita': [65000, 45000, 55000, 50000, 40000],  # USD
        'population_density': [400, 150, 50, 80, 200],  # people per sq km
        'property_values': [350000, 200000, 300000, 250000, 180000],  # USD
        'inflation_rate': [2.5, 3.2, 2.8, 3.0, 3.5],  # Annual %
        'unemployment_rate': [4.5, 5.2, 4.8, 5.0, 5.8],  # %
        'construction_costs': [150, 120, 140, 130, 110],  # Cost per sq ft
        'building_codes': [0.9, 0.7, 0.8, 0.75, 0.6],  # Code strength (0-1)
        'infrastructure_age': [45, 35, 40, 38, 30],  # Average age in years
        'disaster_preparedness': [0.8, 0.6, 0.7, 0.65, 0.5]  # Preparedness score
    }
    
    df_economic = pd.DataFrame(economic_data)
    
    print(f"Generated economic data for {len(df_economic)} regions")
    return df_economic

def scrape_earthquake_data():
    """Generate earthquake-specific risk data"""
    print("Generating earthquake risk data...")
    
    # Earthquake risk data based on seismic hazard maps
    earthquake_data = {
        'region': ['northeast', 'southeast', 'northwest', 'southwest', 'south'],
        'seismic_hazard': [0.05, 0.1, 0.9, 0.7, 0.2],  # Peak ground acceleration
        'fault_proximity': [0.1, 0.2, 0.95, 0.8, 0.3],  # Distance to major faults
        'soil_conditions': [0.3, 0.4, 0.8, 0.6, 0.2],  # Soil amplification factor
        'building_vulnerability': [0.2, 0.3, 0.7, 0.5, 0.4],  # Building vulnerability
        'liquefaction_risk': [0.1, 0.2, 0.6, 0.4, 0.3],  # Liquefaction potential
        'tsunami_risk': [0.05, 0.1, 0.8, 0.6, 0.2],  # Tsunami risk factor
        'expected_magnitude': [4.0, 4.5, 7.2, 6.8, 5.0],  # Expected earthquake magnitude
        'return_period': [1000, 500, 50, 100, 200]  # Return period in years
    }
    
    df_earthquake = pd.DataFrame(earthquake_data)
    
    print(f"Generated earthquake data for {len(df_earthquake)} regions")
    return df_earthquake

def main():
    """Main function to generate all CAT data"""
    print("Starting CAT data scraping for ReRisk AI...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate all datasets
    df_insurance = scrape_cat_insurance_data()
    df_hurricanes = scrape_hurricane_data()
    df_events = scrape_historical_events()
    df_economic = scrape_economic_data()
    df_earthquake = scrape_earthquake_data()
    
    # Save datasets
    df_insurance.to_csv('data/insurance2.csv', index=False)
    df_hurricanes.to_csv('data/hurricanes.csv', index=False)
    df_events.to_csv('data/historical_events.csv', index=False)
    df_economic.to_csv('data/economic_indicators.csv', index=False)
    df_earthquake.to_csv('data/earthquake_risk.csv', index=False)
    
    print("\nCAT data scraping completed successfully!")
    print(f"Generated datasets:")
    print(f"   - CAT Insurance data: {len(df_insurance)} records")
    print(f"   - Hurricane data: {len(df_hurricanes)} regions")
    print(f"   - Historical events: {len(df_events)} events")
    print(f"   - Economic data: {len(df_economic)} regions")
    print(f"   - Earthquake data: {len(df_earthquake)} regions")
    
    # Display sample data
    print(f"\nSample CAT insurance data:")
    print(df_insurance.head())
    
    print(f"\nSample hurricane data:")
    print(df_hurricanes.head())

if __name__ == "__main__":
    main()
