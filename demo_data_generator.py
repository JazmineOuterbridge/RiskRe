"""
Comprehensive Demo Data Generator for ReRisk AI
Ensures the platform always has complete, realistic data for all features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_comprehensive_demo_data():
    """Generate complete demo data for all app features"""
    print("Generating comprehensive demo data for ReRisk AI...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate 10,000 property records
    n_samples = 10000
    
    # Property characteristics
    property_ages = np.random.normal(25, 15, n_samples).clip(1, 100)
    building_types = np.random.choice(['single_family', 'multi_family', 'commercial', 'industrial'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    construction_quality = np.random.normal(0.7, 0.2, n_samples).clip(0.1, 1.0)
    stories = np.random.poisson(2, n_samples).clip(1, 20)
    distance_to_coast = np.random.exponential(50, n_samples).clip(0, 200)
    elevation = np.random.normal(100, 200, n_samples).clip(0, 2000)
    
    # Regional distribution with realistic proportions
    regions = np.random.choice(['northeast', 'southeast', 'northwest', 'southwest', 'south'], 
                              n_samples, p=[0.2, 0.3, 0.15, 0.25, 0.1])
    
    # Property values based on region and building type
    base_values = {
        'single_family': 300000,
        'multi_family': 500000,
        'commercial': 800000,
        'industrial': 1200000
    }
    
    regional_multipliers = {
        'northeast': 1.2,
        'southeast': 0.9,
        'northwest': 1.1,
        'southwest': 1.0,
        'south': 0.8
    }
    
    property_values = []
    for i in range(n_samples):
        base_val = base_values[building_types[i]]
        regional_mult = regional_multipliers[regions[i]]
        value = base_val * regional_mult * np.random.normal(1, 0.3, 1)[0]
        property_values.append(max(50000, value))  # Minimum $50k
    
    # Calculate risk score
    risk_scores = (
        (property_ages / 100) + 
        (1 - construction_quality) + 
        (stories / 20) + 
        (1 / (distance_to_coast + 1)) + 
        (1 / (elevation + 1))
    ) / 5
    
    # CAT exposure multipliers by region
    cat_exposure_map = {
        'northeast': 1.0,
        'southeast': 1.5,  # Higher hurricane risk
        'northwest': 1.2,  # Earthquake risk
        'southwest': 1.3,  # Wildfire risk
        'south': 1.4       # Hurricane and flood risk
    }
    
    cat_exposures = [cat_exposure_map[region] for region in regions]
    
    # CAT charges (1-3% of property value based on risk)
    cat_charges = np.array(property_values) * np.array(risk_scores) * np.array(cat_exposures) * np.random.uniform(0.01, 0.03, n_samples)
    
    # Time series features
    policy_years = np.random.randint(2015, 2025, n_samples)
    years_since_inception = 2025 - policy_years
    inflation_adjusted_charges = cat_charges * (1 + (2025 - policy_years) * 0.03)  # 3% annual inflation
    
    # Seasonal effects
    seasons = np.random.choice(['spring', 'summer', 'fall', 'winter'], n_samples)
    seasonal_multipliers = {
        'spring': 1.1,  # Storm season
        'summer': 1.2,  # Hurricane season
        'fall': 1.0,    # Normal
        'winter': 0.9   # Lower risk
    }
    seasonal_charges = cat_charges * [seasonal_multipliers[season] for season in seasons]
    
    # Claims history
    prior_claims = np.random.poisson(0.3, n_samples)  # Average 0.3 claims per policy
    claims_frequency = prior_claims / np.maximum(years_since_inception, 1)
    
    # Create comprehensive DataFrame
    df_insurance = pd.DataFrame({
        'property_age': property_ages,
        'building_type': building_types,
        'construction_quality': construction_quality,
        'stories': stories,
        'distance_to_coast': distance_to_coast,
        'elevation': elevation,
        'region': regions,
        'property_value': property_values,
        'charges': cat_charges,
        'risk_score': risk_scores,
        'cat_exposure': cat_exposures,
        'policy_year': policy_years,
        'years_since_inception': years_since_inception,
        'inflation_adjusted_charges': inflation_adjusted_charges,
        'season': seasons,
        'seasonal_charges': seasonal_charges,
        'prior_claims': prior_claims,
        'claims_frequency': claims_frequency
    })
    
    # Add peril-specific risk factors
    df_insurance['hurricane_risk'] = np.where(
        df_insurance['region'].isin(['southeast', 'south']), 
        np.random.uniform(0.6, 0.95, n_samples),
        np.random.uniform(0.1, 0.4, n_samples)
    )
    
    df_insurance['earthquake_risk'] = np.where(
        df_insurance['region'].isin(['northwest', 'southwest']), 
        np.random.uniform(0.5, 0.9, n_samples),
        np.random.uniform(0.1, 0.3, n_samples)
    )
    
    df_insurance['fire_following_risk'] = np.where(
        df_insurance['region'].isin(['southwest', 'south']), 
        np.random.uniform(0.4, 0.8, n_samples),
        np.random.uniform(0.2, 0.5, n_samples)
    )
    
    df_insurance['scs_risk'] = np.where(
        df_insurance['region'].isin(['southeast', 'south']), 
        np.random.uniform(0.5, 0.9, n_samples),
        np.random.uniform(0.2, 0.6, n_samples)
    )
    
    df_insurance['wildfire_risk'] = np.where(
        df_insurance['region'].isin(['southwest', 'northwest']), 
        np.random.uniform(0.4, 0.8, n_samples),
        np.random.uniform(0.1, 0.4, n_samples)
    )
    
    print(f"Generated {len(df_insurance)} property records")
    return df_insurance

def generate_demo_hurricane_data():
    """Generate demo hurricane/catastrophe data"""
    print("Generating demo hurricane/catastrophe data...")
    
    regions = ['northeast', 'southeast', 'northwest', 'southwest', 'south']
    
    df_hurricanes = pd.DataFrame({
        'region': regions,
        'cat_exposure': [1.0, 1.5, 1.2, 1.3, 1.4],
        'hurricane_frequency': [0.1, 0.7, 0.2, 0.3, 0.8],
        'hurricane_risk': [0.2, 0.9, 0.3, 0.4, 0.95],
        'earthquake_risk': [0.1, 0.2, 0.8, 0.6, 0.3],
        'fire_following_risk': [0.3, 0.6, 0.4, 0.5, 0.7],
        'scs_risk': [0.4, 0.8, 0.2, 0.6, 0.9],
        'wildfire_risk': [0.1, 0.3, 0.2, 0.8, 0.4]
    })
    
    print(f"Generated hurricane data for {len(regions)} regions")
    return df_hurricanes

def generate_demo_historical_events():
    """Generate comprehensive demo historical events"""
    print("Generating demo historical events...")
    
    # Real historical events with realistic data
    events = [
        # 2024
        {"name": "Hurricane Beryl", "year": 2024, "event_type": "hurricane", "region": "southeast", "damage": 1.2, "location": "Texas", "severity": 4.0, "affected_population": 50000},
        {"name": "California Earthquake", "year": 2024, "event_type": "earthquake", "region": "southwest", "damage": 0.8, "location": "California", "severity": 6.2, "affected_population": 25000},
        {"name": "Texas Wildfire", "year": 2024, "event_type": "wildfire", "region": "south", "damage": 0.5, "location": "Texas", "severity": 15000, "affected_population": 10000},
        
        # 2023
        {"name": "Hurricane Idalia", "year": 2023, "event_type": "hurricane", "region": "southeast", "damage": 2.5, "location": "Florida", "severity": 3.0, "affected_population": 75000},
        {"name": "Hilary", "year": 2023, "event_type": "hurricane", "region": "southwest", "damage": 0.8, "location": "California", "severity": 4.0, "affected_population": 30000},
        {"name": "Lee", "year": 2023, "event_type": "hurricane", "region": "northeast", "damage": 1.2, "location": "Nova Scotia", "severity": 5.0, "affected_population": 40000},
        {"name": "Maui Wildfire", "year": 2023, "event_type": "wildfire", "region": "southwest", "damage": 5.5, "location": "Hawaii", "severity": 2100, "affected_population": 15000},
        {"name": "Oklahoma Tornado", "year": 2023, "event_type": "scs", "region": "south", "damage": 0.3, "location": "Oklahoma", "severity": 85, "affected_population": 5000},
        
        # 2022
        {"name": "Hurricane Ian", "year": 2022, "event_type": "hurricane", "region": "southeast", "damage": 112.9, "location": "Florida", "severity": 5.0, "affected_population": 200000},
        {"name": "Fiona", "year": 2022, "event_type": "hurricane", "region": "northeast", "damage": 2.3, "location": "Puerto Rico", "severity": 4.0, "affected_population": 60000},
        {"name": "Kentucky Tornado", "year": 2022, "event_type": "scs", "region": "south", "damage": 3.9, "location": "Kentucky", "severity": 90, "affected_population": 25000},
        {"name": "Dixie Fire", "year": 2022, "event_type": "wildfire", "region": "southwest", "damage": 1.5, "location": "California", "severity": 963000, "affected_population": 50000},
        
        # 2021
        {"name": "Hurricane Ida", "year": 2021, "event_type": "hurricane", "region": "southeast", "damage": 75.0, "location": "Louisiana", "severity": 4.0, "affected_population": 150000},
        {"name": "Texas Freeze", "year": 2021, "event_type": "scs", "region": "south", "damage": 2.0, "location": "Texas", "severity": 50, "affected_population": 100000},
        {"name": "Bootleg Fire", "year": 2021, "event_type": "wildfire", "region": "northwest", "damage": 0.8, "location": "Oregon", "severity": 413000, "affected_population": 30000},
        
        # 2020
        {"name": "Hurricane Laura", "year": 2020, "event_type": "hurricane", "region": "south", "damage": 19.2, "location": "Louisiana", "severity": 4.0, "affected_population": 80000},
        {"name": "Iowa Derecho", "year": 2020, "event_type": "scs", "region": "northwest", "damage": 7.5, "location": "Iowa", "severity": 100, "affected_population": 120000},
        {"name": "August Complex Fire", "year": 2020, "event_type": "wildfire", "region": "southwest", "damage": 0.5, "location": "California", "severity": 1030000, "affected_population": 40000},
        
        # 2019
        {"name": "Hurricane Dorian", "year": 2019, "event_type": "hurricane", "region": "southeast", "damage": 3.4, "location": "Bahamas", "severity": 5.0, "affected_population": 70000},
        {"name": "Alaska Earthquake", "year": 2019, "event_type": "earthquake", "region": "northwest", "damage": 0.1, "location": "Alaska", "severity": 7.1, "affected_population": 5000},
        {"name": "Kincade Fire", "year": 2019, "event_type": "wildfire", "region": "southwest", "damage": 0.1, "location": "California", "severity": 77000, "affected_population": 15000},
        
        # 2018
        {"name": "Hurricane Michael", "year": 2018, "event_type": "hurricane", "region": "southeast", "damage": 25.1, "location": "Florida", "severity": 5.0, "affected_population": 100000},
        {"name": "Florence", "year": 2018, "event_type": "hurricane", "region": "southeast", "damage": 24.2, "location": "North Carolina", "severity": 4.0, "affected_population": 90000},
        {"name": "Camp Fire", "year": 2018, "event_type": "wildfire", "region": "southwest", "damage": 16.5, "location": "California", "severity": 153000, "affected_population": 60000},
        
        # 2017
        {"name": "Hurricane Harvey", "year": 2017, "event_type": "hurricane", "region": "south", "damage": 125.0, "location": "Texas", "severity": 4.0, "affected_population": 300000},
        {"name": "Hurricane Irma", "year": 2017, "event_type": "hurricane", "region": "southeast", "damage": 50.0, "location": "Florida", "severity": 5.0, "affected_population": 200000},
        {"name": "Hurricane Maria", "year": 2017, "event_type": "hurricane", "region": "southeast", "damage": 91.6, "location": "Puerto Rico", "severity": 5.0, "affected_population": 180000},
        {"name": "Tubbs Fire", "year": 2017, "event_type": "wildfire", "region": "southwest", "damage": 1.2, "location": "California", "severity": 36000, "affected_population": 25000},
        
        # 2016
        {"name": "Hurricane Matthew", "year": 2016, "event_type": "hurricane", "region": "southeast", "damage": 10.3, "location": "South Carolina", "severity": 5.0, "affected_population": 50000},
        {"name": "Louisiana Floods", "year": 2016, "event_type": "scs", "region": "south", "damage": 1.0, "location": "Louisiana", "severity": 60, "affected_population": 40000},
        
        # 2015
        {"name": "Hurricane Joaquin", "year": 2015, "event_type": "hurricane", "region": "southeast", "damage": 0.2, "location": "Bahamas", "severity": 4.0, "affected_population": 10000},
        {"name": "Nepal Earthquake", "year": 2015, "event_type": "earthquake", "region": "international", "damage": 10.0, "location": "Nepal", "severity": 7.8, "affected_population": 500000},
    ]
    
    df_events = pd.DataFrame(events)
    print(f"Generated {len(df_events)} historical events")
    return df_events

def main():
    """Generate all demo data files"""
    print("Starting comprehensive demo data generation for ReRisk AI...")
    
    # Generate insurance data
    df_insurance = generate_comprehensive_demo_data()
    df_insurance.to_csv('data/insurance2.csv', index=False)
    print("Generated insurance2.csv")
    
    # Generate hurricane/catastrophe data
    df_hurricanes = generate_demo_hurricane_data()
    df_hurricanes.to_csv('data/hurricanes.csv', index=False)
    print("Generated hurricanes.csv")
    
    # Generate historical events
    df_events = generate_demo_historical_events()
    df_events.to_csv('data/historical_all_events.csv', index=False)
    print("Generated historical_all_events.csv")
    
    # Generate individual peril datasets
    for event_type in ['hurricane', 'earthquake', 'wildfire', 'scs', 'fire_following']:
        type_events = df_events[df_events['event_type'] == event_type]
        if not type_events.empty:
            type_events.to_csv(f'data/historical_{event_type}s.csv', index=False)
            print(f"Generated historical_{event_type}s.csv")
    
    # Generate economic indicators
    regions = ['northeast', 'southeast', 'northwest', 'southwest', 'south']
    economic_data = []
    
    for region in regions:
        economic_data.append({
            'region': region,
            'gdp_growth': np.random.normal(2.5, 1.0),
            'population_density': np.random.normal(200, 100),
            'property_values': np.random.normal(300000, 100000),
            'inflation_rate': np.random.normal(3.0, 0.5),
            'unemployment_rate': np.random.normal(5.0, 1.5),
            'construction_costs': np.random.normal(150, 30),
            'building_codes': np.random.uniform(0.6, 1.0),
            'infrastructure_age': np.random.normal(25, 10),
            'disaster_preparedness': np.random.uniform(0.4, 0.9)
        })
    
    df_economic = pd.DataFrame(economic_data)
    df_economic.to_csv('data/economic_indicators.csv', index=False)
    print("Generated economic_indicators.csv")
    
    # Generate earthquake risk data
    earthquake_data = []
    for region in regions:
        earthquake_data.append({
            'region': region,
            'seismic_hazard': np.random.uniform(0.1, 0.9),
            'fault_proximity': np.random.uniform(0.1, 0.8),
            'soil_conditions': np.random.uniform(0.2, 0.9),
            'building_vulnerability': np.random.uniform(0.3, 0.8),
            'liquefaction_risk': np.random.uniform(0.1, 0.7),
            'tsunami_risk': np.random.uniform(0.0, 0.5),
            'expected_magnitude': np.random.uniform(5.0, 7.5),
            'return_period': np.random.uniform(100, 1000)
        })
    
    df_earthquake = pd.DataFrame(earthquake_data)
    df_earthquake.to_csv('data/earthquake_risk.csv', index=False)
    print("Generated earthquake_risk.csv")
    
    print(f"\nDemo data generation completed!")
    print(f"Generated datasets:")
    print(f"   - Insurance Data: {len(df_insurance)} records")
    print(f"   - Hurricane Data: {len(df_hurricanes)} regions")
    print(f"   - Historical Events: {len(df_events)} events")
    print(f"   - Economic Indicators: {len(df_economic)} regions")
    print(f"   - Earthquake Risk: {len(df_earthquake)} regions")
    print(f"\nAll demo data ensures the platform works with complete analysis!")

if __name__ == "__main__":
    main()
