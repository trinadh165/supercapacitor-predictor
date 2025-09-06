

import streamlit as st
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np


@st.cache_resource
def load_and_train_models():
    """
    Loads the seed data, generates a large dataset, and trains the XGBoost models.
    """
    # (The data generation and model training code is unchanged)
    degradation_scenarios = [
        {'config': {'Electrode_Material': 'CuO/MnO2@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 1.0}, 'start_cycles': 0, 'end_cycles': 5000, 'start_charge': 192.03, 'end_charge': 173.79, 'start_discharge': 182.89, 'end_discharge': 165.51},
        {'config': {'Electrode_Material': 'CuO/MnO2@MWCNT', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 1.0}, 'start_cycles': 0, 'end_cycles': 5000, 'start_charge': 71.53, 'end_charge': 58.59, 'start_discharge': 68.12, 'end_discharge': 55.80},
        {'config': {'Electrode_Material': 'CuO/CoO@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 2.75}, 'start_cycles': 0, 'end_cycles': 5000, 'start_charge': 29.03, 'end_charge': 23.89, 'start_discharge': 27.65, 'end_discharge': 22.75},
        {'config': {'Electrode_Material': 'CuO/CoO@MWCNT', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 2.75}, 'start_cycles': 0, 'end_cycles': 5000, 'start_charge': 13.86, 'end_charge': 10.76, 'start_discharge': 13.20, 'end_discharge': 10.25},
        {'config': {'Electrode_Material': 'CuO@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 1.5}, 'start_cycles': 0, 'end_cycles': 10000, 'start_charge': 98.22, 'end_charge': 66.02, 'start_discharge': 93.54, 'end_discharge': 62.88},
        {'config': {'Electrode_Material': 'CuO@MWCNT', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 1.5}, 'start_cycles': 0, 'end_cycles': 10000, 'start_charge': 33.86, 'end_charge': 22.05, 'start_discharge': 32.25, 'end_discharge': 21.00},
        {'config': {'Electrode_Material': 'CuO', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 0.475}, 'start_cycles': 0, 'end_cycles': 10000, 'start_charge': 12.68, 'end_charge': 7.50, 'start_discharge': 12.08, 'end_discharge': 7.14},
        {'config': {'Electrode_Material': 'CuO', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 0.375}, 'start_cycles': 0, 'end_cycles': 10000, 'start_charge': 6.87, 'end_charge': 3.80, 'start_discharge': 6.54, 'end_discharge': 3.62},
    ]
    single_point_scenarios = [
        {'Electrode_Material': 'CuO/MnO2@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 2.0, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 175.88, 'Discharge_Capacity_mAh_g-1': 167.50},
        {'Electrode_Material': 'CuO/CoO@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 4.0, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 24.78, 'Discharge_Capacity_mAh_g-1': 23.60},
        {'Electrode_Material': 'CuO/CoO@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 1.5, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 132.51, 'Discharge_Capacity_mAh_g-1': 126.20},
        {'Electrode_Material': 'CuO/CoO@MWCNT', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 1.5, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 58.79, 'Discharge_Capacity_mAh_g-1': 55.99},
        {'Electrode_Material': 'CuO@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 2.5, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 83.79, 'Discharge_Capacity_mAh_g-1': 79.80},
        {'Electrode_Material': 'CuO@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 1.0, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 58.94, 'Discharge_Capacity_mAh_g-1': 56.13},
        {'Electrode_Material': 'CuO@MWCNT', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 1.0, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 37.78, 'Discharge_Capacity_mAh_g-1': 35.98},
        {'Electrode_Material': 'CuO', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 0.5, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 33.78, 'Discharge_Capacity_mAh_g-1': 32.17},
        {'Electrode_Material': 'CuO', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 0.5, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 23.48, 'Discharge_Capacity_mAh_g-1': 22.36},
    ]
    all_data = []
    for scenario in degradation_scenarios:
        charge_drop, discharge_drop = scenario['start_charge'] - scenario['end_charge'], scenario['start_discharge'] - scenario['end_discharge']
        for cycles in range(0, scenario['end_cycles'] + 1, 250):
            cycle_ratio = cycles / scenario['end_cycles'] if scenario['end_cycles'] > 0 else 0
            charge = scenario['start_charge'] - charge_drop * (cycle_ratio ** 0.9)
            discharge = scenario['start_discharge'] - discharge_drop * (cycle_ratio ** 0.9)
            row_data = scenario['config'].copy()
            row_data['Cycles_Completed'], row_data['Charge_Capacity_mAh_g-1'], row_data['Discharge_Capacity_mAh_g-1'] = cycles, charge, discharge
            all_data.append(row_data)
    all_data.extend(single_point_scenarios)
    df_large = pd.DataFrame(all_data)
    df_processed = pd.get_dummies(df_large, columns=['Electrode_Material', 'Electrolyte_Type', 'Device_Type'])
    features_cols = df_processed.drop(columns=['Charge_Capacity_mAh_g-1', 'Discharge_Capacity_mAh_g-1']).columns
    y_charge, y_discharge = df_processed['Charge_Capacity_mAh_g-1'], df_processed['Discharge_Capacity_mAh_g-1']
    charge_model = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(df_processed[features_cols], y_charge)
    discharge_model = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(df_processed[features_cols], y_discharge)
    return charge_model, discharge_model, features_cols

# --- Load the models ---
charge_model_xgb, discharge_model_xgb, feature_columns = load_and_train_models()

# --- WEB APPLICATION INTERFACE ---
st.set_page_config(layout="wide")
st.title("🔋 Supercapacitor & Battery Technology Analyzer")
st.markdown("A Capstone Project to predict supercapacitor performance and compare it against other energy storage technologies.")

# ### NEW FEATURE: Create a tabbed interface ###
tab1, tab2 = st.tabs(["Supercapacitor Predictor", "Technology Comparison"])

# --- TAB 1: The original Supercapacitor Predictor ---
with tab1:
    st.header("Supercapacitor Performance Predictor")
    
    # --- SIDEBAR FOR USER INPUTS ---
    st.sidebar.header("1. Scenario Parameters")
    material_options = ['CuO/MnO2@MWCNT', 'CuO/CoO@MWCNT', 'CuO@MWCNT', 'CuO']
    plot_material = st.sidebar.selectbox("Select Electrode Material", material_options)
    electrolyte_options = ['RAE', 'KOH']
    plot_electrolyte = st.sidebar.selectbox("Select Electrolyte Type", electrolyte_options)
    device_options = ['Coin Cell', 'Assembled_SC']
    plot_device = st.sidebar.selectbox("Select Device Type", device_options)
    plot_current_density = st.sidebar.number_input("Enter Current Density (A/g)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    st.sidebar.header("2. Output Configuration")
    output_format = st.sidebar.selectbox("Select Output Format", ('Graph', 'Tabular Data', 'Simple Prediction'))
    unit_choice = st.sidebar.radio("Select Output Units", ('mAh/g', 'C/g'))

    if output_format in ['Graph', 'Tabular Data']:
        value_type = st.sidebar.radio("Select Value Type", ('Absolute Values', 'Percentage Retention'))
        st.sidebar.subheader("Define Cycle Range")
        start_cycle = st.sidebar.number_input("Start Cycles", 0, 9500, 0, 500)
        end_cycle = st.sidebar.number_input("End Cycles", 500, 10000, 10000, 500)
        step_cycle = st.sidebar.number_input("Cycle Step (Difference)", 100, 2000, 500, 100)
    
    # --- Prediction function and main logic for Tab 1 ---
    def predict_capacity(material, electrolyte, device, current_density, cycles):
        input_data = pd.DataFrame({'Current_Density_Ag-1': [current_density], 'Cycles_Completed': [cycles], 'Electrode_Material': [material], 'Electrolyte_Type': [electrolyte], 'Device_Type': [device]})
        input_encoded = pd.get_dummies(input_data)
        final_input = input_encoded.reindex(columns=feature_columns, fill_value=0)
        charge = charge_model_xgb.predict(final_input)[0]
        discharge = discharge_model_xgb.predict(final_input)[0]
        return float(charge), float(discharge)

    if output_format == 'Simple Prediction':
        # ... (Simple Prediction logic is unchanged)
        st.subheader("Simple Prediction for a Single Point")
        selected_cycles = st.slider("Select Number of Cycles to Predict", 0, 10000, 5000, 500, key="slider_tab1")
        charge_pred, discharge_pred = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, selected_cycles)
        if unit_choice == 'C/g':
            charge_pred *= 3.6
            discharge_pred *= 3.6
        col1, col2 = st.columns(2)
        col1.metric("Predicted Charge Capacity", f"{charge_pred:.2f} {unit_choice}")
        col2.metric("Predicted Discharge Capacity", f"{discharge_pred:.2f} {unit_choice}")

    else:
        if start_cycle >= end_cycle:
            st.error("Error: 'Start Cycles' must be less than 'End Cycles'.")
        else:
            cycles_to_plot = list(range(start_cycle, end_cycle + 1, step_cycle))
            initial_charge, initial_discharge = 1, 1
            if value_type == 'Percentage Retention':
                initial_charge, initial_discharge = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, 0)
            output_data = []
            for cycle in cycles_to_plot:
                charge, discharge = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, cycle)
                if value_type == 'Percentage Retention':
                    charge = (charge / initial_charge) * 100 if initial_charge > 0 else 0
                    discharge = (discharge / initial_discharge) * 100 if initial_discharge > 0 else 0
                output_data.append({'Cycles': cycle, 'Charge Capacity': charge, 'Discharge Capacity': discharge})
            df_output = pd.DataFrame(output_data)
            if unit_choice == 'C/g' and value_type == 'Absolute Values':
                df_output['Charge Capacity'] *= 3.6
                df_output['Discharge Capacity'] *= 3.6
            ylabel = f'Capacity ({unit_choice})' if value_type == 'Absolute Values' else 'Capacity Retention (%)'

            if output_format == 'Graph':
                st.subheader(f"Predictive Degradation Graph ({value_type})")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_output['Cycles'], df_output['Charge Capacity'], marker='o', linestyle='-', markersize=4, label='Predicted Charge Capacity')
                ax.plot(df_output['Cycles'], df_output['Discharge Capacity'], marker='s', linestyle='--', markersize=4, label='Predicted Discharge Capacity')
                ax.set_title(f'Prediction for {plot_material} ({plot_electrolyte})', fontsize=16)
                ax.set_xlabel('Number of Cycles Completed', fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                if value_type == 'Percentage Retention':
                    ax.set_ylim(bottom=max(0, df_output['Discharge Capacity'].min() - 5), top=105)
                ax.legend(), ax.grid(True), st.pyplot(fig)

            elif output_format == 'Tabular Data':
                st.subheader(f"Predictive Degradation Data Table ({value_type})")
                st.dataframe(df_output.style.format({'Charge Capacity': '{:.2f}', 'Discharge Capacity': '{:.2f}', 'Cycles': '{}'}))

# --- TAB 2: The new Technology Comparison page ---
with tab2:
    st.header("⚡ Technology Comparison Dashboard")
    st.markdown("This dashboard compares the predicted performance of our best supercapacitor against typical values for commercial Lithium-ion and emerging Sodium-ion batteries. The data for Li-ion and Na-ion is based on representative industry and research data.")

    # --- Data for comparison (gathered from general knowledge) ---
    # Representative data for OUR best supercapacitor from the seed data
    # Energy Density = 27.53 Wh/kg, Power Density = 1875 W/kg (from Paper 2)
    # Cycle life is typically > 50,000 for supercapacitors
    comparison_data = {
        'Technology': ['This Project\'s Supercapacitor', 'Lithium-ion (Li-ion) Battery', 'Sodium-ion (Na-ion) Battery'],
        'Energy Density (Wh/kg)': [27.53, 150, 120],
        'Power Density (W/kg)': [1875, 300, 200],
        'Cycle Life': [50000, 1000, 2000],
        'Relative Cost': ['Medium', 'High', 'Low'],
        'Safety': ['Very High', 'Medium', 'High']
    }
    df_compare = pd.DataFrame(comparison_data)

    # --- Ragone Plot ---
    st.subheader("Ragone Plot (Energy vs. Power)")
    st.info("This plot is a standard way to compare energy storage devices. Ideal devices are in the top right.")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, row in df_compare.iterrows():
        ax.loglog(row['Energy Density (Wh/kg)'], row['Power Density (W/kg)'], 'o', markersize=15, label=row['Technology'])
        ax.text(row['Energy Density (Wh/kg)'] * 1.1, row['Power Density (W/kg)'], row['Technology'], fontsize=12)

    ax.set_xlabel('Energy Density (Wh/kg) - How long it lasts', fontsize=12)
    ax.set_ylabel('Power Density (W/kg) - How fast it is', fontsize=12)
    ax.set_title('Ragone Plot of Energy Storage Technologies', fontsize=16)
    ax.grid(True, which="both", ls="--")
    ax.legend()
    st.pyplot(fig)

    # --- Data Table ---
    st.subheader("Comparative Data Table")
    st.dataframe(df_compare)
