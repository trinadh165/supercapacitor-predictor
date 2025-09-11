import streamlit as st
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

@st.cache_resource
def load_and_train_models():
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
    return charge_model, discharge_model, features_cols, df_large

# --- Load models and the large dataset ---
charge_model_xgb, discharge_model_xgb, feature_columns, df_training_data = load_and_train_models()

# --- WEB APPLICATION INTERFACE ---
st.set_page_config(layout="wide")
st.title("🔋 Supercapacitor & Battery Technology Analyzer")
st.markdown("A Capstone Project to predict supercapacitor performance and compare it against other energy storage technologies.")

# ### NEW FEATURE: Added a fourth tab ###
tab1, tab2, tab3, tab4 = st.tabs(["Supercapacitor Predictor", "General Comparison", "Training Dataset", "Detailed Comparison"])

# --- TAB 1: The Supercapacitor Predictor ---
with tab1:
    # (Code for Tab 1 is unchanged)
    st.header("Supercapacitor Performance Predictor")
    st.sidebar.header("1. Scenario Parameters")
    material_options = ['CuO/MnO2@MWCNT', 'CuO/CoO@MWCNT', 'CuO@MWCNT', 'CuO']; plot_material = st.sidebar.selectbox("Select Electrode Material", material_options)
    electrolyte_options = ['RAE', 'KOH']; plot_electrolyte = st.sidebar.selectbox("Select Electrolyte Type", electrolyte_options)
    device_options = ['Coin Cell', 'Assembled_SC']; plot_device = st.sidebar.selectbox("Select Device Type", device_options)
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
    
    def predict_capacity(material, electrolyte, device, current_density, cycles):
        input_data = pd.DataFrame({'Current_Density_Ag-1': [current_density], 'Cycles_Completed': [cycles], 'Electrode_Material': [material], 'Electrolyte_Type': [electrolyte], 'Device_Type': [device]})
        input_encoded = pd.get_dummies(input_data)
        final_input = input_encoded.reindex(columns=feature_columns, fill_value=0)
        charge, discharge = charge_model_xgb.predict(final_input)[0], discharge_model_xgb.predict(final_input)[0]
        return float(charge), float(discharge)

    if output_format == 'Simple Prediction':
        st.subheader("Simple Prediction for a Single Point")
        selected_cycles = st.slider("Select Number of Cycles to Predict", 0, 10000, 5000, 500, key="slider_tab1")
        charge_pred, discharge_pred = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, selected_cycles)
        if unit_choice == 'C/g': charge_pred *= 3.6; discharge_pred *= 3.6
        col1, col2 = st.columns(2)
        col1.metric("Predicted Charge Capacity", f"{charge_pred:.2f} {unit_choice}")
        col2.metric("Predicted Discharge Capacity", f"{discharge_pred:.2f} {unit_choice}")
    else:
        if start_cycle >= end_cycle: st.error("Error: 'Start Cycles' must be less than 'End Cycles'.")
        else:
            cycles_to_plot = list(range(start_cycle, end_cycle + 1, step_cycle))
            output_data = []
            if value_type == 'Percentage Retention': initial_charge, initial_discharge = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, 0)
            else: initial_charge, initial_discharge = 1, 1
            for cycle in cycles_to_plot:
                charge, discharge = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, cycle)
                if value_type == 'Percentage Retention':
                    charge = (charge / initial_charge) * 100 if initial_charge > 0 else 0
                    discharge = (discharge / initial_discharge) * 100 if initial_discharge > 0 else 0
                output_data.append({'Cycles': cycle, 'Charge Capacity': charge, 'Discharge Capacity': discharge})
            df_output = pd.DataFrame(output_data)
            if unit_choice == 'C/g' and value_type == 'Absolute Values': df_output['Charge Capacity'] *= 3.6; df_output['Discharge Capacity'] *= 3.6
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
                    ax.set_yticks(np.arange(0, 101, 10))
                    ax.set_ylim(bottom=0, top=105)
                ax.grid(True); _ = ax.legend(); st.pyplot(fig)
            elif output_format == 'Tabular Data':
                st.subheader(f"Predictive Degradation Data Table ({value_type})")
                st.dataframe(df_output.style.format({'Charge Capacity': '{:.2f}', 'Discharge Capacity': '{:.2f}', 'Cycles': '{}'}))

# --- TAB 2: The General Technology Comparison page ---
with tab2:
    # (Code for Tab 2 is unchanged)
    st.header("⚡ General Technology Comparison Dashboard")
    st.markdown("This dashboard compares key performance metrics across the full spectrum of energy storage technologies.")
    comparison_data = {'Technology': ['Conventional Capacitor', 'This Project\'s Supercapacitor', 'Lithium-ion (Li-ion)', 'Sodium-ion (Na-ion)'], 'Energy Density (Wh/kg)': [0.01, 27.53, 150, 120], 'Power Density (W/kg)': [10000, 1875, 300, 200], 'Cycle Life': [1000000, 50000, 1000, 2000]}
    df_compare = pd.DataFrame(comparison_data)
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Energy Density (Wh/kg)"); st.info("How much energy is stored (higher is better).")
        fig1, ax1 = plt.subplots(figsize=(6, 5)); bars1 = ax1.bar(df_compare['Technology'], df_compare['Energy Density (Wh/kg)'], color=colors); ax1.set_ylabel("Energy Density (Wh/kg)"); ax1.set_yscale('log'); _ = ax1.bar_label(bars1); st.pyplot(fig1)
        st.subheader("Cycle Life"); st.info("How many times it can be charged (higher is better).")
        fig3, ax3 = plt.subplots(figsize=(6, 5)); bars3 = ax3.bar(df_compare['Technology'], df_compare['Cycle Life'], color=colors); ax3.set_ylabel("Number of Cycles"); ax3.set_yscale('log'); _ = ax3.bar_label(bars3); st.pyplot(fig3)
    with col2:
        st.subheader("Power Density (W/kg)"); st.info("How quickly energy is delivered (higher is better).")
        fig2, ax2 = plt.subplots(figsize=(6, 5)); bars2 = ax2.bar(df_compare['Technology'], df_compare['Power Density (W/kg)'], color=colors); ax2.set_ylabel("Power Density (W/kg)"); ax2.set_yscale('log'); _ = ax2.bar_label(bars2); st.pyplot(fig2)
        st.subheader("Qualitative Comparison"); st.info("Charge time and safety are critical for real-world use.")
        qualitative_data = {'Technology': ['Conventional Capacitor', 'This Project\'s Supercapacitor', 'Lithium-ion (Li-ion)', 'Sodium-ion (Na-ion)'], 'Charge Time': ['Milliseconds', 'Seconds', 'Hours', 'Hours'], 'Safety': ['Extremely High', 'Very High', 'Medium', 'High']}
        st.dataframe(pd.DataFrame(qualitative_data))
    st.divider()
    st.header("The Verdict: Which Technology is Best?")
    st.markdown("There is no single 'best' technology. The ideal choice depends entirely on the application's priorities.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.subheader("⚡ Capacitor"); st.markdown("**Best for: Instantaneous Power**"); st.success("**Use Case:** Signal filtering, camera flashes.")
    with c2: st.subheader("🚀 Supercapacitor"); st.markdown("**Best for: Speed & Durability**"); st.success("**Use Case:** Regenerative braking, backup power.")
    with c3: st.subheader("🏆 Lithium-ion (Li-ion)"); st.markdown("**Best for: High Energy Storage**"); st.success("**Use Case:** Electric vehicles, smartphones.")
    with c4: st.subheader("💰 Sodium-ion (Na-ion)"); st.markdown("**Best for: Low Cost & Stationary**"); st.success("**Use Case:** Home energy storage, grid backup.")

# --- TAB 3: The Training Dataset Viewer ---
with tab3:
    # (Code for Tab 3 is unchanged)
    st.header("📊 Supercapacitor Model Training Dataset")
    st.markdown("This table displays the **complete, synthetically generated dataset** that was used to train the XGBoost predictive models.")
    st.dataframe(df_training_data)

# ### NEW FEATURE: A fourth tab for Detailed Comparison ###
with tab4:
    st.header("⚙️ Detailed Technology Comparison")
    st.markdown("This dashboard provides a detailed, side-by-side comparison of the key performance metrics for all four energy storage technologies.")

    # --- Data for detailed comparison ---
    detailed_comparison_data = {
        'Metric': [
            "Specific Capacitance (F/g)",
            "Energy Density (Wh/kg)",
            "Power Density (W/kg)",
            "Cycle Life",
            "Charge Time",
            "Energy Storage Mechanism"
        ],
        'Conventional Capacitor': [
            "Very Low (~0.0001)",
            "< 0.1",
            "> 10,000",
            "> 1,000,000",
            "Milliseconds",
            "Physical (Electric Field)"
        ],
        'This Project\'s Supercapacitor': [
            "High (100 - 1,200)",
            "~27.5",
            "~1,875",
            "> 50,000",
            "Seconds to Minutes",
            "Physical & Chemical (EDLC + Faradaic)"
        ],
        'Lithium-ion (Li-ion) Battery': [
            "N/A (Not a capacitor)",
            "~150",
            "~300",
            "~1,000",
            "Hours",
            "Chemical (Intercalation)"
        ],
        'Sodium-ion (Na-ion) Battery': [
            "N/A (Not a capacitor)",
            "~120",
            "~200",
            "~2,000",
            "Hours",
            "Chemical (Intercalation)"
        ]
    }
    df_detailed_compare = pd.DataFrame(detailed_comparison_data)

    st.subheader("Key Performance Metrics")
    st.markdown("This table highlights the fundamental trade-offs between these devices. Notice the orders-of-magnitude differences.")
    st.table(df_detailed_compare.set_index('Metric'))
    
    st.subheader("Core Differences Explained")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("#### Conventional Capacitor")
        st.markdown("- **Strength:** Extremely high power density (instant speed).\n- **Weakness:** Negligible energy storage.")
    with col2:
        st.warning("#### Supercapacitor")
        st.markdown("- **Strength:** Bridges the gap. High power, very long life, and more energy than a capacitor.\n- **Weakness:** Less energy than a battery.")
    with col3:
        st.error("#### Lithium-ion Battery")
        st.markdown("- **Strength:** High energy density (long runtime).\n- **Weakness:** Slower, shorter life, and higher cost/safety concerns.")
    with col4:
        st.success("#### Sodium-ion Battery")
        st.markdown("- **Strength:** Very low cost and high safety.\n- **Weakness:** Lower energy and power than Li-ion.")
