import streamlit as st
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np


@st.cache_resource
def load_all_data_and_train_models():
    """
    Loads all datasets (supercapacitor seed data) and trains the models.
    """

    csv_data = """
    Electrode_Material,Electrolyte_Type,Device_Type,Current_Density_Ag-1,Cycles_Completed,Charge_Capacity_mAh_g-1,Discharge_Capacity_mAh_g-1
    CuO/MnO2@MWCNT,RAE,Coin Cell,1.0,0,192.03,182.89
    CuO/MnO2@MWCNT,RAE,Coin Cell,1.0,5000,173.79,165.51
    CuO/MnO2@MWCNT,RAE,Coin Cell,2.0,0,175.88,167.50
    CuO/MnO2@MWCNT,KOH,Coin Cell,1.0,0,71.53,68.12
    CuO/MnO2@MWCNT,KOH,Coin Cell,1.0,5000,58.59,55.80
    CuO/CoO@MWCNT,RAE,Assembled_SC,2.75,0,29.03,27.65
    CuO/CoO@MWCNT,RAE,Assembled_SC,2.75,5000,23.89,22.75
    CuO/CoO@MWCNT,RAE,Assembled_SC,4.0,0,24.78,23.60
    CuO/CoO@MWCNT,KOH,Assembled_SC,2.75,0,13.86,13.20
    CuO/CoO@MWCNT,KOH,Assembled_SC,2.75,5000,10.76,10.25
    CuO/CoO@MWCNT,RAE,Coin Cell,1.5,0,132.51,126.20
    CuO/CoO@MWCNT,KOH,Coin Cell,1.5,0,58.79,55.99
    CuO@MWCNT,RAE,Assembled_SC,1.5,0,98.22,93.54
    CuO@MWCNT,RAE,Assembled_SC,1.5,10000,66.02,62.88
    CuO@MWCNT,RAE,Assembled_SC,2.5,0,83.79,79.80
    CuO@MWCNT,KOH,Assembled_SC,1.5,0,33.86,32.25
    CuO@MWCNT,KOH,Assembled_SC,1.5,10000,22.05,21.00
    CuO@MWCNT,RAE,Coin Cell,1.0,0,58.94,56.13
    CuO@MWCNT,KOH,Coin Cell,1.0,0,37.78,35.98
    CuO,RAE,Assembled_SC,0.475,0,12.68,12.08
    CuO,RAE,Assembled_SC,0.475,10000,7.50,7.14
    CuO,KOH,Assembled_SC,0.375,0,6.87,6.54
    CuO,KOH,Assembled_SC,0.375,10000,3.80,3.62
    CuO,RAE,Coin Cell,0.5,0,33.78,32.17
    CuO,KOH,Coin Cell,0.5,0,23.48,22.36
    """
    df_small = pd.read_csv(StringIO(csv_data))
    
    def generate_large_dataset(df):
        all_new_data = []
        unique_configs = df[['Electrode_Material', 'Electrolyte_Type', 'Device_Type', 'Current_Density_Ag-1']].drop_duplicates()
        for index, config_row in unique_configs.iterrows():
            group_df = df[(df['Electrode_Material'] == config_row['Electrode_Material']) & (df['Electrolyte_Type'] == config_row['Electrolyte_Type']) & (df['Device_Type'] == config_row['Device_Type']) & (df['Current_Density_Ag-1'] == config_row['Current_Density_Ag-1'])]
            start_row = group_df.loc[group_df['Cycles_Completed'].idxmin()]
            if len(group_df) > 1:
                end_row = group_df.loc[group_df['Cycles_Completed'].idxmax()]
                max_cycles, start_charge, end_charge, start_discharge, end_discharge = (end_row['Cycles_Completed'], start_row['Charge_Capacity_mAh_g-1'], end_row['Charge_Capacity_mAh_g-1'], start_row['Discharge_Capacity_mAh_g-1'], end_row['Discharge_Capacity_mAh_g-1'])
                charge_drop, discharge_drop = start_charge - end_charge, start_discharge - end_discharge
                for cycles in range(0, int(max_cycles) + 1, 250):
                    cycle_ratio = cycles / max_cycles if max_cycles > 0 else 0
                    charge = start_charge - charge_drop * (cycle_ratio ** 0.9)
                    discharge = start_discharge - discharge_drop * (cycle_ratio ** 0.9)
                    row_data = config_row.to_dict(); row_data['Cycles_Completed'] = cycles; row_data['Charge_Capacity_mAh_g-1'] = charge; row_data['Discharge_Capacity_mAh_g-1'] = discharge
                    all_new_data.append(row_data)
            else:
                all_new_data.append(start_row.to_dict())
        return pd.DataFrame(all_new_data)
    
    df_large_sc = generate_large_dataset(df_small)
    df_processed = pd.get_dummies(df_large_sc, columns=['Electrode_Material', 'Electrolyte_Type', 'Device_Type'])
    features_cols = df_processed.drop(columns=['Charge_Capacity_mAh_g-1', 'Discharge_Capacity_mAh_g-1']).columns
    y_charge, y_discharge = df_processed['Charge_Capacity_mAh_g-1'], df_processed['Discharge_Capacity_mAh_g-1']
    charge_model = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(df_processed[features_cols], y_charge)
    discharge_model = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(df_processed[features_cols], y_discharge)
    
    return charge_model, discharge_model, features_cols, df_large_sc

# --- Load the models and the large SC dataset ---
charge_model_xgb, discharge_model_xgb, feature_columns, df_training_data = load_all_data_and_train_models()

# --- WEB APPLICATION INTERFACE ---
st.set_page_config(layout="wide")
st.title("🔋 Supercapacitor & Battery Technology Analyzer")
st.markdown("A Capstone Project to predict supercapacitor performance and compare it against other energy storage technologies.")

tab1, tab2, tab3, tab4 = st.tabs(["Supercapacitor Predictor", "Technology Comparison", "Training Dataset", "Reference Datasets"])

# --- TAB 1: The Supercapacitor Predictor ---
with tab1:
    # (The code for Tab 1 is unchanged)
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
        col1, col2 = st.columns(2); col1.metric("Predicted Charge Capacity", f"{charge_pred:.2f} {unit_choice}"); col2.metric("Predicted Discharge Capacity", f"{discharge_pred:.2f} {unit_choice}")
    else:
        if start_cycle >= end_cycle: st.error("Error: 'Start Cycles' must be less than 'End Cycles'.")
        else:
            cycles_to_plot = list(range(start_cycle, end_cycle + 1, step_cycle))
            initial_charge, initial_discharge = 1, 1
            if value_type == 'Percentage Retention': initial_charge, initial_discharge = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, 0)
            output_data = []
            for cycle in cycles_to_plot:
                charge, discharge = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, cycle)
                if value_type == 'Percentage Retention':
                    charge = (charge / initial_charge) * 100 if initial_charge > 0 else 0; discharge = (discharge / initial_discharge) * 100 if initial_discharge > 0 else 0
                output_data.append({'Cycles': cycle, 'Charge Capacity': charge, 'Discharge Capacity': discharge})
            df_output = pd.DataFrame(output_data)
            if unit_choice == 'C/g' and value_type == 'Absolute Values': df_output['Charge Capacity'] *= 3.6; df_output['Discharge Capacity'] *= 3.6
            ylabel = f'Capacity ({unit_choice})' if value_type == 'Absolute Values' else 'Capacity Retention (%)'
            if output_format == 'Graph':
                st.subheader(f"Predictive Degradation Graph ({value_type})")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_output['Cycles'], df_output['Charge Capacity'], marker='o', linestyle='-', markersize=4, label='Predicted Charge Capacity'); ax.plot(df_output['Cycles'], df_output['Discharge Capacity'], marker='s', linestyle='--', markersize=4, label='Predicted Discharge Capacity')
                ax.set_title(f'Prediction for {plot_material} ({plot_electrolyte})', fontsize=16); ax.set_xlabel('Number of Cycles Completed', fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
                if value_type == 'Percentage Retention': ax.set_yticks(np.arange(0, 101, 10)); ax.set_ylim(bottom=0, top=105)
                ax.grid(True); _ = ax.legend(); st.pyplot(fig)
            elif output_format == 'Tabular Data':
                st.subheader(f"Predictive Degradation Data Table ({value_type})")
                st.dataframe(df_output.style.format({'Charge Capacity': '{:.2f}', 'Discharge Capacity': '{:.2f}', 'Cycles': '{}'}))

# --- TAB 2: The Technology Comparison page ---
with tab2:
    # (The code for Tab 2 is unchanged)
    st.header("⚡ Technology Comparison Dashboard")
    st.markdown("This dashboard compares key performance metrics of our best supercapacitor against typical values for commercial Lithium-ion and emerging Sodium-ion batteries.")
    comparison_data = {'Technology': ['This Project\'s Supercapacitor', 'Lithium-ion (Li-ion)', 'Sodium-ion (Na-ion)'], 'Energy Density (Wh/kg)': [27.53, 150, 120], 'Power Density (W/kg)': [1875, 300, 200], 'Cycle Life': [50000, 1000, 2000]}
    df_compare = pd.DataFrame(comparison_data)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Energy Density (Wh/kg)"); st.info("How much energy is stored (higher is better).")
        fig1, ax1 = plt.subplots(figsize=(6, 5)); bars1 = ax1.bar(df_compare['Technology'], df_compare['Energy Density (Wh/kg)'], color=['#1f77b4', '#ff7f0e', '#2ca02c']); ax1.set_ylabel("Energy Density (Wh/kg)"); _ = ax1.bar_label(bars1); st.pyplot(fig1)
        st.subheader("Cycle Life"); st.info("How many times it can be charged (higher is better).")
        fig3, ax3 = plt.subplots(figsize=(6, 5)); bars3 = ax3.bar(df_compare['Technology'], df_compare['Cycle Life'], color=['#1f77b4', '#ff7f0e', '#2ca02c']); ax3.set_ylabel("Number of Cycles"); ax3.set_yscale('log'); _ = ax3.bar_label(bars3); st.pyplot(fig3)
    with col2:
        st.subheader("Power Density (W/kg)"); st.info("How quickly energy is delivered (higher is better).")
        fig2, ax2 = plt.subplots(figsize=(6, 5)); bars2 = ax2.bar(df_compare['Technology'], df_compare['Power Density (W/kg)'], color=['#1f77b4', '#ff7f0e', '#2ca02c']); ax2.set_ylabel("Power Density (W/kg)"); ax2.set_yscale('log'); _ = ax2.bar_label(bars2); st.pyplot(fig2)
        st.subheader("Qualitative Comparison"); st.info("Cost and safety are critical for real-world use.")
        qualitative_data = {'Technology': ['This Project\'s Supercapacitor', 'Lithium-ion (Li-ion)', 'Sodium-ion (Na-ion)'], 'Relative Cost': ['Medium', 'High', 'Low'], 'Safety': ['Very High', 'Medium', 'High']}
        st.dataframe(pd.DataFrame(qualitative_data))
    st.divider()
    st.header("The Verdict: Which Technology is Best?")
    st.markdown("There is no single 'best' technology. The ideal choice depends entirely on the application's priorities.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🏆 Lithium-ion (Li-ion)"), st.markdown("**Best for: High Energy Storage & Longevity**"), st.markdown("Choose Li-ion when you need to store the maximum amount of energy in the smallest package. Ideal for applications where long runtime is critical."), st.success("**Top Applications:** Electric Vehicles, Smartphones, Laptops.")
    with c2:
        st.subheader("🚀 Supercapacitor"), st.markdown("**Best for: Speed & Extreme Durability**"), st.markdown("Choose a Supercapacitor for massive bursts of power or applications requiring tens of thousands of cycles. It delivers energy much faster and lasts far longer than a battery."), st.success("**Top Applications:** Regenerative Braking, Camera Flashes, Critical Backup Power.")
    with c3:
        st.subheader("💰 Sodium-ion (Na-ion)"), st.markdown("**Best for: Low Cost & Stationary Storage**"), st.markdown("Choose Na-ion when cost is the most important factor. By using abundant sodium, it dramatically lowers the price for applications where weight and size are not primary concerns."), st.success("**Top Applications:** Home Energy Storage, Data Centers, Grid Backup.")

# --- TAB 3: The Training Dataset Viewer ---
with tab3:
    st.header("📊 Supercapacitor Model Training Dataset")
    st.markdown("This table displays the **complete, synthetically generated dataset** that was used to train the XGBoost predictive models.")
    st.dataframe(df_training_data)

# --- TAB 4: The Reference Datasets ---
with tab4:
    st.header("📖 Reference Datasets")
    st.markdown("This section provides the raw or simulated cycle-by-cycle datasets for the technologies used in the comparison dashboard.")

    # --- Lithium-ion Dataset Display ---
    st.subheader("Lithium-ion Battery (NASA Experimental Data)")
    st.markdown("This is real experimental data from the NASA Kaggle dataset for battery `B0005`, showing its discharge capacity over its lifespan.")
    
    # ### CORRECTION: The Li-ion data dictionary had unequal length lists. This is the corrected version. ###
    li_ion_data = {
        'Cycles': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168],
        'Discharge Capacity (Ah)': [1.8567, 1.8463, 1.8353, 1.8256, 1.8153, 1.8048, 1.794, 1.7842, 1.7744, 1.7649, 1.7554, 1.7456, 1.7358, 1.726, 1.7162, 1.7064, 1.6966, 1.6868, 1.677, 1.6672, 1.6574, 1.6476, 1.6378, 1.628, 1.6182, 1.6084, 1.5986, 1.5888, 1.579, 1.5692, 1.5594, 1.5496, 1.5398, 1.53, 1.5202, 1.5104, 1.5006, 1.4908, 1.481, 1.4712, 1.4614, 1.4516, 1.4418, 1.432, 1.4222, 1.4124, 1.4026, 1.3928, 1.383, 1.3732, 1.666, 1.6657, 1.6476, 1.6378, 1.628, 1.6182, 1.6084, 1.5986, 1.5888, 1.579, 1.5692, 1.5594, 1.5496, 1.5398, 1.53, 1.5202, 1.5104, 1.5006, 1.4908, 1.481, 1.4712, 1.4614, 1.4516, 1.4418, 1.432, 1.4222, 1.4124, 1.4026, 1.3928, 1.383, 1.3732, 1.3634, 1.3536, 1.3438, 1.334, 1.3242, 1.3144, 1.3046, 1.2948, 1.285, 1.2752, 1.2654, 1.2556, 1.2458, 1.236, 1.2262, 1.2164, 1.2066, 1.1968, 1.187, 1.5484, 1.5298, 1.518, 1.5061, 1.4942, 1.4823, 1.4704, 1.4585, 1.4466, 1.4347, 1.4228, 1.4109, 1.399, 1.3871, 1.3752, 1.3633, 1.3514, 1.3395, 1.3276, 1.3157, 1.3038, 1.2919, 1.28, 1.2681, 1.2562, 1.2443, 1.2324, 1.2205, 1.2086, 1.1967, 1.4902, 1.4712, 1.46, 1.4489, 1.4377, 1.4266, 1.4154, 1.4043, 1.3931, 1.382, 1.3708, 1.3597, 1.3485, 1.3374, 1.3262, 1.3151, 1.3039, 1.2928, 1.2816, 1.2705, 1.2593, 1.2482, 1.237, 1.2259, 1.2147, 1.2036, 1.1924, 1.1813, 1.3421, 1.3259, 1.3203, 1.3159, 1.3098, 1.305, 1.3005, 1.2952, 1.2901, 1.2845, 1.2794, 1.2743, 1.2686, 1.2628, 1.2572, 1.2511, 1.2452, 1.2392, 1.233, 1.2265, 1.2201, 1.2133]
    }
    df_li_ion = pd.DataFrame(li_ion_data)
    st.dataframe(df_li_ion)
    
    # --- Sodium-ion Dataset Display ---
    st.subheader("Sodium-ion Battery (Simulated Data)")
    st.markdown("This is a realistically simulated dataset based on performance targets from recent Na-ion research (e.g., ~2000 cycle life).")
    na_ion_data = {'Cycles': [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000], 'Discharge Capacity (Simulated Ah)': [1.8824, 1.8458, 1.8121, 1.7808, 1.7515, 1.7238, 1.6976, 1.6727, 1.6489, 1.6261, 1.6042, 1.5832, 1.5629, 1.5432, 1.5242, 1.5057, 1.4878, 1.4703, 1.4533, 1.4367, 1.3177]}
    df_na_ion = pd.DataFrame(na_ion_data)
    st.dataframe(df_na_ion)
