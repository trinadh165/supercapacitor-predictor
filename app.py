# ==============================================================================
# CAPSTONE PROJECT: INTERACTIVE SUPERAPACITOR PREDICTOR WEB APP (CORRECTED)
# ==============================================================================

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
    This function is cached to run only once.
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

    # --- Generate the Large, Smooth Dataset (CORRECTED FUNCTION) ---
    def generate_large_dataset(df):
        new_data = []
        grouping_cols = ['Electrode_Material', 'Electrolyte_Type', 'Device_Type', 'Current_Density_Ag-1']
        # Define the columns that contain the actual values
        value_cols = ['Cycles_Completed', 'Charge_Capacity_mAh_g-1', 'Discharge_Capacity_mAh_g-1']
        
        grouped = df.groupby(grouping_cols)
        
        for name, group in grouped:
            start_row = group.loc[group['Cycles_Completed'].idxmin()]
            name_dict = dict(zip(grouping_cols, name))
            
            if len(group) > 1:
                end_row = group.loc[group['Cycles_Completed'].idxmax()]
                max_cycles, start_charge, end_charge, start_discharge, end_discharge = (
                    end_row['Cycles_Completed'], start_row['Charge_Capacity_mAh_g-1'], end_row['Charge_Capacity_mAh_g-1'],
                    start_row['Discharge_Capacity_mAh_g-1'], end_row['Discharge_Capacity_mAh_g-1']
                )
                charge_drop, discharge_drop = start_charge - end_charge, start_discharge - end_discharge
                for cycles in range(0, int(max_cycles) + 1, 250):
                    cycle_ratio = cycles / max_cycles if max_cycles > 0 else 0
                    charge = start_charge - charge_drop * (cycle_ratio ** 0.9)
                    discharge = start_discharge - discharge_drop * (cycle_ratio ** 0.9)
                    new_data.append({**name_dict, 'Cycles_Completed': cycles, 'Charge_Capacity_mAh_g-1': charge, 'Discharge_Capacity_mAh_g-1': discharge})
            else:
                # ### CORRECTION IS HERE ###
                # Instead of dropping columns, we explicitly select the value columns we want.
                # This is safer and avoids the KeyError.
                values_dict = start_row[value_cols].to_dict()
                row_data = {**name_dict, **values_dict}
                new_data.append(row_data)
                
        return pd.DataFrame(new_data)
    
    df_large = generate_large_dataset(df_small)
    
    df_processed = pd.get_dummies(df_large, columns=['Electrode_Material', 'Electrolyte_Type', 'Device_Type'])
    features_cols = df_processed.drop(columns=['Charge_Capacity_mAh_g-1', 'Discharge_Capacity_mAh_g-1']).columns
    y_charge = df_processed['Charge_Capacity_mAh_g-1']
    y_discharge = df_processed['Discharge_Capacity_mAh_g-1']
    
    charge_model = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(df_processed[features_cols], y_charge)
    discharge_model = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(df_processed[features_cols], y_discharge)
    
    return charge_model, discharge_model, features_cols

# --- Load the models (this will only run once) ---
charge_model_xgb, discharge_model_xgb, feature_columns = load_and_train_models()

# --- WEB APPLICATION INTERFACE ---
st.set_page_config(layout="wide")
st.title("🔋 Supercapacitor Performance Predictor")
st.markdown("A Capstone Project to predict supercapacitor degradation using Machine Learning. Select parameters from the sidebar to generate a prediction.")

# --- SIDEBAR FOR USER INPUTS ---
st.sidebar.header("Input Parameters")
material_options = ['CuO/MnO2@MWCNT', 'CuO/CoO@MWCNT', 'CuO@MWCNT', 'CuO']
plot_material = st.sidebar.selectbox("1. Select Electrode Material", material_options)
electrolyte_options = ['RAE', 'KOH']
plot_electrolyte = st.sidebar.selectbox("2. Select Electrolyte Type", electrolyte_options)
device_options = ['Coin Cell', 'Assembled_SC']
plot_device = st.sidebar.selectbox("3. Select Device Type", device_options)
plot_current_density = st.sidebar.number_input("4. Enter Current Density (A/g)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
unit_choice = st.sidebar.radio("5. Select Output Units", ('mAh/g', 'C/g'))
output_format = st.sidebar.radio("6. Select Output Format", ('Simple Prediction', 'Graph', 'Tabular Data'))

# --- MAIN PANEL FOR DISPLAYING OUTPUTS ---
def predict_capacity(material, electrolyte, device, current_density, cycles):
    input_data = pd.DataFrame({'Current_Density_Ag-1': [current_density], 'Cycles_Completed': [cycles], 'Electrode_Material': [material], 'Electrolyte_Type': [electrolyte], 'Device_Type': [device]})
    input_encoded = pd.get_dummies(input_data)
    final_input = input_encoded.reindex(columns=feature_columns, fill_value=0)
    charge = charge_model_xgb.predict(final_input)[0]
    discharge = discharge_model_xgb.predict(final_input)[0]
    return float(charge), float(discharge)

# --- Logic to Display Different Outputs Based on User Choice ---
if output_format == 'Simple Prediction':
    st.subheader("Simple Prediction for a Single Point")
    selected_cycles = st.slider("Select Number of Cycles to Predict", 0, 10000, 5000, 500)
    charge_pred, discharge_pred = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, selected_cycles)
    if unit_choice == 'C/g':
        charge_pred *= 3.6
        discharge_pred *= 3.6
    col1, col2 = st.columns(2)
    col1.metric("Predicted Charge Capacity", f"{charge_pred:.2f} {unit_choice}")
    col2.metric("Predicted Discharge Capacity", f"{discharge_pred:.2f} {unit_choice}")

elif output_format == 'Graph':
    st.subheader("Predictive Degradation Graph")
    cycles_to_plot = list(range(0, 10001, 500))
    charges, discharges = [], []
    for cycle in cycles_to_plot:
        charge, discharge = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, cycle)
        charges.append(charge)
        discharges.append(discharge)
    df_plot = pd.DataFrame({'Cycles': cycles_to_plot, 'Charge Capacity': charges, 'Discharge Capacity': discharges})
    if unit_choice == 'C/g':
        df_plot['Charge Capacity'] *= 3.6
        df_plot['Discharge Capacity'] *= 3.6
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_plot['Cycles'], df_plot['Charge Capacity'], marker='o', linestyle='-', markersize=4, label='Predicted Charge Capacity')
    ax.plot(df_plot['Cycles'], df_plot['Discharge Capacity'], marker='s', linestyle='--', markersize=4, label='Predicted Discharge Capacity')
    ax.set_title(f'Prediction for {plot_material} ({plot_electrolyte})', fontsize=16)
    ax.set_xlabel('Number of Cycles Completed', fontsize=12)
    ax.set_ylabel(f'Capacity ({unit_choice})', fontsize=12)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

elif output_format == 'Tabular Data':
    st.subheader("Predictive Degradation Data Table")
    cycles_to_plot = list(range(0, 10001, 500))
    table_data = []
    for cycle in cycles_to_plot:
        charge, discharge = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, cycle)
        table_data.append({'Cycles': cycle, 'Charge Capacity': charge, 'Discharge Capacity': discharge})
    df_table = pd.DataFrame(table_data)
    if unit_choice == 'C/g':
        df_table['Charge Capacity'] *= 3.6
        df_table['Discharge Capacity'] *= 3.6
    st.dataframe(df_table.style.format({'Charge Capacity': '{:.2f}', 'Discharge Capacity': '{:.2f}'}))
