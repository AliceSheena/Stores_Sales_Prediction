import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# Page Configuration
st.set_page_config(page_title="Big Mart Sales Prediction", page_icon="🛒", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight:bold;
    color:#2C3E50;
}
.metric-font {
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("🛒 Big Mart Sales Prediction")
st.markdown("### Predict Item Sales for Different Stores", unsafe_allow_html=True)
st.markdown("Input the item and outlet details to estimate sales.", unsafe_allow_html=True)





# Load Pre-trained Model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model.pkl' is in the working directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Input Data", "Results", "Insights"])

# Input Section
if menu == "Input Data":
    st.header("📋 Input Store & Item Details")
    
    col1, col2 = st.columns(2)
    with col1:
        item_visibility = st.slider(
        "Item Visibility (%)",
        min_value=0.0,  
        value=0.0,      
        step=0.0001,
        format="%.4f", 
        help="Percentage of total display area allocated to this product"
        )
        item_mrp = st.number_input(
            "Maximum Retail Price (MRP)",
            min_value=0.0,
            value=100.0,
            step=10.0,
            help="List price of the product"
        )

    with col2:
        outlet_size = st.selectbox(
            "Outlet Size",
            options=["Small", "Medium", "High"],
            help="Size of the store in terms of ground area"
        )
        outlet_location = st.selectbox(
            "Outlet Location",
            options=["Tier 1", "Tier 2", "Tier 3"],
            help="Type of city where the store is located"
        )

    # Outlet Type Selection
    outlet_type = st.radio(
        "Outlet Type",
        options=["Grocery Store", "Supermarket Type 1", "Supermarket Type 2", "Supermarket Type 3"],
        horizontal=True
    )

    # Outlet Identifier Selection
    outlet_identifier = st.selectbox(
        "Outlet Identifier",
        options=['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'],
        help="Choose Outlet Identifier"
    )

    # Save Input for Prediction
    input_data = {
        'item_visibility': item_visibility,
        'item_mrp': item_mrp,
        'outlet_size': outlet_size,
        'outlet_location': outlet_location,
        'outlet_type': outlet_type,
        'outlet_identifier': outlet_identifier
    }

    # Proceed to Prediction
    if st.button("Submit Details"):
        st.session_state['input_data'] = input_data
        st.session_state['submitted'] = True
        st.success("Data Submitted Successfully! View the results from the 'Results' page.")





def safe_dataframe_conversion(data):
    """
    Safely convert input to a DataFrame with appropriate handling.
    
    Parameters:
    - data: Input data (dict, list, numpy array, or DataFrame)
    
    Returns:
    - pandas DataFrame
    """
    # If already a DataFrame, return as-is
    if isinstance(data, pd.DataFrame):
        return data
    
    # If dictionary
    if isinstance(data, dict):
        # Handle dictionary with scalar values
        if all(np.isscalar(v) for v in data.values()):
            return pd.DataFrame([data])
        
        # Handle dictionary with list/array values
        return pd.DataFrame(data)
    
    # If list or numpy array
    if isinstance(data, (list, np.ndarray)):
        return pd.DataFrame(data)
    
    # If unable to convert, return empty DataFrame
    st.error("Unable to convert input data to DataFrame")
    return pd.DataFrame()

def visualize_features_vs_sales(train_data, input_data=None):
    """
    Create visualizations for features in a single row
    
    Parameters:
    - train_data (DataFrame): Training dataset
    - input_data (dict or DataFrame, optional): Input dataset to compare
    """
    # Safely convert inputs to DataFrames
    train_data = safe_dataframe_conversion(train_data)
    
    # Convert input_data from dict to DataFrame if necessary
    if isinstance(input_data, dict):
        mapped_input = {
            'Item_Visibility': input_data.get('item_visibility', None),
            'Item_MRP': input_data.get('item_mrp', None),
            'Outlet_Size': input_data.get('outlet_size', None),
            'Outlet_Location_Type': input_data.get('outlet_location', None)
        }
        input_data = pd.DataFrame([mapped_input])
    elif input_data is None:
        input_data = pd.DataFrame(columns=train_data.columns)
    else:
        input_data = safe_dataframe_conversion(input_data)
    
    # Features to plot
    features = [
        'Item_Visibility', 
        'Item_MRP', 
        'Outlet_Size', 
        'Outlet_Location_Type'
    ]
    
    # Verify features exist
    missing_features = [f for f in features if f not in train_data.columns]
    if missing_features:
        st.error(f"Missing features in train data: {missing_features}")
        return
    
    # Create a single row with 4 columns
    cols = st.columns(4)
    
    # Plot each feature
    for i, feature in enumerate(features):
        with cols[i]:
            try:
                # Numerical feature handling
                if pd.api.types.is_numeric_dtype(train_data[feature]):
                    fig = go.Figure()
                    
                    # Train data scatter
                    fig.add_trace(go.Scatter(
                        x=train_data[feature],
                        y=train_data['Item_Outlet_Sales'] if 'Item_Outlet_Sales' in train_data.columns else [0]*len(train_data),
                        mode='markers',
                        name='Training Data',
                        marker=dict(color='blue', opacity=0.6)
                    ))
                    
                    # Input data scatter
                    if not input_data.empty and feature in input_data.columns:
                        fig.add_trace(go.Scatter(
                            x=input_data[feature],
                            y=[0] * len(input_data),
                            mode='markers',
                            name='Input Data',
                            marker=dict(color='red', size=10, symbol='star')
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f'{feature} vs Sales',
                        xaxis_title=feature,
                        yaxis_title='Item Outlet Sales',
                        height=300,
                        margin=dict(l=50, r=50, t=30, b=50)
                    )
                
                # Categorical feature handling
                else:
                    # Prepare data
                    train_counts = train_data[feature].value_counts()
                    input_counts = input_data[feature].value_counts() if not input_data.empty and feature in input_data.columns else pd.Series()
                    
                    # Create bar plot for distribution
                    fig = go.Figure()
                    
                    # Train data bar
                    fig.add_trace(go.Bar(
                        x=train_counts.index,
                        y=train_counts.values,
                        name='Training Data',
                        marker_color='blue'
                    ))
                    
                    # Input data bar
                    if not input_counts.empty:
                        fig.add_trace(go.Bar(
                            x=input_counts.index,
                            y=input_counts.values,
                            name='Input Data',
                            marker_color='red'
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f'{feature} Distribution',
                        xaxis_title=feature,
                        yaxis_title='Count',
                        height=300,
                        margin=dict(l=50, r=50, t=30, b=50)
                    )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error plotting {feature}: {str(e)}")



                
if menu == "Results":
    st.header("🔮 Prediction Results")
    
    if 'submitted' not in st.session_state or 'input_data' not in st.session_state:
        st.warning("No input data found. Please go to the 'Input Data' page and submit details first.")
    else:
        model = load_model()
        if model is not None:
            # Prediction Function with Correct Log Transformation Reversal
            def predict_sales(model, input_data):
                # Preprocessing mapping
                outlet_size_map = {'Small': 0, 'Medium': 1, 'High': 2}
                outlet_location_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
                outlet_type_map = {
                    'Grocery Store': [1, 0, 0, 0],
                    'Supermarket Type 1': [0, 1, 0, 0],
                    'Supermarket Type 2': [0, 0, 1, 0],
                    'Supermarket Type 3': [0, 0, 0, 1]
                }
                outlet_identifier_map = {
                    'OUT010': 0, 'OUT013': 1, 'OUT017': 2, 'OUT018': 3, 'OUT019': 4, 'OUT027': 5, 'OUT035': 6, 'OUT045': 7, 'OUT046': 8, 'OUT049': 9
                }

                # Transform inputs
                input_data_transformed = pd.DataFrame({
                    'Item_Visibility': [np.log(input_data['item_visibility'] + 1e-5)],
                    'Item_MRP': [input_data['item_mrp']],
                    'Outlet_Size': [outlet_size_map[input_data['outlet_size']]],
                    'Outlet_Location_Type': [outlet_location_map[input_data['outlet_location']]],
                    'Outlet_Type_0': [outlet_type_map[input_data['outlet_type']][0]],
                    'Outlet_Type_1': [outlet_type_map[input_data['outlet_type']][1]],
                    'Outlet_Type_2': [outlet_type_map[input_data['outlet_type']][2]],
                    'Outlet_Type_3': [outlet_type_map[input_data['outlet_type']][3]],
                    'Outlet_Identifier_0': [1 if outlet_identifier_map[input_data['outlet_identifier']] == 0 else 0],
                    'Outlet_Identifier_1': [1 if outlet_identifier_map[input_data['outlet_identifier']] == 1 else 0],
                    'Outlet_Identifier_2': [1 if outlet_identifier_map[input_data['outlet_identifier']] == 2 else 0],
                    'Outlet_Identifier_3': [1 if outlet_identifier_map[input_data['outlet_identifier']] == 3 else 0],
                    'Outlet_Identifier_4': [1 if outlet_identifier_map[input_data['outlet_identifier']] == 4 else 0],
                    'Outlet_Identifier_5': [1 if outlet_identifier_map[input_data['outlet_identifier']] == 5 else 0],
                    'Outlet_Identifier_6': [1 if outlet_identifier_map[input_data['outlet_identifier']] == 6 else 0],
                    'Outlet_Identifier_7': [1 if outlet_identifier_map[input_data['outlet_identifier']] == 7 else 0],
                    'Outlet_Identifier_8': [1 if outlet_identifier_map[input_data['outlet_identifier']] == 8 else 0],
                    'Outlet_Identifier_9': [1 if outlet_identifier_map[input_data['outlet_identifier']] == 9 else 0],
                })

                # Predict and correctly revert log transformation
                log_prediction = model.predict(input_data_transformed)
                actual_prediction = np.exp(log_prediction[0]) - 1e-5
                return max(0, actual_prediction)  # Ensure non-negative sales

            # Perform prediction
            try:
                input_data = st.session_state['input_data']
                predicted_sales = predict_sales(model, input_data)
                st.success(f"Predicted Sales: ₹{predicted_sales:.2f}")

                # Display Detailed Stats
                st.subheader("Details of Input Values")
                st.write(f"**Item Visibility:** {input_data['item_visibility']}")
                st.write(f"**Item MRP:** ₹{input_data['item_mrp']}")
                st.write(f"**Outlet Size:** {input_data['outlet_size']}")
                st.write(f"**Outlet Location:** {input_data['outlet_location']}")
                st.write(f"**Outlet Type:** {input_data['outlet_type']}")
                st.write(f"**Outlet Identifier:** {input_data['outlet_identifier']}")
                train_data = pd.read_csv(r"train_data_edited.csv")

                visualize_features_vs_sales(train_data, input_data)
        
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.error("Prediction cannot proceed without a valid model.")


if menu == "Insights":
    st.header("📊 Sales Prediction Insights")
    st.markdown("""
    ### Understanding Big Mart Sales Prediction

    This section provides deeper insights into the sales prediction model:

    - **Model Methodology:** 
      - Log-transformed regression model
      - Considers multiple features like item visibility, MRP, outlet characteristics

    - **Key Predictive Features:**
      1. Maximum Retail Price (MRP)
      2. Item Visibility
      3. Outlet Size and Type
      4. Outlet Location

    - **Interpretation Tips:**
      - Higher MRP doesn't always mean higher sales
      - Item visibility plays a crucial role in sales prediction
      - Different outlet types have varying sales potentials
    """)