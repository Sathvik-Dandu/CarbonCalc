import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

# Streamlit Page Config
st.set_page_config(
    page_title="CarbonCalc - CO2 Emissions Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Eco-Friendly Styling
st.markdown("""
    <style>
        :root {
            --primary: #2E8B57;
            --secondary: #3CB371;
            --accent: #20B2AA;
            --background: #F0FFF0;
            --text: #2F4F4F;
            --card: #FFFFFF;
        }
        
        .main {
            background-color: var(--background);
            color: var(--text);
        }

        /* Dark mode support: only change text color to white if in dark mode */
        body[data-theme="dark"], .main[data-theme="dark"] {
            color: #fff !important;
        }
        h1[data-theme="dark"], h2[data-theme="dark"], h3[data-theme="dark"], h4[data-theme="dark"], h5[data-theme="dark"], h6[data-theme="dark"] {
            color: #fff !important;
        }
        p[data-theme="dark"], li[data-theme="dark"], ol[data-theme="dark"], ul[data-theme="dark"], .footer[data-theme="dark"] {
            color: #fff !important;
        }
        
        .stTextInput, .stFileUploader, .stSelectbox, .stSlider {
            border-radius: 10px;
            border: 1px solid var(--secondary);
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--primary);
        }
        
        .stAlert {
            border-radius: 10px;
            background-color: var(--card);
        }
        
        .css-1aumxhk {
            background-color: var(--card) !important;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .eco-card {
            background-color: var(--card);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-left: 5px solid var(--accent);
            color: var(--text);
        }

        /* Eco-card text visibility for dark mode */
        .eco-card[data-theme="dark"], .eco-card *[data-theme="dark"] {
            color: #fff !important;
        }
        .eco-card h4[data-theme="dark"], .eco-card p[data-theme="dark"] {
            color: #fff !important;
        }
        }
        
        .sidebar .sidebar-content {
            background-color: var(--card);
        }
        
        button {
            background-color: var(--primary) !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 8px 16px !important;
        }
        
        button:hover {
            background-color: var(--secondary) !important;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            margin-top: 30px;
            color: var(--text);
            font-size: 0.9em;
        }
        
        .leaf-icon {
            color: var(--primary);
            font-size: 1.2em;
            margin-right: 5px;
        }
        
        /* Alternative navigation */
        .alt-nav {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .alt-nav button {
            flex: 1;
        }
        
        /* Hide preload warning */
        link[rel=preload] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Required columns
REQUIRED_COLUMNS = [
    "MODELYEAR", "MAKE", "MODEL", "VEHICLECLASS", "ENGINESIZE", 
    "CYLINDERS", "TRANSMISSION", "FUELTYPE", "FUELCONSUMPTION_CITY", 
    "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB", "FUELCONSUMPTION_COMB_MPG", "CO2EMISSIONS"
]

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_valid' not in st.session_state:
    st.session_state.data_valid = False
if 'missing_values_handled' not in st.session_state:
    st.session_state.missing_values_handled = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "landing"

# Landing Page
def show_landing_page():
    st.image(
        "https://assets.thehansindia.com/h-upload/2024/01/26/1417931-co2.webp", 
        use_container_width=True,  # Updated from use_column_width to use_container_width
        caption="Reducing Carbon Footprint One Vehicle at a Time"
    )
    
    st.markdown("""
    <div style="background-color: #F0FFF0; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h1 style="text-align: center; color: #2E8B57;">🌿 CarbonCalc</h1>
        <h3 style="text-align: center; color: #3CB371;">AI-Powered CO₂ Emissions Analysis Tool</h3>
        <p style="text-align: center; color: #2F4F4F;">
            CarbonCalc helps organizations and individuals analyze vehicle CO₂ emissions patterns, 
            predict future emissions, and identify opportunities for reducing environmental impact.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="eco-card">
            <h4><span class="leaf-icon">🌱</span> Emission Analysis</h4>
            <p>Analyze vehicle emissions across multiple dimensions with interactive visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="eco-card">
            <h4><span class="leaf-icon">📊</span> Predictive Modeling</h4>
            <p>Use machine learning to predict CO₂ emissions based on vehicle specifications.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="eco-card">
            <h4><span class="leaf-icon">🔍</span> Anomaly Detection</h4>
            <p>Identify unusual emission patterns that may indicate data errors or special cases.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #FFFFFF; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 20px;">
        <h3 style="color: #2E8B57;">How to Use CarbonCalc</h3>
        <ol style="color: #2F4F4F;">
            <li>Upload a CSV file containing vehicle emission data</li>
            <li>Our system will verify the data structure</li>
            <li>Handle any missing values if needed</li>
            <li>Choose between predictive modeling or data analysis</li>
            <li>Explore the results through interactive visualizations</li>
        </ol>
        <p style="color: #2F4F4F;"><strong>Note:</strong> Your CSV file should contain the following columns:</p>
        <ul style="color: #2F4F4F;">
            <li>MODELYEAR</li>
            <li>MAKE</li>
            <li>MODEL</li>
            <li>VEHICLECLASS</li>
            <li>ENGINESIZE</li>
            <li>CYLINDERS</li>
            <li>TRANSMISSION</li>
            <li>FUELTYPE</li>
            <li>FUELCONSUMPTION_CITY</li>
            <li>FUELCONSUMPTION_HWY</li>
            <li>FUELCONSUMPTION_COMB</li>
            <li>FUELCONSUMPTION_COMB_MPG</li>
            <li>CO2EMISSIONS</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Data Validation
def validate_data(df):
    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Proceed with Available Data", key="proceed_available_data"):
                st.session_state.data_valid = True
                st.session_state.missing_cols = missing_cols
                st.experimental_rerun()
        with col2:
            if st.button("Upload Different File", key="upload_different_file"):
                st.session_state.df = None
                st.session_state.data_valid = False
                st.session_state.missing_values_handled = False
                st.experimental_rerun()
    else:
        st.session_state.data_valid = True
        st.success("✅ All required columns are present!")
        check_missing_values(df)

# Missing Value Handling
def check_missing_values(df):
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    
    if not missing_values.empty:
        st.warning("⚠ Missing values detected in the dataset:")
        st.write(missing_values)
        
        st.markdown("### Missing Value Handling Options")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Replace Missing Values with Averages", key="replace_missing_values"):
                imputer = SimpleImputer(strategy='mean')
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                st.session_state.df = df
                st.session_state.missing_values_handled = True
                st.success("Missing values replaced with column averages!")
                st.experimental_rerun()
        
        with col2:
            if st.button("Proceed Without Replacement", key="proceed_without_replacement"):
                st.session_state.missing_values_handled = True
                st.experimental_rerun()
    else:
        st.session_state.missing_values_handled = True
        st.success("No missing values detected in the dataset!")

# Alternative Navigation (Removed buttons from the main page)
def show_alt_navigation():
    pass  # Removed alternative navigation buttons from the main page

# Prediction Page
def show_prediction_page():
    st.markdown("""
    <div class="eco-card">
        <h2>🚗 CO₂ Emissions Prediction</h2>
        <p>Fill in the vehicle details below to predict its CO₂ emissions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.df

    # Handle missing values in the dataset
    if df.isnull().any().any():
        df = df.dropna()

    # Train model
    features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']
    target = 'CO2EMISSIONS'
    X = df[features]
    y = df[target]
    
    # Add polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Train model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Prediction form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            engine_size = st.slider("Engine Size (L)", min_value=1.0, max_value=8.0, value=2.5, step=0.1)
        
        with col2:
            cylinders = st.slider("Number of Cylinders", min_value=3, max_value=12, value=4)
        
        with col3:
            fuel_consumption = st.slider("Fuel Consumption (L/100km)", 
                                        min_value=4.0, max_value=30.0, value=10.0, step=0.5)
        
        submitted = st.form_submit_button("Predict CO₂ Emissions")
        
        if submitted:
            # Prepare input for prediction
            input_data = np.array([[engine_size, cylinders, fuel_consumption]])
            input_poly = poly.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_poly)
            
            # Display result
            st.markdown(f"""
            <div class="eco-card" style="background-color: #E8F5E9; border-left: 5px solid #4CAF50;">
                <h3>Prediction Result</h3>
                <p>The predicted CO₂ emissions for this vehicle configuration is:</p>
                <h2 style="color: #2E8B57;">{prediction[0]:.2f} g/km</h2>
                <p>This vehicle is in the <strong>{get_emission_category(prediction[0])}</strong> category.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show comparison with similar vehicles
            similar_vehicles = df[
                (df['ENGINESIZE'].between(engine_size-0.5, engine_size+0.5)) &
                (df['CYLINDERS'].between(cylinders-1, cylinders+1)) &
                (df['FUELCONSUMPTION_COMB'].between(fuel_consumption-2, fuel_consumption+2))
            ]
            
            if not similar_vehicles.empty:
                st.markdown("### Similar Vehicles in Dataset")
                st.dataframe(similar_vehicles[['MAKE', 'MODEL', 'ENGINESIZE', 'CYLINDERS', 
                                             'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']].head(10))

# Define the get_emission_category function
def get_emission_category(co2_value):
    """
    Categorize CO₂ emissions into predefined categories based on the value.
    """
    if co2_value < 100:
        return "Ultra Low"
    elif 100 <= co2_value < 150:
        return "Low"
    elif 150 <= co2_value < 200:
        return "Medium"
    else:
        return "High"

# Analysis Page
def show_analysis_page():
    st.markdown("""
    <div class="eco-card">
        <h2>📊 Data Analysis</h2>
        <p>Explore your vehicle emissions data through interactive visualizations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.df
    
    analysis_type = st.radio("Choose Analysis Type", 
                            ["Univariate Analysis", "Multivariate Analysis"],
                            horizontal=True)
    
    if analysis_type == "Univariate Analysis":
        show_univariate_analysis(df)
    else:
        show_multivariate_analysis(df)

# Univariate Analysis (Added heatmap option)
def show_univariate_analysis(df):
    st.markdown("""
    <div class="eco-card">
        <h3>🔍 Univariate Analysis</h3>
        <p>Analyze the relationship between a single factor and CO₂ emissions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected_column = st.selectbox("Select Factor to Analyze", 
                                   df.select_dtypes(include=['object', 'number']).columns.difference(['CO2EMISSIONS']))
    
    graph_type = st.selectbox("Select Graph Type", ["Scatter Plot", "Box Plot", "Bar Chart", "Heatmap"])
    
    st.markdown(f"### {selected_column} vs CO₂ Emissions")
    
    if graph_type == "Scatter Plot":
        fig = px.scatter(df, x=selected_column, y="CO2EMISSIONS", 
                         color_discrete_sequence=['#2E8B57'])
    elif graph_type == "Box Plot":
        fig = px.box(df, x=selected_column, y="CO2EMISSIONS", 
                     color_discrete_sequence=['#3CB371'])
    elif graph_type == "Bar Chart":
        fig = px.bar(df, x=selected_column, y="CO2EMISSIONS", 
                     color_discrete_sequence=['#20B2AA'])
    elif graph_type == "Heatmap":
        fig = px.density_heatmap(df, x=selected_column, y="CO2EMISSIONS", 
                                 color_continuous_scale='Emrld')
    
    st.plotly_chart(fig, use_container_width=True)

# Multivariate Analysis (Fix for correlation calculation with non-numeric columns)
def show_multivariate_analysis(df):
    st.markdown("""
    <div class="eco-card">
        <h3>📈 Multivariate Analysis</h3>
        <p>Explore relationships between CO₂ emissions and multiple factors.</p>
    </div>
    """, unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=['number']).columns.difference(['CO2EMISSIONS']).tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox("X-Axis", numeric_cols)
    
    with col2:
        z_axis = st.selectbox("Z-Axis", numeric_cols)
    
    color_by = st.selectbox("Color By (Optional)", ['None'] + categorical_cols)
    graph_type = st.selectbox("Select Graph Type", ["3D Scatter Plot", "3D Line Plot", "Heatmap", "Feature Correlation"])
    
    st.markdown(f"### CO₂ Emissions vs {x_axis} and {z_axis}")
    
    if graph_type == "3D Scatter Plot":
        if color_by == 'None':
            fig = px.scatter_3d(df, x=x_axis, y="CO2EMISSIONS", z=z_axis,
                                color_discrete_sequence=['#2E8B57'])
        else:
            fig = px.scatter_3d(df, x=x_axis, y="CO2EMISSIONS", z=z_axis, color=color_by,
                                color_continuous_scale='Emrld')
    elif graph_type == "3D Line Plot":
        if color_by == 'None':
            fig = px.line_3d(df, x=x_axis, y="CO2EMISSIONS", z=z_axis,
                             color_discrete_sequence=['#3CB371'])
        else:
            fig = px.line_3d(df, x=x_axis, y="CO2EMISSIONS", z=z_axis, color=color_by,
                             color_continuous_scale='Emrld')
    elif graph_type == "Heatmap":
        fig = px.density_heatmap(df, x=x_axis, y="CO2EMISSIONS", z=z_axis,
                                 color_continuous_scale='Emrld')
    elif graph_type == "Feature Correlation":
        numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
        corr = numeric_df.corr()  # Compute correlation matrix
        fig = px.imshow(corr, labels=dict(x="Features", y="Features", color="Correlation"),
                        x=corr.columns, y=corr.columns, color_continuous_scale='Emrld')
        fig.update_xaxes(side="top")
    
    st.plotly_chart(fig, use_container_width=True)

# Machine Learning Insights
def show_ml_insights():
    st.markdown("""
    <div class="eco-card">
        <h2>🤖 Advanced Machine Learning Insights</h2>
        <p>Discover hidden patterns and insights using advanced machine learning techniques.</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.df
    
    ml_tabs = ["🔍 Anomaly Detection", "📊 Vehicle Clustering", "📈 CO2 Emissions Prediction Model"]
    tab1, tab2, tab3 = st.tabs(ml_tabs)
    
    with tab1:
        st.markdown("""
        <div class="eco-card">
            <h3>🔍 Anomaly Detection</h3>
            <p>Identify unusual patterns in vehicle emissions that may indicate data errors or special cases.</p>
        </div>
        """, unsafe_allow_html=True)
        
        features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']
        X = df[features]
        
        # Handle missing values
        if X.isnull().any().any():
            st.warning("⚠ Missing values detected. Imputing missing values with column means.")
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        contamination = st.slider("Anomaly Sensitivity", 0.01, 0.2, 0.05, 0.01,
                                 help="Lower values detect only the most extreme anomalies")
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        df['anomaly'] = iso_forest.fit_predict(X)
        df['anomaly'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
        
        st.write("### Anomaly Detection Results")
        st.dataframe(df[['MAKE', 'MODEL', 'ENGINESIZE', 'CYLINDERS', 
                        'FUELCONSUMPTION_COMB', 'CO2EMISSIONS', 'anomaly']].sort_values('CO2EMISSIONS'))
        
        fig = px.scatter(df, x='ENGINESIZE', y='CO2EMISSIONS', 
                        color='anomaly', color_discrete_map={'Anomaly': '#FF6347', 'Normal': '#2E8B57'},
                        hover_data=['MAKE', 'MODEL'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("""
        <div class="eco-card">
            <h3>📊 Vehicle Clustering</h3>
            <p>Group vehicles into clusters based on their specifications and emissions characteristics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']
        X = df[features]
        
        # Handle missing values
        if X.isnull().any().any():
            st.warning("⚠ Missing values detected. Imputing missing values with column means.")
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_clusters = st.slider("Number of Clusters", 2, 6, 3)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        silhouette_avg = silhouette_score(X_scaled, df['cluster'])
        st.write(f"### Clustering Quality Score: {silhouette_avg:.2f}")
        st.markdown("""
        The Silhouette Score ranges from -1 to 1, where:
        - Values near +1 indicate well-separated clusters
        - Values near 0 indicate overlapping clusters
        - Values near -1 indicate incorrect clustering
        """)
        
        fig = px.scatter_3d(df, x='ENGINESIZE', y='CO2EMISSIONS', z='FUELCONSUMPTION_COMB',
                           color='cluster', color_continuous_scale='Emrld',
                           hover_data=['MAKE', 'MODEL'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### Cluster Characteristics")
        cluster_stats = df.groupby('cluster')[features].mean()
        st.dataframe(cluster_stats.style.background_gradient(cmap='Greens'))
    
    with tab3:
        st.markdown("""
        <div class="eco-card">
            <h3>📈 CO2 Emissions Prediction Model</h3>
            <p>Explore the performance of our machine learning model for predicting CO2 emissions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']
        target = 'CO2EMISSIONS'
        X = df[features]
        y = df[target]
        
        # Handle missing values
        if X.isnull().any().any():
            st.warning("⚠ Missing values detected. Imputing missing values with column means.")
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("Mean Squared Error", f"{mse:.2f}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y, 
            y=y_pred, 
            mode='markers',
            marker=dict(color='#2E8B57'),
            name='Actual vs Predicted'
        ))
        fig.add_trace(go.Scatter(
            x=[y.min(), y.max()], 
            y=[y.min(), y.max()],
            mode='lines',
            line=dict(color='#FF6347', dash='dash'),
            name='Perfect Prediction'
        ))
        fig.update_layout(
            title="Actual vs Predicted CO2 Emissions",
            xaxis_title="Actual Emissions (g/km)",
            yaxis_title="Predicted Emissions (g/km)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### Model Coefficients")
        coef_df = pd.DataFrame({
            'Feature': poly.get_feature_names_out(features),
            'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=False)
        st.dataframe(coef_df.style.bar(color='#3CB371'))

# Main App Logic (Removed alternative navigation call)
def main():
    # Sidebar uploader
    with st.sidebar:
        st.header("📤 Data Upload")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="sidebar_uploader")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                validate_data(df)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        # Navigation when data is ready
        if st.session_state.get('data_valid') and st.session_state.get('missing_values_handled'):
            st.header("🔍 Analysis Options")
            page = st.radio("Choose page:", 
                           ["Data Analysis", "Prediction", "ML Insights"],
                           key="sidebar_nav")
            
            if page == "Data Analysis":
                st.session_state.current_page = "analysis"
            elif page == "Prediction":
                st.session_state.current_page = "prediction"
            elif page == "ML Insights":
                st.session_state.current_page = "ml"

    # Main content area
    if st.session_state.df is None:
        show_landing_page()
    else:
        if not st.session_state.data_valid:
            validate_data(st.session_state.df)
        elif not st.session_state.missing_values_handled:
            check_missing_values(st.session_state.df)
        else:
            # Removed alternative navigation call
            if st.session_state.current_page == "analysis":
                show_analysis_page()
            elif st.session_state.current_page == "prediction":
                show_prediction_page()
            elif st.session_state.current_page == "ml":
                show_ml_insights()
            else:
                show_landing_page()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🌍 CarbonCalc - Helping reduce carbon emissions through data analysis</p>
        <p>© 2023 GreenTech Analytics. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
