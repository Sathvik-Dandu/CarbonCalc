🌍 CarbonCalc - CO₂ Emission Analysis & Prediction Tool

CarbonCalc is an eco-conscious web application built using Python and Machine Learning that enables users to analyze, predict, and visualize CO₂ emissions based on vehicle specifications. It uses a clean and interactive frontend powered by Streamlit and runs entirely in a web browser.
I had also provided two csv files in which observed_emission_full have full data while observed_emission_missed have few incomplete(missing values) and the co2 emission values in the dataset is based on the recent survey which says is 121.3g/km on average.
---

 🚀 Project Overview

CarbonCalc allows users to:
- Upload CSV datasets containing vehicle and fuel consumption data.
- Validate the dataset format automatically.
- Handle missing values using smart options.
- Predict CO₂ emissions using ML algorithms.
- Explore powerful visualizations using univariate and multivariate analysis in 2D/3D plots.
- Enjoy an eco-friendly user interface with meaningful illustrations and graphs.

---

 🧠 Machine Learning Integration

CarbonCalc uses a trained Multivariate Linear Regression model to predict CO₂ emissions. It considers attributes like:
- Engine size
- Cylinders
- Fuel consumption (city, highway, combined)

---

 🖼️ App Structure

- Landing Page: Title + Beautiful eco-friendly image + App description.
- CSV Upload: Upload your dataset with clear instructions.
  - Required columns:
    
    MODELYEAR, MAKE, MODEL, VEHICLECLASS, ENGINESIZE, CYLINDERS, TRANSMISSION, 
    FUELTYPE, FUELCONSUMPTION_CITY, FUELCONSUMPTION_HWY, FUELCONSUMPTION_COMB, 
    FUELCONSUMPTION_COMB_MPG, CO2EMISSIONS
    
  - If any columns are missing, an error message is shown with options to:
    - Re-upload CSV file.
    - Proceed with available data.

- Missing Values Handling:
  - View count of missing values per column.
  - Choose between:
    - Replace Missing Values (with column mean).
    - Proceed Normally (skip handling).

- Prediction Page:
  - Fill in vehicle data and get predicted CO₂ emissions instantly.

- Analysis Page:
  - Choose between:
    - Univariate Analysis:
      - Select one column to analyze with graphs (Dropdown).
    - Multivariate Analysis:
      - Select multiple columns to compare with CO₂ emissions in 3D.
      - Graph types include: Scatter Plot, Histogram, Bar Chart, Line Graph, Heatmap, and more.

---

 🛠️ Installation & Setup

 📦 Install Dependencies

Run the following command to install all required packages:
pip install streamlit pandas matplotlib seaborn scikit-learn numpy

▶️ Run the App
streamlit run app.py

🖌️ Design Features
Fully responsive and eco-friendly UI 🌿

Graphs powered by Matplotlib and Seaborn

Cool and minimal design with Earth-tone themes

Informative error handling and smooth navigation

📁 Folder Structure
CarbonCalc/
├── app.py
├── csv file 1              
├── csv file 2

📸 Screenshots


👨‍💻 Author
Developed by Sathvik

🌱 Let's Build a Greener Future Together!
