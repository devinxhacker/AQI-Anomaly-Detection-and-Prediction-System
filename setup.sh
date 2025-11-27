# #!/bin/bash

# # ============================================================================
# # Automated Setup Script for AQI Anomaly Detection & Prediction System
# # ============================================================================

# set -e  # Exit on error

# # Get the directory where the script is located
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"

# echo "============================================================================"
# echo "AQI Anomaly Detection & Prediction System - Setup v3.0"
# echo "   Enhanced with Live Weather API + 16 ML Models"
# echo "============================================================================"
# echo ""

# # Color codes for output
# GREEN='\033[0;32m'
# BLUE='\033[0;34m'
# YELLOW='\033[1;33m'
# RED='\033[0;31m'
# NC='\033[0m' # No Color

# # Check Python version
# echo -e "${BLUE}[INFO] Checking Python version...${NC}"
# python_version=$(python3 --version 2>&1 | awk '{print $2}')
# required_version="3.8"

# if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
#     echo -e "${RED}[ERROR] Python 3.8+ is required. You have $python_version${NC}"
#     exit 1
# fi
# echo -e "${GREEN}[SUCCESS] Python $python_version detected${NC}"
# echo ""

# # Create virtual environment
# echo -e "${BLUE}[INFO] Creating virtual environment...${NC}"
# if [ ! -d "venv" ]; then
#     python3 -m venv venv
#     echo -e "${GREEN}[SUCCESS] Virtual environment created${NC}"
# else
#     echo -e "${YELLOW}[WARNING] Virtual environment already exists${NC}"
# fi
# echo ""

# # Activate virtual environment
# echo -e "${BLUE}[INFO] Activating virtual environment...${NC}"
# source venv/bin/activate
# echo -e "${GREEN}[SUCCESS] Virtual environment activated${NC}"
# echo ""

# # Upgrade pip
# echo -e "${BLUE}[INFO] Upgrading pip...${NC}"
# pip install --upgrade pip > /dev/null 2>&1
# echo -e "${GREEN}[SUCCESS] pip upgraded${NC}"
# echo ""

# # Install requirements
# echo -e "${BLUE}[INFO] Installing dependencies...${NC}"
# echo -e "${YELLOW}This may take 3-5 minutes...${NC}"
# pip install -r requirements.txt
# echo -e "${GREEN}[SUCCESS] All dependencies installed${NC}"
# echo ""

# # Create directory structure
# echo -e "${BLUE}[INFO] Creating project directories...${NC}"
# mkdir -p data models notebooks src dashboard results tests docs logs
# echo -e "${GREEN}[SUCCESS] Directory structure created${NC}"
# echo ""

# # Check for dataset
# echo -e "${BLUE}[INFO] Checking for dataset...${NC}"
# dataset_found=false

# if [ -f "data/City_Day.csv" ]; then
#     echo -e "${GREEN}[SUCCESS] Dataset found at data/City_Day.csv${NC}"
#     dataset_found=true
# elif [ -f "../dataset.csv" ]; then
#     echo -e "${YELLOW}[WARNING] Linking dataset from parent directory...${NC}"
#     ln -sf "$(pwd)/../dataset.csv" data/City_Day.csv
#     echo -e "${GREEN}[SUCCESS] Dataset linked successfully${NC}"
#     dataset_found=true
# elif [ -f "../City_Day.csv" ]; then
#     echo -e "${YELLOW}[WARNING] Linking dataset from parent directory...${NC}"
#     ln -sf "$(pwd)/../City_Day.csv" data/City_Day.csv
#     echo -e "${GREEN}[SUCCESS] Dataset linked successfully${NC}"
#     dataset_found=true
# else
#     echo -e "${RED}[ERROR] Dataset not found!${NC}"
#     echo -e "${YELLOW}[WARNING] Please download City_Day.csv and place it in the data/ directory${NC}"
#     echo -e "${YELLOW}[WARNING] Dataset: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india${NC}"
# fi
# echo ""

# # Check for API key
# echo -e "${BLUE}[INFO] Checking for OpenWeatherMap API key...${NC}"
# if [ -f ".env" ]; then
#     if grep -q "OPENWEATHER_API_KEY" .env; then
#         echo -e "${GREEN}[SUCCESS] .env file found with API key${NC}"
#     else
#         echo -e "${YELLOW}[WARNING] .env file exists but missing OPENWEATHER_API_KEY${NC}"
#         echo -e "${YELLOW}[WARNING] Please add your API key to .env file${NC}"
#     fi
# else
#     echo -e "${YELLOW}[WARNING] .env file not found${NC}"
#     if [ -f ".env.example" ]; then
#         cp .env.example .env
#         echo -e "${YELLOW}[WARNING] Created .env from template. Please add your API key:${NC}"
#         echo -e "${BLUE}   1. Get free key: https://openweathermap.org/api${NC}"
#         echo -e "${BLUE}   2. Edit .env and add: OPENWEATHER_API_KEY=your_key${NC}"
#     else
#         echo -e "${YELLOW}[WARNING] Create .env file with: OPENWEATHER_API_KEY=your_key${NC}"
#         echo -e "${YELLOW}[WARNING] Get free key from: https://openweathermap.org/api${NC}"
#     fi
# fi
# echo ""

# # Run data preprocessing
# echo -e "${BLUE}[INFO] Running data preprocessing...${NC}"
# if [ -f "src/data_preprocessing.py" ]; then
#     python src/data_preprocessing.py
#     echo -e "${GREEN}[SUCCESS] Data preprocessing completed${NC}"
# else
#     echo -e "${YELLOW}[WARNING] data_preprocessing.py not found. Skipping...${NC}"
# fi
# echo ""

# # Train anomaly detection models
# echo -e "${BLUE}[INFO] Training anomaly detection models...${NC}"
# echo -e "${YELLOW}This may take 5-10 minutes depending on your system...${NC}"
# if [ -f "src/anomaly_detectors.py" ]; then
#     python src/anomaly_detectors.py --train-all
#     echo -e "${GREEN}[SUCCESS] Anomaly detection models trained${NC}"
# else
#     echo -e "${YELLOW}[WARNING] anomaly_detectors.py not found. Skipping...${NC}"
# fi
# echo ""

# # Train prediction models (NEW)
# echo -e "${BLUE}[INFO] Training prediction models (16 ML algorithms)...${NC}"
# echo -e "${YELLOW}This may take 5-10 minutes...${NC}"
# if [ -f "src/aqi_predictor.py" ]; then
#     python3 src/aqi_predictor.py 2>&1 | tail -20
#     if [ $? -eq 0 ]; then
#         echo -e "${GREEN}[SUCCESS] Prediction models trained (7 regression + 6 classification + 3 clustering)${NC}"
#     else
#         echo -e "${RED}[ERROR] Error training prediction models${NC}"
#         exit 1
#     fi
# else
#     echo -e "${YELLOW}[WARNING] aqi_predictor.py not found. Skipping...${NC}"
# fi
# echo ""

# # Generate explanations
# echo -e "${BLUE}[INFO] Generating explainable AI insights...${NC}"
# if [ -f "src/explainable_ai.py" ]; then
#     python src/explainable_ai.py
#     echo -e "${GREEN}[SUCCESS] XAI analysis completed${NC}"
# else
#     echo -e "${YELLOW}[WARNING] explainable_ai.py not found. Skipping...${NC}"
# fi
# echo ""

# echo "============================================================================"
# echo -e "${GREEN}[SUCCESS] Setup completed successfully!${NC}"
# echo "============================================================================"
# echo ""
# echo "Project Structure:"
# echo "   ├── data/              - Raw and processed datasets"
# echo "   ├── models/            - Trained models (generated at runtime)"
# echo "   ├── notebooks/         - Jupyter notebooks"
# echo "   ├── src/               - Source code modules"
# echo "   │   ├── data_preprocessing.py    - Preprocessing pipeline"
# echo "   │   ├── anomaly_detectors.py     - 3 detection algorithms"
# echo "   │   ├── explainable_ai.py        - SHAP + LIME"
# echo "   │   ├── aqi_predictor.py         - 16 ML models (NEW)"
# echo "   │   ├── weather_api.py           - Live weather API (NEW)"
# echo "   │   ├── utils.py                 - Utilities"
# echo "   │   ├── alert_system.py          - Alerts"
# echo "   │   └── visualization.py         - Visualizations"
# echo "   ├── dashboard/         - Streamlit web dashboard (6 pages)"
# echo "   │   └── pages/         - Dashboard pages"
# echo "   │       └── aqi_prediction.py    - Live prediction page (NEW)"
# echo "   ├── results/           - Output visualizations (runtime)"
# echo "   ├── tests/             - Unit tests"
# echo "   └── docs/              - Documentation"
# echo ""
# echo "Next Steps:"
# echo ""
# echo "1. Get OpenWeatherMap API Key (for live predictions):"
# echo "   - Visit: https://openweathermap.org/api"
# echo "   - Sign up (free)"
# echo "   - Copy API key"
# echo "   - Add to .env: OPENWEATHER_API_KEY=your_key"
# echo ""
# echo "2. Read Documentation:"
# echo "   cat QUICK_START_GUIDE.md        # User guide with examples"
# echo "   cat ENHANCEMENT_SUMMARY.md      # What's new in v3.0"
# echo "   cat FEATURE_ENHANCEMENTS.md     # Technical details"
# echo ""
# echo "3. Run Tests:"
# echo "   python3 test_integration.py"
# echo ""
# echo "4. Launch Dashboard:"
# echo "   streamlit run dashboard/streamlit_app.py"
# echo "   (Opens at http://localhost:8501)"
# echo ""
# echo "5. Try New Features:"
# echo "   - Click 'AQI Prediction' in sidebar"
# echo "   - Enter city name (e.g., Delhi, Mumbai)"
# echo "   - Get live AQI + ML predictions"
# echo ""
# echo "6. Use Python API:"
# echo "   python3"
# echo "   >>> from src.aqi_predictor import AQIPredictorSystem"
# echo "   >>> from src.weather_api import WeatherAPI"
# echo "   >>> from src.alert_system import AlertManager"
# echo "   >>> from src.visualization import AnomalyVisualizer"
# echo ""
# echo "Documentation:"
# echo "   - README.md                    - Main overview"
# echo "   - QUICK_START_GUIDE.md         - Setup & usage guide (START HERE)"
# echo "   - ENHANCEMENT_SUMMARY.md       - v3.0 enhancements overview"
# echo "   - FEATURE_ENHANCEMENTS.md      - Technical feature details"
# echo "   - QUICK_REFERENCE.md           - API reference"
# echo "   - IMPLEMENTATION_SUMMARY.md    - What's implemented"
# echo "   - docs/TECHNICAL_REPORT.md     - Technical details"
# echo ""
# if [ "$dataset_found" = false ]; then
#     echo -e "${RED}[WARNING] IMPORTANT: Dataset is missing!${NC}"
#     echo "   Download City_Day.csv from:"
#     echo "   https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india"
#     echo "   Place it in the data/ directory"
#     echo ""
# fi

# echo -e "${BLUE}Happy Analyzing & Predicting!${NC}"
# echo ""
# echo "New in v3.0:"
# echo "   [+] Live Weather API Integration"
# echo "   [+] 16 ML Models (7 regression + 6 classification + 3 clustering)"
# echo "   [+] Real-time AQI Prediction Dashboard"
# echo "   [+] AdaBoost, KNN, Random Forest, Decision Tree, Linear Regression"
# echo "   [+] K-Means, DBSCAN, Hierarchical Clustering"
# echo ""
# echo "For help, refer to QUICK_START_GUIDE.md or run:"
# echo "   python3 test_integration.py"
# echo ""
# echo "============================================================================"
# echo ""

# # Ask if user wants to launch dashboard
# read -p "Would you like to launch the dashboard now? (y/n): " -n 1 -r
# echo ""
# if [[ $REPLY =~ ^[Yy]$ ]]; then
#     echo -e "${GREEN}Launching Streamlit dashboard...${NC}"
#     echo -e "${BLUE}Press Ctrl+C to stop the server${NC}"
#     echo ""
#     streamlit run dashboard/streamlit_app.py
# else
#     echo -e "${YELLOW}To launch later, run:${NC}"
#     echo "   streamlit run dashboard/streamlit_app.py"
#     echo ""
# fi
