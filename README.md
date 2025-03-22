# Flight Delay Prediction Analysis
A statistical project analyzing flight delay predictions using historical data. It employs logistic regression to classify flights as delayed or on-time based on factors like departure time, carrier, origin airport, weather conditions, and past delay patterns.

## Features
- **Data Exploration**: Analyzes the `flights` and `weather` datasets to identify trends and missing values.
- **Data Preprocessing**: Merges datasets, extracts relevant features, and converts data into suitable formats for modeling.
- **Logistic Regression Model**: Trains a logistic regression model to predict flight delays based on selected features.
- **Model Evaluation**: Uses metrics like accuracy, sensitivity, and specificity to evaluate model performance.
- **Visualization**: Plots confusion matrices, feature importance, and delay rates by carrier, hour, and weather conditions.

## Requirements
- R 4.0+
- `tidyverse` for data manipulation
- `tidymodels` for modeling
- `nycflights13` for flight data
- `ggplot2` for visualization

## Installation
1. Clone the repository: `git clone https://github.com/yourusername/flight-delay-prediction-analysis.git`
2. Navigate into the project directory: `cd flight-delay-prediction-analysis`
3. Install dependencies: `install.packages(c("tidyverse", "tidymodels", "nycflights13", "ggplot2"))`

## Usage
1. Run the application: `Rscript main.R`
2. Follow the prompts to explore data, train the model, and visualize results.

## Contributing
Contributions are welcome! Please submit a pull request with your changes.

## License
[MIT License](https://opensource.org/licenses/MIT)
