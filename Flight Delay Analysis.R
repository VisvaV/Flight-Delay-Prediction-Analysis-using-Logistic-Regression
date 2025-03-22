# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(nycflights13)
library(ggplot2)
library(vip)  # For feature importance visualization
library(caret)  # For additional evaluation metrics

# Step 1: Data Exploration
data("flights")
data("weather")

# Preview the datasets
glimpse(flights)
glimpse(weather)

# Check for missing values in flights and weather datasets
missing_values_flights <- sum(is.na(flights))
missing_values_weather <- sum(is.na(weather))

cat("Number of missing values in flights:", missing_values_flights, "\n")
cat("Number of missing values in weather:", missing_values_weather, "\n")

# Step 2: Data Preprocessing
# Create a new column indicating if the flight is delayed
flights <- flights %>%
  mutate(arr_delay = ifelse(arr_delay > 30, "late", "on_time"),
         hour = dep_time %/% 100,  # Extract hour from dep_time
         minute = dep_time %% 100,  # Extract minute from dep_time
         origin = factor(origin),     # Convert origin to factor
         carrier = factor(carrier))   # Convert carrier to factor

# Summarize weather data to avoid many-to-many relationship
weather_summary <- weather %>%
  group_by(year, month, day, origin) %>%
  summarize(temp = mean(temp, na.rm = TRUE), 
            precip = sum(precip, na.rm = TRUE), 
            wind_speed = mean(wind_speed, na.rm = TRUE),
            .groups = 'drop')

# Merge flights with summarized weather data on year, month, day, and origin
flights_weather <- flights %>%
  left_join(weather_summary, by = c("year", "month", "day", "origin"))

# Select relevant features including additional ones from weather
flights_weather <- flights_weather %>%
  select(year, month, day, hour, minute, carrier, origin,
         arr_delay,
         temp,           # Temperature in Fahrenheit
         precip,         # Precipitation in inches
         wind_speed)     # Wind speed in mph

# Remove rows with missing values
flights_weather <- na.omit(flights_weather)

# Convert 'arr_delay' to a factor
flights_weather$arr_delay <- as.factor(flights_weather$arr_delay)

# Step 3: Splitting the Data
set.seed(123)
data_split <- initial_split(flights_weather, prop = 0.75)
train_data <- training(data_split)
test_data <- testing(data_split)

cat("Training data size:", nrow(train_data), "\n")
cat("Testing data size:", nrow(test_data), "\n")

# Step 4: Creating a Recipe
flights_rec <- recipe(arr_delay ~ ., data = train_data) %>%
  step_zv(all_predictors()) %>%   # Remove zero variance predictors
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Step 5: Model Specification
logistic_model <- logistic_reg() %>%
  set_engine("glm")

# Combine recipe and model into a workflow
flights_workflow <- workflow() %>%
  add_recipe(flights_rec) %>%
  add_model(logistic_model)

# Step 6: Model Training
flights_fit <- fit(flights_workflow, data = train_data)
cat("Model training completed.\n")

# Step 7: Predictions and Evaluation
predictions <- predict(flights_fit, new_data = test_data) %>%
  bind_cols(test_data)

# Ensure predictions are factors for evaluation metrics
predictions$.pred_class <- as.factor(predictions$.pred_class)

# Evaluate model performance using confusion matrix
confusion_matrix <- conf_mat(predictions, truth = arr_delay, estimate = .pred_class)
print(confusion_matrix)

# Calculate accuracy and other metrics
accuracy_val <- accuracy(predictions, truth = arr_delay, estimate = .pred_class)
sensitivity_val <- sensitivity(predictions$.pred_class, reference = predictions$arr_delay)
specificity_val <- specificity(predictions$.pred_class, reference = predictions$arr_delay)

cat("Accuracy:", round(accuracy_val$.estimate * 100, 2), "%\n")
cat("Sensitivity (Recall):", round(sensitivity_val * 100, 2), "%\n")
cat("Specificity:", round(specificity_val * 100, 2), "%\n")

# Step 8: Visualizing Results

# Plotting confusion matrix as a heatmap
confusion_matrix_plot <- autoplot(confusion_matrix) +
  labs(title = "Confusion Matrix Heatmap", x = "Predicted", y = "Actual") +
  theme_minimal()

# Visualizing predicted vs actual delays
predicted_vs_actual_plot <- ggplot(predictions, aes(x = .pred_class, fill = arr_delay)) +
  geom_bar(position = "dodge") +
  labs(title = "Predicted vs Actual Flight Delays", x = "Predicted Delay", y = "Count") +
  theme_minimal()

# Step 9: Feature Importance Visualization 
importance_model <- vip::vip(flights_fit) +
  labs(title = "Feature Importance for Flight Delay Prediction") +
  theme_minimal()

# Additional EDA: Average Delay by Carrier with Weather Conditions
avg_delay_by_carrier_weather <- flights_weather %>%
  group_by(carrier) %>%
  summarize(avg_arr_delay_rate = mean(arr_delay == "late"),
            avg_temp = mean(temp),
            avg_precipitation = mean(precip),
            avg_wind_speed = mean(wind_speed)) %>%
  ggplot(aes(x=reorder(carrier,-avg_arr_delay_rate), y=avg_arr_delay_rate)) +
  geom_col(fill="lightblue") +
  labs(title="Average Delay Rate by Carrier with Weather Conditions",
       x="Carrier", y="Average Delay Rate") +
  theme_minimal() + 
  coord_flip()

# Additional EDA: Flight Counts by Origin Airport with Average Temperature
avg_temp_by_origin_plot <- flights_weather %>%
  group_by(origin) %>%
  summarize(avg_temp = mean(temp, na.rm = TRUE)) %>%
  arrange(desc(avg_temp)) %>%  # Sort the values in descending order
  ggplot(aes(x = reorder(origin, -avg_temp), y = avg_temp)) +  # Reorder x-axis
  geom_col(fill = "orange", width = 0.6) +  # Increase bar width
  geom_text(aes(label = round(avg_temp, 1)), vjust = -0.5, size = 4) +  # Add data labels
  labs(title = "Average Temperature by Origin Airport",
       x = "Origin Airport", 
       y = "Average Temperature (°F)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +  # Rotate x-axis labels for better readability
  geom_hline(yintercept = mean(flights_weather$temp, na.rm = TRUE), linetype = "dashed", color = "blue") +  # Overall average line
  annotate("text", x = 1, y = mean(flights_weather$temp, na.rm = TRUE) + 2, 
           label = paste("Overall Avg Temp:", round(mean(flights_weather$temp, na.rm = TRUE), 1), "°F"), 
           color = "blue", size = 4)

# Print the updated plot
predicted_vs_actual_plot <- ggplot(predictions, aes(x = .pred_class, fill = arr_delay)) +
  geom_bar(position = "dodge") +
  labs(title = "Predicted vs Actual Flight Delays", x = "Predicted Delay", y = "Count") +
  theme_minimal() +
  geom_text(stat = 'count', aes(label = ..count..), 
            position = position_dodge(width = 0.9), 
            vjust = -0.5,  # Adjusts the vertical position of the labels
            color = "black")  # Color of the text labels





# Step 10: Expanded Menu-driven system with 10 features
menu_choice <- 0
while(menu_choice != 11) {
  cat("\n*** Flight Delay Prediction Project ***\n")
  cat("1. View Overall Model Accuracy\n")
  cat("2. View Impact of Weather Conditions on Delays\n")
  cat("3. View Carrier with Highest Delay Rate\n")
  cat("4. View Delay Rates by Departure Hour\n")
  cat("5. View Delay Rates by Origin Airport and Temperature\n")
  cat("6. Compare Actual vs Predicted Delay Rates\n")
  cat("7. View Delay Rate by Precipitation\n")
  cat("8. View Impact of Wind Speed on Delay Rate\n")
  cat("9. View Delay Rate by Month\n")
  cat("10. View Delay Rate by Carrier and Hour of Day\n")
  cat("11. Exit\n")
  
  menu_choice <- as.integer(readline("Choose an option: "))
  
  if (menu_choice == 1) {
    # Overall Accuracy
    print(confusion_matrix)
    cat("Accuracy:", round(accuracy_val$.estimate * 100, 2), "%\n")
    
  } else if (menu_choice == 2) {
    # Feature importance visualization for weather conditions
    print(importance_model)
    
  } else if (menu_choice == 3) {
    # Carrier with highest delay rate
    print(avg_delay_by_carrier_weather)
    
  } else if (menu_choice == 4) {
    # Delay rates by departure hour
    delay_by_hour_plot <- ggplot(flights_weather, aes(x = hour, fill = arr_delay)) +
      geom_bar(position = "fill") +
      labs(title = "Delay Rate by Departure Hour", x = "Hour of Day", y = "Proportion of Delays")
    print(delay_by_hour_plot)
    
  } else if (menu_choice == 5) {
    # Delay rate by origin and temperature
    print(avg_temp_by_origin_plot)
    
  } else if (menu_choice == 6) {
    # Compare actual vs predicted delays
    print(predicted_vs_actual_plot)
    
  } else if (menu_choice == 7) {
    # Delay rate by precipitation
    delay_by_precip_plot <- ggplot(flights_weather, aes(x = precip, fill = arr_delay)) +
      geom_histogram(position = "fill", bins = 30) +
      labs(title = "Delay Rate by Precipitation", x = "Precipitation (inches)", y = "Proportion of Delays")
    print(delay_by_precip_plot)
    
  } else if (menu_choice == 8) {
    # Impact of wind speed on delay rate
    delay_by_wind_plot <- ggplot(flights_weather, aes(x = wind_speed, fill = arr_delay)) +
      geom_histogram(position = "fill", bins = 30) +
      labs(title = "Delay Rate by Wind Speed", x = "Wind Speed (mph)", y = "Proportion of Delays") +
      theme_minimal()
    print(delay_by_wind_plot)
    
  } else if (menu_choice == 9) {
    # Delay rate by month
    delay_by_month_plot <- ggplot(flights_weather, aes(x = month, fill = arr_delay)) +
      geom_bar(position = "fill") +
      labs(title = "Delay Rate by Month", x = "Month", y = "Proportion of Delays") +
      theme_minimal()
    print(delay_by_month_plot)
    
  } else if (menu_choice == 10) {
    # Delay rate by carrier and hour of day
    delay_by_carrier_hour_plot <- flights_weather %>%
      group_by(carrier, hour) %>%
      summarize(delay_rate = mean(arr_delay == "late")) %>%
      ggplot(aes(x = hour, y = delay_rate, color = carrier)) +
      geom_line() +
      labs(title = "Delay Rate by Carrier and Hour of Day", x = "Hour of Day", y = "Delay Rate") +
      theme_minimal()
    print(delay_by_carrier_hour_plot)
    
  } else if (menu_choice == 11) {
    cat("Exiting the menu. Goodbye!\n")
  } else {
    cat("Invalid choice. Please choose a valid option.\n")
  }
}

          