from housing_prediction import HousingPricePredictor
import os

### Test load_data function
# Create an instance of the predictor
predictor = HousingPricePredictor()

# Test loading sample data (California Housing dataset)
df_sample = predictor.load_data(use_sample_data=True)
print(df_sample.head())  # Display first few rows to verify

# Optionally, test loading data from a CSV file (provide your CSV path)
# df_csv = predictor.load_data(filepath='path/to/your/data.csv', use_sample_data=False)
# print(df_csv.head())  # Display to verify

# Additional checks
print("Columns loaded:", list(predictor.df.columns))
print("Data shape:", predictor.df.shape)


### Test _create_visualizations function
# Call the visualization creation method
predictor._create_visualizations()

# Verify plot files were created
assert os.path.exists("correlation_heatmap.png"), "Heatmap was not created"
assert os.path.exists(
    "price_distribution.png"
), "Price distribution plot was not created"

print("✓ Visualization files were created successfully!")
