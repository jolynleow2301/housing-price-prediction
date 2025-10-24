# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Settings
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("✓ All libraries imported successfully!")


class HousingPricePredictor:
    """A class to handle housing price predictions."""

    def __init__(self):
        """Initialize the predictor with empty attributes."""
        self.model = None
        self.scaler = StandardScaler()
        self.df = None  # stores the DataFrame with data, to store, organize and manipulate structured data
        # train-test splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def load_data(
        self, filepath=None, use_sample_data=True
    ):  # data is loaded into the dataframe
        """
        Load housing data.

        Args:
            filepath: str - Path to your CSV file (for Kaggle datasets)
            use_sample_data: bool - If True, uses California Housing dataset from sklearn

        Returns:
            DataFrame with housing data
        """

        if use_sample_data:
            print("Loading California Housing dataset...")
            housing = fetch_california_housing()
            self.df = pd.DataFrame(housing.data, columns=housing.feature_names)
            self.df["PRICE"] = housing.target
            print(
                f"✓ Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns"
            )
        else:
            print(f"Loading data from {filepath}...")
            self.df = pd.read_csv(filepath)
            print(
                f"✓ Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns"
            )
            print(f"Columns: {list(self.df.columns)}")

        return self.df

    def explore_data(self):
        """Explore and visualize the dataset."""
        print("\n" + "=" * 70)
        print("DATA EXPLORATION")
        print("=" * 70)

        # Basic information
        print("\n📊 First few rows:")
        print(self.df.head())

        print("\n📋 Dataset Info:")
        print(self.df.info())

        print("\n📈 Statistical Summary:")
        print(self.df.describe())

        print("\n❓ Missing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")

        # Correlation with target (assumes last column or column named 'PRICE')
        target_col = "PRICE" if "PRICE" in self.df.columns else self.df.columns[-1]
        print(f"\n🔗 Correlation with {target_col}:")
        correlations = self.df.corr()[target_col].sort_values(ascending=False)
        print(correlations)

        # Create visualizations
        self._create_visualizations()

    def _create_visualizations(self):
        """Create exploratory visualizations."""
        sns.set_style("whitegrid")

        # Determine target column
        target_col = "PRICE" if "PRICE" in self.df.columns else self.df.columns[-1]

        # 1. Correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
        # plt.show()
        print("\n✓ Saved: correlation_heatmap.png")
        plt.close()

        # 2. Price distribution
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(self.df[target_col], kde=True, bins=50, color="blue")
        plt.title(f"Distribution of {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        sns.boxplot(y=self.df[target_col], color="skyblue")
        plt.title(f"Boxplot of {target_col}")
        plt.ylabel(target_col)

        plt.tight_layout()
        plt.savefig("price_distribution.png", dpi=300, bbox_inches="tight")
        # plt.show()
        print("✓ Saved: price_distribution.png")
        plt.close()

    def prepare_data(self, target_column='PRICE', test_size=0.2, random_state=42):
        
