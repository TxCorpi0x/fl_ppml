"""
Dataset loaders and processors for federated learning experiments.
Supports multiple datasets including healthcare, financial, and standard ML benchmarks.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader


class HealthcareDataset(Dataset):
    """
    Heart Disease Dataset for Federated Learning

    Features:
    - age: Age in years
    - sex: Sex (1 = male; 0 = female)
    - cp: Chest pain type (0-3)
    - trestbps: Resting blood pressure (mm Hg)
    - chol: Serum cholesterol (mg/dl)
    - fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    - restecg: Resting electrocardiographic results (0-2)
    - thalach: Maximum heart rate achieved
    - exang: Exercise induced angina (1 = yes; 0 = no)
    - oldpeak: ST depression induced by exercise
    - slope: Slope of peak exercise ST segment (0-2)
    - ca: Number of major vessels colored by fluoroscopy (0-3)
    - thal: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)

    Target:
    - target: Heart disease presence (1 = disease, 0 = no disease)
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, transform=None):
        """
        Args:
            data: Feature matrix (n_samples, n_features)
            labels: Target labels (n_samples,)
            transform: Optional transform to apply to features
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    @staticmethod
    def get_feature_names() -> List[str]:
        """Returns list of feature names"""
        return [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]

    @staticmethod
    def get_feature_descriptions() -> Dict[str, str]:
        """Returns detailed feature descriptions"""
        return {
            "age": "Age in years",
            "sex": "Sex (1=male, 0=female)",
            "cp": "Chest pain type (0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic)",
            "trestbps": "Resting blood pressure (mm Hg)",
            "chol": "Serum cholesterol (mg/dl)",
            "fbs": "Fasting blood sugar > 120 mg/dl (1=true, 0=false)",
            "restecg": "Resting ECG results (0=normal, 1=ST-T abnormality, 2=left ventricular hypertrophy)",
            "thalach": "Maximum heart rate achieved",
            "exang": "Exercise induced angina (1=yes, 0=no)",
            "oldpeak": "ST depression induced by exercise relative to rest",
            "slope": "Slope of peak exercise ST segment (0=upsloping, 1=flat, 2=downsloping)",
            "ca": "Number of major vessels colored by fluoroscopy (0-3)",
            "thal": "Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)",
        }


class CreditCardDataset(Dataset):
    """
    Credit Card Fraud Detection Dataset for Federated Learning

    Privacy-sensitive financial transaction data where federated learning
    is essential - banks collaboratively train fraud detection models
    without sharing customer transaction data (PCI-DSS compliance).

    Features (30 total):
    - Time: Seconds elapsed between first transaction and this one
    - V1-V28: Principal components from PCA transformation (anonymized)
    - Amount: Transaction amount

    Target:
    - Class: 0 = legitimate transaction, 1 = fraudulent transaction

    Dataset characteristics:
    - 284,807 transactions
    - Highly imbalanced: ~0.17% fraud rate (492 frauds / 284,315 normal)
    - Real-world European credit card transactions (2 days, Sept 2013)
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, transform=None):
        """
        Args:
            data: Feature matrix (n_samples, 30)
            labels: Target labels (n_samples,) - 0=normal, 1=fraud
            transform: Optional transform to apply
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    @staticmethod
    def get_feature_names() -> List[str]:
        """Returns list of feature names"""
        return ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    @staticmethod
    def get_feature_descriptions() -> Dict[str, str]:
        """Returns detailed feature descriptions"""
        desc = {
            "Time": "Seconds elapsed between first transaction and current",
            "Amount": "Transaction amount in Euros",
        }
        for i in range(1, 29):
            desc[f"V{i}"] = f"PCA component {i} (anonymized feature)"
        return desc


class StockMarketDataset(Dataset):
    """
    Stock Market Dataset for Federated Learning
    Used for both classification (direction) and regression (price prediction)
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)  # Classification labels must be Long
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


def load_creditcard_data(
    filepath: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
    subsample: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess credit card fraud detection dataset

    Args:
        filepath: Path to CSV file. If None, uses default path
        test_size: Proportion of dataset for testing
        random_state: Random seed for reproducibility
        scale: Whether to standardize features
        subsample: Optional - use only first N samples (for faster testing)
                   If None, checks FL_CREDITCARD_SUBSAMPLE env var

    Returns:
        X_train, X_test, y_train, y_test
    """

    # Check environment variable for subsample if not specified
    if subsample is None:
        import os

        env_subsample = os.environ.get("FL_CREDITCARD_SUBSAMPLE")
        if env_subsample:
            subsample = int(env_subsample)

    if filepath is None:
        # Try to load from local dataset folder
        import os

        possible_paths = [
            "../dataset/kaggle/input/mlg-ulb/creditcard.csv",
            "./dataset/kaggle/input/mlg-ulb/creditcard.csv",
            "../../dataset/kaggle/input/mlg-ulb/creditcard.csv",
            "../dataset/creditcard.csv",
            "./dataset/creditcard.csv",
        ]

        filepath = None
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break

        if filepath is None:
            raise FileNotFoundError(
                "Could not find credit card dataset. Please provide filepath or place the dataset in: "
                "dataset/kaggle/input/mlg-ulb/creditcard.csv"
            )

    # Load data
    df = pd.read_csv(filepath)

    # Subsample if requested (for faster testing)
    if subsample is not None and subsample < len(df):
        # Stratified subsample to maintain fraud ratio
        fraud_samples = df[df["Class"] == 1]
        normal_samples = df[df["Class"] == 0]

        # Ensure minimum samples per class for stratification (need at least 2 per class)
        min_fraud = max(
            10, int(subsample * 0.005)
        )  # At least 10 frauds or 0.5% of subsample
        min_normal = subsample - min_fraud

        # Cap at available samples
        n_fraud = min(min_fraud, len(fraud_samples))
        n_normal = min(min_normal, len(normal_samples))

        df = pd.concat(
            [
                fraud_samples.sample(n=n_fraud, random_state=random_state),
                normal_samples.sample(n=n_normal, random_state=random_state),
            ]
        ).sample(
            frac=1, random_state=random_state
        )  # Shuffle

        print(
            f"Subsampled to {len(df)} transactions ({n_fraud} frauds, {n_normal} normal)"
        )

    # Separate features and target
    if "Class" in df.columns:
        X = df.drop("Class", axis=1).values
        y = df["Class"].values
    else:
        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

    # Convert to binary if needed
    y = y.astype(int)

    # Split data (stratified to maintain fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_heart_disease_data(
    filepath: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess heart disease dataset

    Args:
        filepath: Path to CSV file. If None, uses default path
        test_size: Proportion of dataset for testing
        random_state: Random seed for reproducibility
        scale: Whether to standardize features

    Returns:
        X_train, X_test, y_train, y_test
    """

    if filepath is None:
        # Try to load from local dataset folder
        import os

        possible_paths = [
            "../dataset/kaggle/input/heart-disease-data/heart_statlog_cleveland_hungary_final.csv",
            "./dataset/kaggle/input/heart-disease-data/heart_statlog_cleveland_hungary_final.csv",
            "../../dataset/kaggle/input/heart-disease-data/heart_statlog_cleveland_hungary_final.csv",
        ]

        filepath = None
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break

        if filepath is None:
            raise FileNotFoundError(
                "Could not find heart disease dataset. Please provide filepath or place the dataset in: "
                "dataset/kaggle/input/heart-disease-data/heart_statlog_cleveland_hungary_final.csv"
            )

    # Load data
    df = pd.read_csv(filepath)

    # Handle missing values
    df = df.dropna()

    # Separate features and target
    if "target" in df.columns:
        X = df.drop("target", axis=1).values
        y = df["target"].values
    else:
        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

    # Convert target to binary if needed (0: no disease, 1: disease)
    if len(np.unique(y)) > 2:
        y = (y > 0).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def create_federated_creditcard_data(
    num_clients: int = 10,
    alpha: float = 0.5,
    test_size: float = 0.2,
    random_state: int = 42,
    min_samples_per_client: int = 100,
    filepath: Optional[str] = None,
    subsample: Optional[int] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Create federated splits of credit card fraud data with non-IID distribution

    This simulates realistic scenarios where:
    - Different banks have different transaction patterns
    - Some banks experience more fraud than others
    - Geographic and demographic variations create heterogeneity

    Args:
        num_clients: Number of federated clients (e.g., banks)
        alpha: Dirichlet concentration parameter (lower = more non-IID)
               - alpha < 0.5: Highly non-IID (different fraud patterns)
               - alpha = 1.0: Moderately non-IID
               - alpha > 5.0: Nearly IID
        test_size: Proportion for global test set
        random_state: Random seed
        min_samples_per_client: Minimum samples each client must have
        filepath: Optional path to dataset file
        subsample: Optional - use only first N samples (for faster testing)

    Returns:
        client_data: List of (X_client, y_client) tuples
        test_data: (X_test, y_test) tuple for global evaluation
    """

    # Load data
    X_train, X_test, y_train, y_test = load_creditcard_data(
        filepath=filepath,
        test_size=test_size,
        random_state=random_state,
        scale=True,
        subsample=subsample,
    )

    # Get number of classes
    num_classes = len(np.unique(y_train))

    # Create non-IID split using Dirichlet distribution
    client_data = []
    indices_per_client = [[] for _ in range(num_clients)]

    # For each class, distribute samples among clients according to Dirichlet
    for class_id in range(num_classes):
        # Get indices for this class
        class_indices = np.where(y_train == class_id)[0]
        np.random.seed(random_state + class_id)
        np.random.shuffle(class_indices)

        # Generate Dirichlet distribution for client proportions
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))

        # Allocate samples to clients
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        client_class_indices = np.split(class_indices, proportions)

        # Add to client indices
        for client_id, indices in enumerate(client_class_indices):
            indices_per_client[client_id].extend(indices)

    # Create client datasets
    for client_id in range(num_clients):
        # Shuffle client's data
        indices = indices_per_client[client_id]
        np.random.seed(random_state + client_id)
        np.random.shuffle(indices)

        # Check minimum samples requirement
        if len(indices) < min_samples_per_client:
            print(
                f"Warning: Client {client_id} has only {len(indices)} samples "
                f"(min required: {min_samples_per_client})"
            )

        # Extract client's data
        X_client = X_train[indices]
        y_client = y_train[indices]
        client_data.append((X_client, y_client))

    return client_data, (X_test, y_test)


def create_federated_healthcare_data(
    num_clients: int = 10,
    alpha: float = 0.5,
    test_size: float = 0.2,
    random_state: int = 42,
    min_samples_per_client: int = 10,
    filepath: Optional[str] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Create federated splits of healthcare data with non-IID distribution

    This simulates realistic scenarios where:
    - Different hospitals have different patient demographics
    - Some hospitals specialize in certain conditions
    - Data is naturally heterogeneous across institutions

    Args:
        num_clients: Number of federated clients (e.g., hospitals)
        alpha: Dirichlet concentration parameter (lower = more non-IID)
               - alpha < 0.5: Highly non-IID (realistic for hospitals)
               - alpha = 1.0: Moderately non-IID
               - alpha > 5.0: Nearly IID
        test_size: Proportion for global test set
        random_state: Random seed
        min_samples_per_client: Minimum samples each client must have
        filepath: Optional path to dataset file

    Returns:
        client_data: List of (X_client, y_client) tuples
        test_data: (X_test, y_test) tuple for global evaluation
    """

    # Load data
    X_train, X_test, y_train, y_test = load_heart_disease_data(
        filepath=filepath, test_size=test_size, random_state=random_state, scale=True
    )

    # Get number of classes
    num_classes = len(np.unique(y_train))

    # Create non-IID split using Dirichlet distribution
    client_data = []
    indices_per_client = [[] for _ in range(num_clients)]

    # For each class, distribute samples among clients according to Dirichlet
    for class_id in range(num_classes):
        # Get indices for this class
        class_indices = np.where(y_train == class_id)[0]
        np.random.seed(random_state + class_id)
        np.random.shuffle(class_indices)

        # Generate Dirichlet distribution for client proportions
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions * len(class_indices)).astype(int)

        # Ensure at least min_samples_per_client
        proportions = np.maximum(proportions, min_samples_per_client // num_classes)

        # Adjust to match total samples
        diff = len(class_indices) - proportions.sum()
        proportions[0] += diff

        # Distribute indices to clients
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + proportions[client_id]
            indices_per_client[client_id].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx

    # Create client datasets
    for client_id in range(num_clients):
        client_indices = indices_per_client[client_id]

        if len(client_indices) < min_samples_per_client:
            print(f"Warning: Client {client_id} has only {len(client_indices)} samples")

        X_client = X_train[client_indices]
        y_client = y_train[client_indices]

        client_data.append((X_client, y_client))

    return client_data, (X_test, y_test)


def analyze_data_distribution(
    client_data: List[Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """
    Analyze the distribution of data across clients

    Args:
        client_data: List of (X, y) tuples for each client

    Returns:
        DataFrame with distribution statistics
    """
    stats = []

    for client_id, (X, y) in enumerate(client_data):
        unique, counts = np.unique(y, return_counts=True)
        total_samples = len(y)

        stat_dict = {
            "client_id": client_id,
            "total_samples": total_samples,
        }

        # Add class distribution
        for class_id, count in zip(unique, counts):
            stat_dict[f"class_{class_id}_count"] = count
            stat_dict[f"class_{class_id}_ratio"] = count / total_samples

        # Calculate entropy (measure of heterogeneity)
        proportions = counts / total_samples
        entropy = -np.sum(proportions * np.log(proportions + 1e-10))
        stat_dict["entropy"] = entropy

        stats.append(stat_dict)

    return pd.DataFrame(stats)


def create_iid_healthcare_data(
    num_clients: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    filepath: Optional[str] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Create IID (independent and identically distributed) splits of healthcare data

    This is useful for baseline comparisons where data is uniformly distributed.

    Args:
        num_clients: Number of federated clients
        test_size: Proportion for global test set
        random_state: Random seed
        filepath: Optional path to dataset file

    Returns:
        client_data: List of (X_client, y_client) tuples
        test_data: (X_test, y_test) tuple
    """

    # Load data
    X_train, X_test, y_train, y_test = load_heart_disease_data(
        filepath=filepath, test_size=test_size, random_state=random_state, scale=True
    )

    # Shuffle indices
    indices = np.arange(len(X_train))
    np.random.seed(random_state)
    np.random.shuffle(indices)

    # Split evenly among clients
    client_data = []
    samples_per_client = len(indices) // num_clients

    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (
            start_idx + samples_per_client if i < num_clients - 1 else len(indices)
        )

        client_indices = indices[start_idx:end_idx]
        X_client = X_train[client_indices]
        y_client = y_train[client_indices]

        client_data.append((X_client, y_client))

    return client_data, (X_test, y_test)


def load_stock_market_data(
    filepath: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess stock market dataset (Cerner Corporation - CERN)

    Features engineered from OHLC data:
    - Volume (shifted)
    - SMA: Simple Moving Average (20-day)
    - Std_20: 20-day standard deviation
    - Band_1, Band_2: Bollinger Bands
    - ON_returns_signal: Overnight returns signal (categorical)
    - dist_from_mean: Distance from SMA

    Target: StockMove (1 = price goes up, 0 = price goes down)

    Args:
        filepath: Path to CSV file. If None, uses default path
        test_size: Proportion of dataset for testing
        random_state: Random seed for reproducibility
        scale: Whether to standardize features

    Returns:
        X_train, X_test, y_train, y_test
    """

    if filepath is None:
        # Try to load from local dataset folder
        import os

        possible_paths = [
            "../dataset/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks/cern.us.txt",
            "./dataset/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks/cern.us.txt",
            "../../dataset/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks/cern.us.txt",
        ]

        filepath = None
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break

        if filepath is None:
            raise FileNotFoundError(
                "Could not find stock market dataset. Please provide filepath or place the dataset in: "
                "dataset/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks/cern.us.txt"
            )

    # Load data
    df = pd.read_csv(filepath)

    # Feature engineering
    features = df.copy()

    # Drop Date column
    if "Date" in features.columns:
        features = features.drop(["Date"], axis=1)

    # Create features
    features["Volume"] = features["Volume"].shift(1)
    features["SMA"] = features["Close"].rolling(window=20).mean().shift(1)
    features["Std_20"] = features["Close"].rolling(window=20).std().shift(1)
    features["Band_1"] = features["SMA"] - features["Std_20"]
    features["Band_2"] = features["SMA"] + features["Std_20"]
    features["ON_returns"] = features["Close"] - features["Open"].shift(-1)
    features["ON_returns"] = features["ON_returns"].shift(1)
    features["ON_returns_signal"] = np.where(features["ON_returns"] < 0, 1, 0)
    features["dist_from_mean"] = features["Close"].shift(1) - features["SMA"]

    # Drop NaN values
    features = features.dropna()

    # One-hot encode categorical variables
    features = pd.get_dummies(features, columns=["ON_returns_signal"], dtype=int)

    # Drop unnecessary columns
    features = features.drop("ON_returns", axis=1)

    # Create target column (StockMove: 1 = price goes up, 0 = goes down)
    features["StockMove"] = np.where(
        features["Close"] - features["Close"].shift(-1) < 0, 1, 0
    )

    # Drop rows with NaN in target
    features = features.dropna()

    # Drop last row which doesn't have a signal
    features = features[:-1]

    # Separate features and target
    drop_cols = ["Low", "High", "StockMove"]
    X = features.drop(drop_cols, axis=1).values
    y = features["StockMove"].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def create_federated_stock_data(
    num_clients: int = 10,
    alpha: float = 0.5,
    test_size: float = 0.2,
    random_state: int = 42,
    min_samples_per_client: int = 10,
    filepath: Optional[str] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Create federated splits of stock market data with non-IID distribution

    This simulates realistic scenarios where:
    - Different trading firms have different trading strategies
    - Some firms specialize in certain market conditions
    - Data is naturally heterogeneous across institutions

    Args:
        num_clients: Number of federated clients (e.g., trading firms)
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        test_size: Proportion for global test set
        random_state: Random seed
        min_samples_per_client: Minimum samples each client must have
        filepath: Optional path to dataset file

    Returns:
        client_data: List of (X_client, y_client) tuples
        test_data: (X_test, y_test) tuple for global evaluation
    """

    # Load data
    X_train, X_test, y_train, y_test = load_stock_market_data(
        filepath=filepath, test_size=test_size, random_state=random_state, scale=True
    )

    # Get number of classes
    num_classes = len(np.unique(y_train))

    # Create non-IID split using Dirichlet distribution
    client_data = []
    indices_per_client = [[] for _ in range(num_clients)]

    # For each class, distribute samples among clients according to Dirichlet
    for class_id in range(num_classes):
        # Get indices for this class
        class_indices = np.where(y_train == class_id)[0]
        np.random.seed(random_state + class_id)
        np.random.shuffle(class_indices)

        # Generate Dirichlet distribution for client proportions
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions * len(class_indices)).astype(int)

        # Ensure at least min_samples_per_client
        proportions = np.maximum(proportions, min_samples_per_client // num_classes)

        # Adjust to match total samples
        diff = len(class_indices) - proportions.sum()
        proportions[0] += diff

        # Distribute indices to clients
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + proportions[client_id]
            indices_per_client[client_id].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx

    # Create client datasets
    for client_id in range(num_clients):
        client_indices = indices_per_client[client_id]

        if len(client_indices) < min_samples_per_client:
            print(f"Warning: Client {client_id} has only {len(client_indices)} samples")

        X_client = X_train[client_indices]
        y_client = y_train[client_indices]

        client_data.append((X_client, y_client))

    return client_data, (X_test, y_test)


def get_dataloader(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    dataset_type: str = "healthcare",
) -> DataLoader:
    """
    Create a PyTorch DataLoader for the given data

    Args:
        data: Feature matrix
        labels: Target labels
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        dataset_type: Type of dataset ('healthcare' or 'stock')

    Returns:
        DataLoader instance
    """

    if dataset_type == "healthcare":
        dataset = HealthcareDataset(data, labels)
    elif dataset_type == "stock":
        dataset = StockMarketDataset(data, labels)
    elif dataset_type == "creditcard":
        dataset = CreditCardDataset(data, labels)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Healthcare Dataset Loader...")

    # Test 1: Load basic data
    print("\n1. Loading heart disease data...")
    try:
        X_train, X_test, y_train, y_test = load_heart_disease_data()
        print(f"[OK] Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Class distribution (train): {np.bincount(y_train)}")

        # Test 2: Create federated non-IID split
        print("\n2. Creating federated non-IID split...")
        client_data, test_data = create_federated_healthcare_data(
            num_clients=5, alpha=0.5, random_state=42
        )

        print(f"[OK] Number of clients: {len(client_data)}")
        for i, (X, y) in enumerate(client_data):
            print(f"  Client {i}: {len(X)} samples, class dist: {np.bincount(y)}")

        # Test 3: Analyze distribution
        print("\n3. Analyzing data distribution...")
        dist_stats = analyze_data_distribution(client_data)
        print(dist_stats)

        # Test 4: Create DataLoader
        print("\n4. Creating DataLoader...")
        dataloader = get_dataloader(client_data[0][0], client_data[0][1], batch_size=16)
        batch_X, batch_y = next(iter(dataloader))
        print(f"[OK] Batch shape: {batch_X.shape}, Labels: {batch_y.shape}")

        print("\n[OK] All tests passed!")

    except FileNotFoundError as e:
        print(f"\n[FAIL] Error: {e}")
        print("\nPlease ensure the heart disease dataset is available at:")
        print(
            "  dataset/kaggle/input/heart-disease-data/heart_statlog_cleveland_hungary_final.csv"
        )
