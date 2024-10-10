
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (Replace 'your_dataset.csv' with the actual dataset path)
df = pd.read_csv('dataset/sample_customer_data_for_exam.csv')

# a. Display the first few rows and summary statistics for numerical columns
print("First few rows of the dataset:")
print(df.head())

print("\nSummary statistics for numerical columns:")
print(df.describe())

# b. Create a heatmap to visualize the correlation between numerical variables
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numerical_cols].corr()

# Create the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,  # Show correlation values
    cmap='coolwarm',  # Color scheme from red (negative) to blue (positive)
    center=0,  # Center the colormap at 0
    fmt='.2f',  # Format correlation values to 2 decimal places
    square=True,  # Make the plot square-shaped
    linewidths=0.5  # Add gridlines
)

# Customize the plot
plt.title('Correlation Heatmap of Numerical Variables', pad=20, size=16)
plt.tight_layout()

# Show the plot
plt.show()

# Print the correlation matrix (optional)
print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))
# c. Create histograms for the "age" and "income" columns
plt.figure(figsize=(12, 5))

# Histogram for 'age'
plt.subplot(1, 2, 1)
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Histogram for 'income'
plt.subplot(1, 2, 2)
plt.hist(df['income'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Income')
plt.xlabel('Income')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# d. Generate a box plot to show the distribution of "purchase_amount" across different "product_factory" values
plt.figure(figsize=(12, 6))

# Create the box plot
sns.boxplot(
    data=df,
    x='product_category',
    y='purchase_amount',
    palette='Set3'  # Use a colorful palette for different categories
)

# Customize the plot
plt.title('Distribution of Purchase Amounts by Product Category', pad=20, size=14)
plt.xlabel('Product Category', size=12)
plt.ylabel('Purchase Amount', size=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print summary statistics (optional)
print("\nSummary Statistics by Product Category:")
print(df.groupby('product_category')['purchase_amount'].describe().round(2))
# e. Create a pie chart to visualize the proportion of customers by "gender"
gender_counts = df['gender'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'], startangle=90)
plt.title('Proportion of Customers by Gender')
plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
plt.show()


# Part B


# Calculate average purchase amount for each education level
result = df.groupby('education')['purchase_amount'].mean().reset_index()

# Sort the result by average purchase amount in descending order
result = result.sort_values('purchase_amount', ascending=False)

# Display the result
print(result)

result = df.groupby('loyalty_status')['satisfaction_score'].mean().reset_index()

# Sort the result by average satisfaction score in descending order
result = result.sort_values('satisfaction_score', ascending=False)

# Display the result
print(result)

total_customers = len(df)
promo_users = df['promotion_usage'].sum()
percentage_promo_users = (promo_users / total_customers) * 100

# Display the result
print(f"Percentage of customers who used promotional offers: {percent}



# Step 1: Prepare the data for modeling

# Handle missing values (if any)
df = df.dropna()


# Encode categorical variables
def one_hot_encode(df, column):
    encoded = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, encoded], axis=1)
    df = df.drop(column, axis=1)
    return df


categorical_columns = ['loyalty_status', 'gender']  # Add other categorical columns as needed
for col in categorical_columns:
    df = one_hot_encode(df, col)

# Select features and target
features = ['age', 'satisfaction_score', 'purchase_frequency']  # Add other relevant features
features += [col for col in df.columns if col.startswith(tuple(categorical_columns))]
X = df[features]
y = df['purchase_amount']


# Step 2: Split the data into training and testing sets
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]


X_train, X_test, y_train, y_test = train_test_split(X, y)


# Step 3: Implement a custom linear regression model
class CustomLinearRegression:
    def _init_(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        X = np.column_stack((np.ones(X.shape[0]), X))
        self.coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        return X.dot(self.coefficients) + self.intercept


# Train the model
model = CustomLinearRegression()
model.fit(X_train, y_train)


# Step 4: Evaluate the model's performance
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r_squared(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Step 5: Identify the top three features
feature_importance = pd.Series(abs(model.coefficients), index=X.columns)
top_features = feature_importance.nlargest(3)

print("\nTop 3 features contributing to purchase amount prediction:")
for feature, importance in top_features.items():
    print(f"{feature}: {importance:.4f}")
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('dataset/sample_customer_data_for_exam.csv')


# Step 1: Prepare the data for classification
def one_hot_encode(df, column):
    encoded = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, encoded], axis=1)
    df = df.drop(column, axis=1)
    return df


# Handle missing values
df = df.dropna()

# Encode categorical variables
categorical_columns = ['loyalty_status', 'gender']  # Add other categorical columns as needed
for col in categorical_columns:
    df = one_hot_encode(df, col)

# Select features and target
features = ['age', 'satisfaction_score', 'purchase_frequency', 'purchase_amount']  # Add other relevant features
features += [col for col in df.columns if col.startswith(tuple(categorical_columns))]
X = df[features]
y = df['promotion_usage']  # Assuming this is the correct column name for promotion usage


# Step 2: Split the data
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]


X_train, X_test, y_train, y_test = train_test_split(X, y)


# Step 3: Implement logistic regression
class CustomLogisticRegression:
    def _init_(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]


# Train the model
model = CustomLogisticRegression()
model.fit(X_train.values, y_train.values)


# Step 4: Evaluate the model
def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score


y_pred = model.predict(X_test.values)
accuracy, precision, recall, f1_score = calculate_metrics(y_test.values, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")


# Step 5: Create a confusion matrix
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


cm = confusion_matrix(y_test.values, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Step 6: Identify top three factors
feature_importance = pd.Series(abs(model.weights), index=X.columns)
top_features = feature_importance.nlargest(3)

print("\nTop 3 factors contributing to promotion usage prediction:")
for feature, importance in top_features.items():
    print(f"{feature}: {importance:.4f}")