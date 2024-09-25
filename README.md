# cbrane ML - misc. ML models

## NHL xGoals Regression Trees
This notebook performs regression tree analysis on NHL shot data to predict the expected goals (xGoals) of a shot, which quantifies the quality of offensive opportunities. The analysis includes:

- **Data Loading**: Importing the dataset from `shots_2023.csv`. A dataset from Moneypuck.com containing shot-by-shot data from the 2023-2024 NHL season. You can view and download the data [here](https://moneypuck.com/data).
- **Preprocessing**: Selecting relevant features, handling categorical variables through one-hot encoding, and normalizing numerical features.
- **Model Training**: Utilizing `DecisionTreeRegressor` from scikit-learn to build the regression tree model.
- **Evaluation**: Assessing model performance using Mean Squared Error (MSE) and R-squared (R²) metrics.
- **Feature Importance**: Analyzing which features have the most significant impact on predicting xGoals.
- **Model Optimization**: Training a smaller regression tree with a limited depth to prevent overfitting and improve generalization.
- **Visualization**: Plotting the decision tree structure to interpret the decision-making process of the model.

## Email Spam Data Boosting
This notebook applies boosting techniques using XGBoost to classify email messages as spam or not spam based on the `Spam_Data.csv` dataset. The workflow includes:

- **Data Importing**: Loading the dataset and examining its structure.
- **Preprocessing**: Redefining the response variable, encoding categorical features using `LabelEncoder` and one-hot encoding, and handling missing values.
- **Data Splitting**: Dividing the data into training and testing sets with stratification for classification tasks.
- **Model Training**: Configuring and training an `XGBClassifier` with specified hyperparameters to enhance performance.
- **Cross-Validation**: Implementing K-Fold cross-validation to determine the optimal number of boosting rounds based on validation scores.
- **Evaluation Metrics**: Calculating accuracy, Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE) to evaluate model performance.
- **Model Explanation**: Utilizing LIME (Local Interpretable Model-agnostic Explanations) to interpret individual predictions by highlighting feature contributions.
- **Visualization**: Creating confusion matrices and feature importance plots to gain insights into the model's decision-making and key predictive features.

## MNIST Digit Classification
This notebook trains a neural network to classify handwritten digits from the MNIST dataset. The workflow includes:

- **Data Loading**: Importing the dataset from `mnist_train.csv` and `mnist_test.csv`.
- **Data Visualization**: Plotting sample digits to understand the data.
- **Data Preparation**: Normalizing pixel values, reshaping data, and converting labels to categorical format.
- **Model Building**: Constructing a neural network using TensorFlow and Keras with the following architecture:
  - Input layer (28x28x1)
  - Flatten layer
  - Dense layer with 15 neurons and ReLU activation
  - Output layer with 10 neurons and softmax activation
- **Model Training**: Training the model on the training data with validation split.
- **Model Evaluation**: Evaluating the model's performance on the test set.
- **Misclassification Analysis**: Visualizing misclassified digits to understand the model's errors.
- **Confusion Matrix**: Generating and plotting a normalized confusion matrix to show the percentage accuracy for each class.

Key improvements and additions:
- Reshaped input data to include channel dimension (28x28x1)
- Added a confusion matrix visualization for better performance analysis
- Implemented a function to display misclassified digits

The model achieves approximately 91.67% accuracy on the test set.

## References
- [Moneypuck](https://moneypuck.com/data)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- [LIME Documentation](https://lime-ml.readthedocs.io/en/latest/)
- [MNIST Dataset & Information](http://yann.lecun.com/exdb/mnist/)