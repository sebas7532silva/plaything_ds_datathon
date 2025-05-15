from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def run_grid_search(model, param_grid, X_train, y_train, scoring='accuracy', cv=5):
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_model, best_params


def evaluate_overfitting_classification(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        'Accuracy': [
            accuracy_score(y_train, y_train_pred),
            accuracy_score(y_test, y_test_pred)
        ],
        'Precision': [
            precision_score(y_train, y_train_pred, average='weighted'),
            precision_score(y_test, y_test_pred, average='weighted')
        ],
        'Recall': [
            recall_score(y_train, y_train_pred, average='weighted'),
            recall_score(y_test, y_test_pred, average='weighted')
        ],
        'F1-Score': [
            f1_score(y_train, y_train_pred, average='weighted'),
            f1_score(y_test, y_test_pred, average='weighted')
        ]
    }

    return pd.DataFrame(metrics, index=['Train', 'Test']).T

def evaluate_overfitting_regression(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        'MSE': [
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)
        ],
        'MAE': [
            mean_absolute_error(y_train, y_train_pred),
            mean_absolute_error(y_test, y_test_pred)
        ],
        'R2': [
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)
        ]
    }

    return pd.DataFrame(metrics, index=['Train', 'Test']).T

def plot_confusion_matrix(model, X_test, y_test, labels=None, normalize='true'):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize=normalize)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format=".2f" if normalize else "d")
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()

def plot_regression_predictions(model, X_test, y_test):
    y_pred = model.predict(X_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Values")
    plt.grid(True)
    plt.show()
