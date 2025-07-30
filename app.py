import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, session
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend
import matplotlib.pyplot as plt
import io
import base64
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['MODEL_FOLDER'] = 'models'
app.secret_key = 'your_secret_key'  # Replace with a secure key in production
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Global storage for last trained model metadata
model_metadata = {'feature_names': [], 'task_type': None, 'is_text_data': False}

def preprocess_data(df, task_type, target_column):
    original_rows = len(df)
    df = df.drop_duplicates()
    cleaned_rows = len(df)
    
    # Handle missing values
    missing_values = df.isnull().sum().sum()
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df.loc[:, col] = df[col].fillna(df[col].mean())
        else:
            df.loc[:, col] = df[col].fillna(df[col].mode()[0])
    
    # Track removed rows
    removed_rows = original_rows - cleaned_rows
    
    # Specialized preprocessing for spam.csv-like data
    if task_type == 'classification' and target_column == 'Category' and 'Message' in df.columns:
        # Validate and clean Category column
        valid_labels = {'ham', 'spam'}
        invalid_labels = set(df['Category'].dropna()) - valid_labels
        if invalid_labels:
            raise ValueError(f"Invalid labels found in Category: {invalid_labels}")
        # Create new spam column with integer labels
        df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
        # Use CountVectorizer for text data
        vectorizer = CountVectorizer(stop_words='english', max_features=2000)
        X = df['Message']
        X_vectorized = vectorizer.fit_transform(X)
        y = df['spam'].values
        feature_names = vectorizer.get_feature_names_out()
        # Save vectorizer
        vectorizer_path = os.path.join(app.config['MODEL_FOLDER'], 'vectorizer.pkl')
        try:
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            if not os.path.exists(vectorizer_path):
                raise IOError(f"Failed to create vectorizer file at {vectorizer_path}")
        except Exception as e:
            raise
        return X_vectorized, y, feature_names, original_rows, cleaned_rows, missing_values, removed_rows, True
    else:
        # Numerical preprocessing
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols)
        
        feature_names = df.drop(columns=[target_column]).columns.tolist()
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        if len(X.columns) == 1:
            X = X.values.reshape(-1, 1)
        else:
            X = X.values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Save scaler
        scaler_path = os.path.join(app.config['MODEL_FOLDER'], 'scaler.pkl')
        try:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            if not os.path.exists(scaler_path):
                raise IOError(f"Failed to create scaler file at {scaler_path}")
        except Exception as e:
            raise
        return X_scaled, y, feature_names, original_rows, cleaned_rows, missing_values, removed_rows, False

def train_and_evaluate_models(X, y, task_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if task_type == 'classification' and model_metadata.get('is_text_data', False):
        # Use Naive Bayes for spam classification
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            'Naive Bayes': {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        }
    else:
        if task_type == 'classification':
            models = {
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True)
            }
        else:  # regression
            models = {
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Random Forest': RandomForestRegressor(random_state=42),
                'Linear Regression': LinearRegression(),
                'SVM': SVR()
            }
        metrics = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if task_type == 'classification':
                metrics[name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)
                }
            else:
                metrics[name] = {
                    'MSE': mean_squared_error(y_test, y_pred),
                    'R2 Score': r2_score(y_test, y_pred)
                }
    
    return metrics

def plot_metrics(metrics, task_type):
    plt.figure(figsize=(10, 6))
    if task_type == 'classification':
        metrics_names = ['Accuracy', 'Precision', 'Recall']
        bar_width = 0.25
        index = np.arange(len(metrics))
        colors = ['#4A90E2', '#50C878', '#4682B4']  # Sober colors
        for i, metric in enumerate(metrics_names):
            values = [metrics[model][metric] for model in metrics]
            bars = plt.bar(index + i * bar_width, values, bar_width, label=metric, color=colors[i])
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.3f}',
                         ha='center', va='bottom')
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title(f'Model Performance Metrics ({task_type.capitalize()})')
        plt.xticks(index + bar_width, metrics.keys())
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    else:  # regression
        fig, ax1 = plt.subplots(figsize=(10, 6))
        index = np.arange(len(metrics))
        bar_width = 0.35
        
        # MSE bars on primary y-axis
        mse_values = [metrics[model]['MSE'] for model in metrics]
        bars1 = ax1.bar(index, mse_values, bar_width, label='MSE', color='#4A4A4A')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('MSE', color='#4A4A4A')
        ax1.set_ylim(0, max(mse_values) * 1.2)
        ax1.tick_params(axis='y', labelcolor='#4A4A4A')
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2e}' if height > 1e5 else f'{height:.2f}',
                     ha='center', va='bottom')

        # R2 Score bars on secondary y-axis
        ax2 = ax1.twinx()
        r2_values = [metrics[model]['R2 Score'] for model in metrics]
        bars2 = ax2.bar(index + bar_width, r2_values, bar_width, label='R2 Score', color='#90EE90')
        ax2.set_ylabel('R2 Score', color='#90EE90')
        ax2.set_ylim(0, 1.2)
        ax2.tick_params(axis='y', labelcolor='#90EE90')
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom')

        plt.title(f'Model Performance Metrics (Regression)')
        plt.xticks(index + bar_width / 2, metrics.keys(), rotation=45, ha='right')
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic

@app.route('/', methods=['GET', 'POST'])
def index():
    global df  # Declare global df
    if request.method == 'POST':
        file = request.files.get('file')
        task_type = request.form.get('task_type')
        target_column = request.form.get('target_column')
        
        if not file or not task_type or not target_column:
            return render_template('index.html', error='Please provide a CSV file, task type, and target column.')
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            df = pd.read_csv(file_path)
            if target_column not in df.columns:
                return render_template('index.html', error='Target column not found in CSV.')
            
            # Reset model_metadata to avoid stale data
            model_metadata.clear()
            model_metadata.update({'feature_names': [], 'task_type': None, 'is_text_data': False})
            
            # Clear models folder before preprocessing
            for model_file in os.listdir(app.config['MODEL_FOLDER']):
                try:
                    file_path = os.path.join(app.config['MODEL_FOLDER'], model_file)
                    os.remove(file_path)
                except Exception as e:
                    raise
            
            X, y, feature_names, original_rows, cleaned_rows, missing_values, removed_rows, is_text_data = preprocess_data(df, task_type, target_column)
            metrics = train_and_evaluate_models(X, y, task_type)
            plot_data = plot_metrics(metrics, task_type)
            metric_names = ['Accuracy', 'Precision', 'Recall'] if task_type == 'classification' else ['MSE', 'R2 Score']
            
            if task_type == 'classification' and is_text_data:
                best_model = 'Naive Bayes'
                model_obj = MultinomialNB()
                model_obj.fit(X, y)
            else:
                if task_type == 'classification':
                    best_model = max(metrics.items(), key=lambda x: x[1]['Accuracy'])[0]
                else:
                    best_model = min(metrics.items(), key=lambda x: x[1]['MSE'])[0]
                
                model_map = {
                    'Decision Tree': DecisionTreeClassifier(random_state=42) if task_type == 'classification' else DecisionTreeRegressor(random_state=42),
                    'Random Forest': RandomForestClassifier(random_state=42) if task_type == 'classification' else RandomForestRegressor(random_state=42),
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                    'Linear Regression': LinearRegression(),
                    'SVM': SVC(random_state=42, probability=True) if task_type == 'classification' else SVR()
                }
                model_obj = model_map[best_model]
                model_obj.fit(X, y)
            
            # Store metadata for prediction
            model_metadata['feature_names'] = feature_names
            model_metadata['task_type'] = task_type
            model_metadata['is_text_data'] = is_text_data
            
            # Save the new best model
            model_path = os.path.join(app.config['MODEL_FOLDER'], f"{best_model.replace(' ', '_')}_model.pkl")
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model_obj, f)
                if not os.path.exists(model_path):
                    raise IOError(f"Failed to create model file at {model_path}")
            except Exception as e:
                raise
            
            return render_template('results.html', metrics=metrics, plot_data=plot_data, task_type=task_type, 
                                 metric_names=metric_names, original_rows=original_rows, 
                                 cleaned_rows=cleaned_rows, missing_values=missing_values, 
                                 removed_rows=removed_rows, best_model=best_model)
        except Exception as e:
            return render_template('index.html', error=f'Error processing file: {str(e)}')
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    raise
    
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model_exists = len(os.listdir(app.config['MODEL_FOLDER'])) > 0
    model_path = None
    for file in os.listdir(app.config['MODEL_FOLDER']):
        if file.endswith('_model.pkl'):
            model_path = os.path.join(app.config['MODEL_FOLDER'], file)
            break
    model = None
    if model_exists and model_path:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            return render_template('predict.html', feature_names=model_metadata['feature_names'], 
                                 model_exists=False, error=f'Error loading model: {str(e)}')
    
    if request.method == 'POST' and model_exists and model:
        # Get input values from form
        input_data = {}
        for feature in model_metadata['feature_names']:
            value = request.form.get(feature, '')
            input_data[feature] = value if value else ('' if model_metadata['is_text_data'] else 0)
        
        try:
            # Convert input data to the correct format
            if model_metadata['is_text_data']:
                vectorizer_path = os.path.join(app.config['MODEL_FOLDER'], 'vectorizer.pkl')
                if not os.path.exists(vectorizer_path):
                    return render_template('predict.html', feature_names=model_metadata['feature_names'], 
                                         model_exists=model_exists, error='Vectorizer not found. Please retrain the model.')
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                message = input_data.get('Message', '')
                if not message:
                    return render_template('predict.html', feature_names=model_metadata['feature_names'], 
                                         model_exists=model_exists, error='Please provide a message for prediction.')
                X = vectorizer.transform([message])
            else:
                input_df = pd.DataFrame([input_data])
                categorical_cols = input_df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    input_df = pd.get_dummies(input_df, columns=categorical_cols)
                
                # Align with training features
                input_df = input_df.reindex(columns=model_metadata['feature_names'], fill_value=0)
                
                # Load saved scaler
                scaler_path = os.path.join(app.config['MODEL_FOLDER'], 'scaler.pkl')
                if not os.path.exists(scaler_path):
                    X = input_df.values
                    if len(X.shape) == 1:
                        X = X.reshape(1, -1)
                else:
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    X = scaler.transform(input_df)
                    if len(X.shape) == 1:
                        X = X.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(X)[0]
            if model_metadata['is_text_data']:
                prediction = 'spam' if prediction == 1 else 'ham'
            return render_template('predict.html', prediction=prediction, feature_names=model_metadata['feature_names'], 
                                 model_exists=model_exists)
        except Exception as e:
            return render_template('predict.html', feature_names=model_metadata['feature_names'], 
                                 model_exists=model_exists, error=f'Prediction error: {str(e)}')
    
    return render_template('predict.html', feature_names=model_metadata['feature_names'], model_exists=model_exists)

if __name__ == '__main__':
    app.run(debug=True)