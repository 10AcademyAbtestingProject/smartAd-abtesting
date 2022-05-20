from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             mean_squared_error,
                             r2_score,
                             mean_absolute_error,
                             log_loss,
                             precision_score,
                             recall_score)

import mlflow
import time

from Scripts.cleaning import CleanDataFrame


class TrainingPipeline(Pipeline):
    '''
    Class -> TrainingPipeline, ParentClass -> Sklearn-Pipeline
    Extends from Scikit-Learn Pipeline class. Additional functionality to track 
    model metrics and log model artifacts with mlflow
    params:
    steps: list of tuple (similar to Scikit-Learn Pipeline class)
    '''

    def __init__(self, steps):
        super().__init__(steps)

    def fit(self, X_train, y_train):
        self.__pipeline = super().fit(X_train, y_train)
        return self.__pipeline

    def get_metrics(self, y_true, y_pred, y_pred_prob):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        entropy = log_loss(y_true, y_pred_prob)
        return {
            'accuracy': round(acc, 2),
            'precision': round(prec, 2),
            'recall': round(recall, 2),
            'entropy': round(entropy, 2)
        }

    def make_model_name(self, experiment_name, run_name):
        clock_time = time.ctime().replace(' ', '-')
        return experiment_name + '_' + run_name + '_' + clock_time

    def log_model(self, model_key, X_test, y_test, experiment_name, run_name, run_params=None):
        model = self.__pipeline.get_params()[model_key]
        y_pred = self.__pipeline.predict(X_test)
        y_pred_prob = self.__pipeline.predict_proba(X_test)
        run_metrics = self.get_metrics(y_test, y_pred, y_pred_prob)
        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri('http://localhost:5000')
        with mlflow.start_run(run_name=run_name):
            if run_params:
                for name in run_params:
                    mlflow.log_param(name, run_params[name])
            for name in run_metrics:
                mlflow.log_metric(name, run_metrics[name])
            
            mlflow.log_param("columns", X_test.columns.to_list())
        model_name = self.make_model_name(experiment_name, run_name)
        mlflow.sklearn.log_model(
            sk_model=self.__pipeline, artifact_path='models', registered_model_name=model_name)
        print('Run - %s is logged to Experiment - %s' %
              (run_name, experiment_name))
        return run_metrics


def get_pipeline(model, x):
    cat_cols = CleanDataFrame.get_categorical_columns(x)
    num_cols = CleanDataFrame.get_numerical_columns(
        x)   # Remove the target column

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=15))
    ])
    numerical_transformer = Pipeline(steps=[
        ('scale', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    train_pipeline = TrainingPipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return train_pipeline


def run_train_pipeline(model, x, y, experiment_name, run_name):
    '''
    function which executes the training pipeline
    Args:
        model : an sklearn model object
        x : features dataframe
        y : labels
        experiment_name : MLflow experiment name
        run_name : Set run name inside each experiment
    '''
    train_pipeline = get_pipeline(model, x)

    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.3,
                                                        random_state=123)
    run_params = model.get_params()
    train_pipeline.fit(X_train, y_train)
    return train_pipeline.log_model('model', X_test, y_test, experiment_name, run_name, run_params=run_params)
