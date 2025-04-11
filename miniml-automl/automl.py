def run_automl_pipeline(df, target):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_squared_error
    import numpy as np

    y = df[target]
    X = df.drop(columns=[target])

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)
        problem_type = "classification"
        model = RandomForestClassifier()
    else:
        problem_type = "regression"
        model = RandomForestRegressor()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if problem_type == "classification":
        score = accuracy_score(y_test, y_pred)
    else:
        score = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "problem_type": problem_type,
        "score": score,
        "features_used": list(X.columns),
    }