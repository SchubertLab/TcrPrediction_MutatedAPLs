from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def train_classification_model(data_tcr, df_tcr, apls_train, idx):
    x_train = data_tcr[df_tcr['epitope'].isin(apls_train)]
    y_train = df_tcr[df_tcr['epitope'].isin(apls_train)]['is_activated']
    clf = RandomForestClassifier(random_state=idx).fit(x_train, y_train)
    return clf


def predict_classification_on_test(data_tcr, df_tcr, clf, apls_train):
    x_test = data_tcr[~df_tcr['epitope'].isin(apls_train)]
    y_test = df_tcr[~df_tcr['epitope'].isin(apls_train)]['is_activated']
    p_test = clf.predict_proba(x_test)
    p_test = p_test[:, (1 if p_test.shape[1] == 2 else 0)]
    return p_test, y_test


def train_regression_model(data_tcr, df_tcr, apls_train, idx):
    x_train = data_tcr[df_tcr['epitope'].isin(apls_train)]
    y_train = df_tcr[df_tcr['epitope'].isin(apls_train)]['activation']
    reg = RandomForestRegressor(n_estimators=250, max_features='sqrt',
                                criterion='mae', random_state=idx).fit(x_train, y_train)
    return reg

def predict_regression_on_test(data_tcr, df_tcr, reg, apls_train):
    x_test = data_tcr[~df_tcr['epitope'].isin(apls_train)]
    y_truth = df_tcr[~df_tcr['epitope'].isin(apls_train)]['activation']
    y_pred = reg.predict(x_test)
    return y_pred, y_truth


def evaluate_classification_models(y_test, p_test, metrics, results, idx):
    # evaluate the classifier for binary metrics
    y_pred = p_test > 0.5
    for name, metric in metrics[1].items():
        score = metric(y_test, y_pred)
        results[name].append([idx, score])

    # no auc can be formed if all data is from same class
    if sum(y_test) == len(y_test) or sum(y_test) == 0:
        return

    # evaluate the classifier for probability metrics
    for name, metric in metrics[0].items():
        score = metric(y_test, p_test)
        results[name].append([idx, score])
    return


def evaluate_regression_models(y_truth, y_test, metrics, results, idx):
    # evaluate the classifier for probability metrics
    for name, metric in metrics.items():
        score = metric(y_truth, y_test)
        results[name].append([idx, score])
    return
