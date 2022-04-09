import os
import glob
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from bayes_opt import BayesianOptimization
import plotly as plt
import plotly.express as px
import numpy as np
from tqdm import tqdm
import shap

def csv_to_df(path):
    all_files = glob.glob(os.path.join(path, "*.csv"))

    df = pd.DataFrame()
    for f in tqdm(all_files):
        row = pd.read_csv(f, sep=',', names=['filename', 'site', 'size', 'footprint', 'height', 'num_stories', 'num_units',
                                         'num_adiabatic', 'inf_rate', 'orientation', 'wwr', 'frame', 'polyiso_t', 'cellulose_t',
                                         'setback', 'rear_setback', 'side_setback', 'structure_setback', 'assembly_r',
                                         'area_buildable', 'surf_tot', 'surf_glaz', 'surf_opaq', 'volume', 'surf_vol_ratio', 'cooling',
                                         'heating', 'lighting', 'equipment', 'water', 'eui_kwh', 'eui_kbtu', 'carbon', 'kg_CO2e'])
        df = df.append(row, ignore_index=True)
    return df


def pickle_df(data):
    pckl = data.to_pickle('./energy_data.pkl')
    # pickle_out = open('energy_data.pkl', 'wb')
    # pickle.dump(data, pickle_out)
    # pickle_out.close()
    print('Dataframe has been pickled')
    return pckl


def plot_feature_importance(model):
    xgb.plot_importance(model)
    plt.rcParams['figure.figsize'] = [5, 5]
    fig = plt.show()
    return fig


def plot_prediction_analysis(y, y_pred):
    fig = px.scatter(x=y, y=y_pred, labels={
                     'x': 'ground truth', 'y': 'prediction'})
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y.min(), y0=y.min(),
        x1=y.max(), y1=y.max()
    )
    return fig


def prepare_data(df):
    data['surf_vol_ratio'] = data['surf_tot'] / data['volume']
    labels_drop = ['filename', 'num_adiabatic', 'frame', 'polyiso_t', 'cellulose_t', 
                   'rear_setback', 'side_setback', 'structure_setback', 'area_buildable',
                   'surf_tot', 'surf_glaz', 'surf_opaq', 'volume', 'cooling', 'heating', 
                   'lighting', 'equipment', 'water', 'eui_kwh', 'carbon', 'kg_CO2e']

    data.drop(labels=labels_drop, axis=1, inplace=True)
    print(data.shape)
    print(data.head())

    X, y = data.iloc[:,:-1], data.iloc[:,-1]

    # train xgboost model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    prepared_data = {'dataframe': data,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test}
    
    return prepared_data


def xgboost_regression(prepared_data, loss='rmse', random_search=False, grid_search=False, bay_opt=False):

    X_train = prepared_data['X_train']
    X_test = prepared_data['X_test']
    y_train = prepared_data['y_train']
    y_test = prepared_data['y_test']
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    
    def xgb_evaluate(max_depth, gamma, colsample_bytree):
        params_ = {'eval_metric': 'rmse',
                'max_depth': int(max_depth),
                'subsample': 0.8,
                'eta': 0.1,
                'gamma': gamma,
                'colsample_bytree': colsample_bytree}

        cv_result = xgb.cv(params_, dtrain, num_boost_round=100, nfold=3)    
        
        # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
        return -1.0 * cv_result['test-rmse-mean'].iloc[-1]
    
    if loss == 'rmse':
        scoring = 'neg_mean_squared_error'
    elif loss == 'mae':
        scoring = 'neg_mean_absolute_error'
    
    if (grid_search and random_search) or (grid_search and bay_opt) or (random_search and bay_opt):
        return print('error: please select only one optimization method')
    
    model = xgb.XGBRegressor()
    params = {
        "max_depth": np.arange(3, 50, 1), 
        "gamma": np.arange(0, 20, 1), 
        "lambda": np.arange(0, 1, .1)
        }
    
    if random_search:
        randomized_search = RandomizedSearchCV(model, params, n_iter=20, 
                                        scoring=scoring, cv=3, verbose=3)
        randomized_search.fit(X_train, y_train)
        xgboost_reg = randomized_search.best_estimator_
        preds = xgboost_reg.predict(X_test)
    
    if grid_search:
        gridded_search = GridSearchCV(model, params, scoring=scoring, verbose=3)
        gridded_search.fit(X_train, y_train)
        xgboost_reg = gridded_search.best_estimator_
        preds = xgboost_reg.predict(X_test)
        
    if bay_opt:
        bay_optimization = BayesianOptimization(xgb_evaluate, {
                                            'max_depth': (3, 7), 
                                             'gamma': (0, 1),
                                             'colsample_bytree': (0.3, 0.9)},
                                            random_state=9,
                                            verbose=3)
        bay_optimization.maximize(init_points=3, n_iter=10, acq='ei')
        # params = bay_optimization.res['max']['max_params']
        targets = []
        for i, rs in enumerate(bay_optimization.res):
            targets.append(rs["target"])
        best_params = bay_optimization.res[targets.index(max(targets))]["params"]
        best_params["max_depth"] = int(best_params["max_depth"])
        print(best_params)
        # best_params = bay_optimization.max['params']
        # best_params['max_depth'] = int(params['max_depth'])
        # print(best_params)
        xgboost_reg = xgb.train(best_params, dtrain, num_boost_round=250)
        preds = xgboost_reg.predict(dtest)
        
    # mae loss
    if loss == 'mae':
        mae = mean_absolute_error(y_test, preds)
        print("MAE: %f" % (mae))
    
    # rmse loss
    if loss == 'rmse':
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print("RMSE: %f" % (rmse))

    if pickle:
        input("press enter if you are happy with error and would like to pickle model")

        pickle_out = open('xgboost_reg.pkl','wb')
        pickle.dump(xgboost_reg, pickle_out)
        pickle_out.close()
        print('Model has been pickled')

    return xgboost_reg
    

if __name__ == '__main__':
    for i in range(50):
        print('')
    
    # read csvs and combine
    # data = csv_to_df('/Users/preston/Documents/GitHub/msdc-thesis/tool/temp')
    # data.to_csv('all_sims.csv')
    
    # read combined csv into df
    data = pd.read_csv('all_sims.csv')
    data.drop('Column1', axis=1, inplace=True)
    prepared_data = prepare_data(data)
    
    model = xgboost_regression(prepared_data, loss='rmse', random_search=False, grid_search=False, bay_opt=True)
 

    X_train = prepared_data['X_train']
    X_test = prepared_data['X_test']
    y_train = prepared_data['y_train']
    y_test = prepared_data['y_test']
    
    # print(X_train.columns)
    X_test_DM = xgb.DMatrix(X_test)
    y_preds = model.predict(X_test_DM)
    
    # SHAP plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    shap.summary_plot(shap_values, X_test)
    
    # Plotly scatter y=x
    fig = px.scatter(x=y_test, y=y_preds, labels={
                     'x': 'ground truth', 'y': 'prediction'})
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max())
    fig.show()
    
        
    fig1 = plot_prediction_analysis(y_test, y_preds)
    fig1.show
    
    # XGB feature importance (F scores)
    fig2 = plot_feature_importance(model)
    fig2.show
