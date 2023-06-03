import sklearn
from flask import Flask, render_template, url_for ,request
import pandas as pd
from tabulate  import tabulate
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from  sklearn.neural_network import MLPClassifier
import numpy as np
import  matplotlib.pyplot as plt
import os
import  seaborn as sns
import pickle
app = Flask(__name__ )
data = pd.read_csv("heartnew.csv").head()
#print(data.columns)
data1 = pd.read_csv("heartnew.csv")
pd1 = pd.DataFrame(data)
desc = data1.describe()
x = data1.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11]].values
y = data1.iloc[:, 13].values
xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size = 0.25 , random_state=0)
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
#use Evaluation predict
#xtest = sc_x.transform(xtest)
scaler_file = 'scalerNN.sav'
sc_x = pickle.load(open(scaler_file, 'rb'))
#scaler_file = 'scaler_SVM.sav'
sc_x_svm = pickle.load(open('scaler_SVM.sav', 'rb'))
#xtest = sc_x.transform(xtest)
xlenth=0
#print (xtrain[0:10, :])
rex = tabulate(pd1 ,headers='keys', tablefmt="html" , showindex="always"  )
rex1 = tabulate(desc ,headers='keys',tablefmt="html" , showindex="always")
rex2 = tabulate(xtrain[0:10, :] , headers='keys',tablefmt='html',showindex='always')
@app.route("/") # like index
def home():
    return render_template('index.html')
@app.route("/Busness")
def Busness():
    target = data1['output'].value_counts()
    sex = data1['sex'].value_counts()
    return render_template('Describe.html', tables= [rex], titles=rex.title(),tables1=[rex1]
                           ,titles1=rex1.title(), ctitles1=rex1.title() ,
                           output =target , sexoutput = sex )

@app.route("/DataUnderstand")
def DataUnderstand():
    return render_template('Explain.html')

@app.route("/Preprocssing")
def Preprocssing():
    return render_template('Preprocssing.html' , tables= [rex2], titles=[rex2.title()])
#==========================================================
@app.route('/predict1', methods=['GET', 'POST'])
def predict1():
    output=''
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        xlenth=len(int_features)
        # print("intial values -->", int_features)
        pre_final_features = [np.array(int_features)]
        final_features = sc_x.transform(pre_final_features)
        mpdel_file = 'model_NN.sav'
        classifier = pickle.load(open("model_NN.sav", 'rb'))
        prediction = classifier.predict(final_features)
        print('predictio value is ', prediction[0])
        if (prediction[0] == 1):
            output = "True"
        elif (prediction[0] == 0):
                output = "False"
        else:
         output = "Not sure"
    return render_template('ModelNureal.html', prediction_text= format(output))
#========================================Predict SVM========================================================
@app.route('/predictSVM', methods=['GET', 'POST'])
def predictSVM():
    output=''
    if request.method == 'POST':

        int_features = [int(x) for x in request.form.values()]
        pre_final_features = [np.array(int_features)]
        final_features = sc_x_svm.transform(pre_final_features)
        classifier_svm = pickle.load(open("model_SVM.sav", 'rb'))
        prediction = classifier_svm.predict(final_features)
        print('predictio value is ', prediction[0])
        if (prediction[0] == 1):
            output = "True"
        elif (prediction[0] == 0):
                output = "False"
        else:
         output = "Not sure"
    return render_template('ModelSVM.html', prediction_text= format(output))
@app.route('/Evalution', methods=['GET', 'POST'])
def Evalution():
    x = data1.iloc[:, [0, 12]].values
    y = data1.iloc[:, 13].values
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)
    sc_x = StandardScaler()
    xtrain = sc_x.fit_transform(xtrain)
    xtest = sc_x.transform(xtest)
    classifierNN = sklearn.neural_network.MLPClassifier(
        activation='logistic',
        max_iter=1000,
        hidden_layer_sizes=(2,),  # default (100,)
        solver='lbfgs')
    classifierNN.fit(xtrain, ytrain)
    ypred = classifierNN.predict(xtest)
    print("Accuracy : ", accuracy_score(ytest, ypred))
    # -----------------------------------------------------------------------------------
    best_score = 1000
    kfolds = 5
    for c in [0.01, 0.1, 0.3, 1, 2, 10, 100]:
        model = classifierNN
        scores = cross_val_score(model, xtrain, ytrain, cv=kfolds)
        score = np.mean(scores)  # average every chunk
        score = 1 / score
        print("score=", score, ' ==> ', c)
        if score < best_score:
            best_score = score
            best_parameters = c
    # -----------------------------------------------------------------
    SelectedModel = classifierNN.fit(xtrain, ytrain)
    test_score = SelectedModel.score(xtest, ytest)
    print("Best score on validation set is:", best_score)
    print("Best parameter for regularization (lambda) is:", best_parameters)
    print("Test set score with best C parameter is", test_score)
    # -----------------------------------------------------------------
    prediction_proba = SelectedModel.predict(xtest)
    print("accuracy_score", accuracy_score(ytest, ypred))
    print("Prediction: ", prediction_proba)
    print('Best score on validation set is: ', best_score)
    print('Best parameter for regularization (lambda) is: ', best_parameters)
    print('Test set score with best C parameter is: ', test_score)
    #============Predict SVM =============================
    svc = SVC()
    parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
                  {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4],
                   'gamma': [0.01, 0.02, 0.03, 0.04, 0.05]}]
    grid_search = GridSearchCV(estimator=svc, param_grid=parameters, scoring='accuracy', cv=5, verbose=0)
    grid_search.fit(xtrain, ytrain)
    print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))
    # print parameters that give the best results
    print('Parameters that give the best results :', '\n\n', (grid_search.best_params_))
    # print estimator that was chosen by the GridSearch
    print('\n\nEstimator that was chosen by the search :', '\n\n', (grid_search.best_estimator_))
    # ========================result=====================================================
    print("Used SVM to predict")
    linear_svc = SVC(kernel='linear', C=10)
    linear_svc.fit(xtrain, ytrain)
    y_predSVM = linear_svc.predict(xtest)
    print("*************")
    print('Model accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(ytest,y_predSVM)))
    return render_template('Evalution.html',best_score = best_score,best_parameters=best_parameters,
                            test_score= test_score, accuracyNN=accuracy_score(ytest,ypred),
                            GridSearchCVbestscore=grid_search.best_score_ , Parameters_the_best_results =grid_search.best_params_,
                            Estimator_that_was_chosen_by_the_search =grid_search.best_estimator_,
                            accuracy_scoreSVM = accuracy_score(ytest,y_predSVM) )

#============================================================================================================
if __name__ == '__main__':
    app.run(host='localhost', port=5545, debug=True)
    #app.run()
