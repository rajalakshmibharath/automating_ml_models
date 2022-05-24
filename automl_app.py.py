import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from io import StringIO


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from streamlit_option_menu import option_menu

import plotly
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from PIL import Image

st.set_page_config(layout="wide")

selected = option_menu(menu_title= None, options = ['Home','EDA','Model'],icons=['house-fill','bar-chart-line-fill','cpu-fill'],menu_icon='cast', orientation ='horizontal',styles={
        "container": {"padding": "0!important", "background-color": "#262730"},
        "icon": {"color": "white", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "grey"},
        "nav-link-selected": {"background-color": "blue"}})

if selected== 'Home':
    
    st.title('Automating ML Models')
    
    
    image = Image.open('C:/Data Science/datasets/ai.jpg')
    st.image(image, use_column_width = True)
    



if selected == 'EDA':
    
    st.title('Basic Exploratory Data Analysis')

    with st.sidebar.header('Choose a csv file '):
        uploaded_file = st.file_uploader("Choose a file")
        
        
    def basic_eda(df):
        with st.sidebar:
            
            st.markdown('Basic EDA')
            
            column_names = st.button('Column Names')
            describe = st.button('Describe')  
            #info = st.button('Info')
            #dtype= st.button('Datatypes')
            null = st.button('Null Values')
            #unique = st.button('Find Unique val in columns')
            
        
        if column_names:
             st.write('Columns',df.columns)
             
        if describe:
             st.write('Description of Data',df.describe())
            
        #if info:
             #st.write('Information',df.info())
        
        #if dtype:
            # st.write('Datatypes',df.dtypes())
        if null:     
            st.write('Null Values',df.isnull().sum())
            st.write(df.loc[:,df.isnull().any()].columns)
        
        #if unique:
            #col = st.selectbox('select a colum', df.columns)
           
            #st.write(df[col ])
            #st.write('Unique values',st.write(df[col].unique()))

        
        
    def chart(df):
        with st.sidebar:
            st.markdown('Charts')
            
            line = st.button('Line Chart')
            area = st.button('Area Chart')
            bar = st.button('Bar Chart')
            map = st.button('Scatter plot on map')
            his = st.button('Histogram')
           
        
        if line:
            st.line_chart(df)
        
        if area:
            st.area_chart(df)
        
        if bar:
            st.bar_chart(df)
            
        if map:
            st.map(st.map(df))
        
        if his:
            fig, ax = plt.subplots()
            ax.hist(df, bins=20)

            st.pyplot(fig)
        
        
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
            
        encoding='latin-1'
        df = pd.read_csv(uploaded_file, encoding= encoding)
        st.markdown('The file contains')
        st.write(df)
        st.write('The dataset has {} rows and {} columns'.format(df.shape[0],df.shape[1]))
        
        
        column_lis = []
        
        
        fig =plt.subplots()
        for i in df.columns:
            column_lis.append(i)
            hist = df.hist(i)
            
        
        basic_eda(df)
        chart(df)

#----------------------------------------------------------------------------------------------------------------------------#

def linear_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         st.markdown('After splitting training and testing datset')
         st.write('Training sample size')
         st.info(x_train.shape)
         st.write('Test sample size')
         st.info(x_test.shape)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         lin_reg = LinearRegression()
         lin_reg.fit(x_train,y_train)
         
         #train_score = lin_reg.score(x_train,y_train)
         #test_score = lin_reg.score(x_test,y_test)
         
         y_pred_train= lin_reg.predict(x_train)
         y_pred_test = lin_reg.predict(x_test)
         
         r2score_train = r2_score(y_train,y_pred_train)
         mse_train = mean_squared_error(y_train,y_pred_train)
         
         r2score_test = r2_score(y_test,y_pred_test)
         mse_test = mean_squared_error(y_test,y_pred_test)
         
         coeffecient = lin_reg.coef_
         intercept = lin_reg.intercept_
                
         
         st.subheader('Model Performance')
         st.markdown('*Training set*')
         
         #st.write('Training score:',train_score )
         
         
         st.write('R2 score in train data :',r2score_train )
         st.write('mse in train data  :',mse_train )

         st.markdown('**Test set**')
         
         #st.write('Test score  :', test_score )
         st.write('R2 score in test data :',r2score_test )
         st.write('mse in test data  :',mse_test)
          
         st.markdown('**Coeffecient and Intercept**')
         
         st.write('Coeffecient :', coeffecient )
         st.write('Intercept :', intercept)
         
         
        
#-----------------------------------------------------------------------------------------------------------------------------------------#



def lasso_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
         lasso_reg = Lasso(alpha= a)
         lasso_reg.fit(x_train,y_train)
        
         y_pred_train= lasso_reg.predict(x_train)
         y_pred_test = lasso_reg.predict(x_test)
         
         r2score_train = r2_score(y_train,y_pred_train)
         mse_train = mean_squared_error(y_train,y_pred_train)
         
         r2score_test = r2_score(y_test,y_pred_test)
         mse_test = mean_squared_error(y_test,y_pred_test)
         
         coeffecient = lasso_reg.coef_
         intercept = lasso_reg.intercept_
                
         
         st.subheader('Model Performance')
         st.markdown('**Training set**')
        
         st.write('R2 score in train data :',r2score_train )
         st.write('mse in train data  :',mse_train )

         st.markdown('**Test set**')
         
         
         st.write('R2 score in test data :',r2score_test )
         st.write('mse in test data  :',mse_test)
          
         st.markdown('**Coeffecient and Intercept**')
         
         st.write('Coeffecient :', coeffecient )
         st.write('Intercept :', intercept)


#-----------------------------------------------------------------------------------------------------------------------------------------#
def knn_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
         
             
         knn_reg = KNeighborsRegressor(n_neighbors= k)
         knn_reg.fit(x_train,y_train)
         
         y_pred_train= knn_reg.predict(x_train)
         y_pred_test = knn_reg.predict(x_test)
         
         r2score_train = r2_score(y_train,y_pred_train)
         mse_train = mean_squared_error(y_train,y_pred_train)
         
         r2score_test = r2_score(y_test,y_pred_test)
         mse_test = mean_squared_error(y_test,y_pred_test)
                
         
         st.subheader('Model Performance')
         st.markdown('**Training set**')
         
         st.write('R2 score in train data :',r2score_train )
         st.write('mse in train data  :',mse_train )

         st.markdown('**Test set**')
         
         st.write('R2 score in test data :',r2score_test )
         st.write('mse in test data  :',mse_test)


         

    

    
#-------------------------------------------------------------------------------------------------------------------------------------------#
       

def svm_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
         
             
         svr_reg = SVR(C= c, epsilon= e)
         svr_reg.fit(x_train,y_train)
         
         y_pred_train= svr_reg.predict(x_train)
         y_pred_test = svr_reg.predict(x_test)
         
         r2score_train = r2_score(y_train,y_pred_train)
         mse_train = mean_squared_error(y_train,y_pred_train)
         
         r2score_test = r2_score(y_test,y_pred_test)
         mse_test = mean_squared_error(y_test,y_pred_test)
                
         
         st.subheader('Model Performance')
         st.markdown('**Training set**')
         
         st.write('R2 score in train data :',r2score_train )
         st.write('mse in train data  :',mse_train )

         st.markdown('**Test set**')
         
         st.write('R2 score in test data :',r2score_test )
         st.write('mse in test data  :',mse_test)
    
#----------------------------------------------------------------------------------------------------------------------------------------#

# Decision Tree Regression

def dt_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
         
             
         dt_reg = DecisionTreeRegressor(
                                        max_depth = max_depth1,
                                        min_samples_split = min_samples_split,
                                        min_samples_leaf= min_samples_leaf,
                                        max_leaf_nodes= max_leaf_nodes,
                                            )
         dt_reg.fit(x_train,y_train)
         
         y_pred_train= dt_reg.predict(x_train)
         y_pred_test = dt_reg.predict(x_test)
         
         r2score_train = r2_score(y_train,y_pred_train)
         mse_train = mean_squared_error(y_train,y_pred_train)
         
         r2score_test = r2_score(y_test,y_pred_test)
         mse_test = mean_squared_error(y_test,y_pred_test)
                
         
         st.subheader('Model Performance')
         st.markdown('**Training set**')
         
         st.write('R2 score in train data :',r2score_train )
         st.write('mse in train data  :',mse_train )

         st.markdown('**Test set**')
         
         st.write('R2 score in test data :',r2score_test )
         st.write('mse in test data  :',mse_test)
    
#--------------------------------------------------------------------------------------------------------------------------------------------#

def rf_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
            
         rf_reg = RandomForestRegressor(n_estimators= n_estimators ,
                                        #criterion = criteria,
                                        max_depth = max_d,
                                        min_samples_split = min_samples_split,
                                        min_samples_leaf = min_samples_leaf,
                                        bootstrap = bootstrap ,
                                        oob_score = oob_score
                                        )
         rf_reg.fit(x_train,y_train)
         
         y_pred_train= rf_reg.predict(x_train)
         y_pred_test = rf_reg.predict(x_test)
         
         r2score_train = r2_score(y_train,y_pred_train)
         mse_train = mean_squared_error(y_train,y_pred_train)
         
         r2score_test = r2_score(y_test,y_pred_test)
         mse_test = mean_squared_error(y_test,y_pred_test)
                
         
         st.subheader('Model Performance')
         st.markdown('**Training set**')
         
         st.write('R2 score in train data :',r2score_train )
         st.write('mse in train data  :',mse_train )

         st.markdown('**Test set**')
         
         st.write('R2 score in test data :',r2score_test )
         st.write('mse in test data  :',mse_test)

#-------------------------------------------------------------------------------------------------------------------------------------#

def xgb_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
            
         xgb_reg = XGBRegressor(n_estimators= estimator, max_depth= max_d , eta= lr, subsample= ss, colsample_bytree=0.8
                                        )
         xgb_reg.fit(x_train,y_train)
         
         y_pred_train= xgb_reg.predict(x_train)
         y_pred_test = xgb_reg.predict(x_test)
         
         r2score_train = r2_score(y_train,y_pred_train)
         mse_train = mean_squared_error(y_train,y_pred_train)
         
         r2score_test = r2_score(y_test,y_pred_test)
         mse_test = mean_squared_error(y_test,y_pred_test)
                
         
         st.subheader('Model Performance')
         st.markdown('**Training set**')
         
         st.write('R2 score in train data :',r2score_train )
         st.write('mse in train data  :',mse_train )

         st.markdown('**Test set**')
         
         st.write('R2 score in test data :',r2score_test )
         st.write('mse in test data  :',mse_test)
         
#------------------------------------------------------------------------------------------------------------------------------------------------#

def gb_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
            
         gb_reg = GradientBoostingRegressor(n_estimators= estimator, max_depth= max_d , min_samples_split = mss, learning_rate= lr1)
         gb_reg.fit(x_train,y_train)
         
         y_pred_train= gb_reg.predict(x_train)
         y_pred_test = gb_reg.predict(x_test)
         
         r2score_train = r2_score(y_train,y_pred_train)
         mse_train = mean_squared_error(y_train,y_pred_train)
         
         r2score_test = r2_score(y_test,y_pred_test)
         mse_test = mean_squared_error(y_test,y_pred_test)
                
         
         st.subheader('Model Performance')
         st.markdown('**Training set**')
         
         st.write('R2 score in train data :',r2score_train )
         st.write('mse in train data  :',mse_train )

         st.markdown('**Test set**')
         
         st.write('R2 score in test data :',r2score_test )
         st.write('mse in test data  :',mse_test)
         
#--------------------------------------------------------------------------------------------------------------------------------------#
#classification models

def log_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
            
         log_cls = LogisticRegression(solver = solvers, penalty = penaltys, C= c)
         log_cls.fit(x_train,y_train)
         
         y_pred_train= log_cls.predict(x_train)
         y_pred_test = log_cls.predict(x_test)
         
         training_accuracy = accuracy_score(y_train,y_pred_train)
         test_accuracy = accuracy_score(y_test, y_pred_test )
         
         confusion = confusion_matrix(y_test, y_pred_test )
         
         cls_report = classification_report(y_test, y_pred_test )
         
         
         
         
                
         
         st.subheader('Model Performance')
        
         
         st.write('Accuracy score in training data :', training_accuracy )
         st.write('Accuracy score in test data:',test_accuracy )
         st.write('Confusion matrix  :', confusion )
         st.write('Classification report   :', cls_report)

#-----------------------------------------------------------------------------------------------------------------------------------------#


def sgd_clf_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
            
         sgd_cls = SGDClassifier(loss=losses , penalty= penalties, max_iter= max_i)     
         sgd_cls.fit(x_train,y_train)
         
         y_pred_train= sgd_cls.predict(x_train)
         
         y_pred_test = sgd_cls.predict(x_test)
         
         training_accuracy = accuracy_score(y_train,y_pred_train)
         test_accuracy = accuracy_score(y_test, y_pred_test )
         
         confusion = confusion_matrix(y_test, y_pred_test )
         
         cls_report = classification_report(y_test, y_pred_test )
         
         
         
         
                
         
         st.subheader('Model Performance')
        
         
         st.write('Accuracy score in training data :', training_accuracy )
         st.write('Accuracy score in test data:',test_accuracy )
         st.write('Confusion matrix  :', confusion )
         st.write('Classification report   :', cls_report)

#--------------------------------------------------------------------------------------------------------------------------------------------#

def nb_clf_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
            
         nb_cls = GaussianNB()     
         nb_cls.fit(x_train,y_train)
         
         y_pred_train= nb_cls.predict(x_train)
         
         y_pred_test = nb_cls.predict(x_test)
         
         training_accuracy = accuracy_score(y_train,y_pred_train)
         test_accuracy = accuracy_score(y_test, y_pred_test )
         
         confusion = confusion_matrix(y_test, y_pred_test )
         
         cls_report = classification_report(y_test, y_pred_test )
         
         
         
         
                
         
         st.subheader('Model Performance')
        
         
         st.write('Accuracy score in training data :', training_accuracy )
         st.write('Accuracy score in test data:',test_accuracy )
         st.write('Confusion matrix  :', confusion )
         st.write('Classification report   :', cls_report)
         
         
#------------------------------------------------------------------------------------------------------------------------------------------#


def dt_clf_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
            
         dt_cls =  DecisionTreeClassifier(criterion = criter, max_depth= max_d,
                                          min_samples_split = min_samples_split,
                                          min_samples_leaf= min_samples_leaf,
                                          max_leaf_nodes= max_leaf_nodes)     
         dt_cls.fit(x_train,y_train)
         
         y_pred_train= dt_cls.predict(x_train)
         
         y_pred_test = dt_cls.predict(x_test)
         
         training_accuracy = accuracy_score(y_train,y_pred_train)
         test_accuracy = accuracy_score(y_test, y_pred_test )
         
         confusion = confusion_matrix(y_test, y_pred_test )
         
         cls_report = classification_report(y_test, y_pred_test )
         
         
         
         
                
         
         st.subheader('Model Performance')
        
         
         st.write('Accuracy score in training data :', training_accuracy )
         st.write('Accuracy score in test data:',test_accuracy )
         st.write('Confusion matrix  :', confusion )
         st.write('Classification report   :', cls_report)
                 
#-------------------------------------------------------------------------------------------------------------------------------------------#

def rf_clf_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
            
         rf_cls = RandomForestClassifier(n_estimators= n_estimators ,
                                        criterion = criter,
                                        max_depth = max_d,
                                        min_samples_split = min_samples_split,
                                        min_samples_leaf = min_samples_leaf,
                                        bootstrap = bootstrap ,
                                        oob_score = oob_score )     
         rf_cls.fit(x_train,y_train)
         
         y_pred_train= rf_cls.predict(x_train)
         
         y_pred_test = rf_cls.predict(x_test)
         
         training_accuracy = accuracy_score(y_train,y_pred_train)
         test_accuracy = accuracy_score(y_test, y_pred_test )
         
         confusion = confusion_matrix(y_test, y_pred_test )
         
         cls_report = classification_report(y_test, y_pred_test )
         
         
         
         
                
         
         st.subheader('Model Performance')
        
         
         st.write('Accuracy score in training data :', training_accuracy )
         st.write('Accuracy score in test data:',test_accuracy )
         st.write('Confusion matrix  :', confusion )
         st.write('Classification report   :', cls_report)
#--------------------------------------------------------------------------------------------------------------------------------------------#

def svc_clf_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
           
         svc_cls = SVC(C= c, kernel= kernel,gamma = gamma)     
         svc_cls.fit(x_train,y_train)
         
         y_pred_train= svc_cls.predict(x_train)
         
         y_pred_test = svc_cls.predict(x_test)
         
         training_accuracy = accuracy_score(y_train,y_pred_train)
         test_accuracy = accuracy_score(y_test, y_pred_test )
         
         confusion = confusion_matrix(y_test, y_pred_test )
         
         cls_report = classification_report(y_test, y_pred_test )
         
         
         
         
                
         
         st.subheader('Model Performance')
        
         
         st.write('Accuracy score in training data :', training_accuracy )
         st.write('Accuracy score in test data:',test_accuracy )
         st.write('Confusion matrix  :', confusion )
         st.write('Classification report   :', cls_report)



#------------------------------------------------------------------------------------------------------------------------------------------#

def knn_clf_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
           
         knn_cls = KNeighborsClassifier(n_neighbors = k, metric = metric, weights= wghts)   
         knn_cls.fit(x_train,y_train)
         
         y_pred_train= knn_cls.predict(x_train)
         
         y_pred_test = knn_cls.predict(x_test)
         
         training_accuracy = accuracy_score(y_train,y_pred_train)
         test_accuracy = accuracy_score(y_test, y_pred_test )
         
         confusion = confusion_matrix(y_test, y_pred_test )
         
         cls_report = classification_report(y_test, y_pred_test )
         
         
         
         
                
         
         st.subheader('Model Performance')
        
         
         st.write('Accuracy score in training data :', training_accuracy )
         st.write('Accuracy score in test data:',test_accuracy )
         st.write('Confusion matrix  :', confusion )
         st.write('Classification report   :', cls_report)

#------------------------------------------------------------------------------------------------------------------------------------------------#

def xgb_clf_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
           
         xgb_cls = XGBClassifier( learning_rate = lr, gamma = gamma , reg_alpha= reg_alpha, reg_lambda= reg_lambda,
                                 max_depth=max_depth, subsample = subsample, colsample_bytree= colsample_bytree )   
         xgb_cls.fit(x_train,y_train)
         
         y_pred_train= xgb_cls.predict(x_train)
         
         y_pred_test = xgb_cls.predict(x_test)
         
         training_accuracy = accuracy_score(y_train,y_pred_train)
         test_accuracy = accuracy_score(y_test, y_pred_test )
         
         confusion = confusion_matrix(y_test, y_pred_test )
         
         cls_report = classification_report(y_test, y_pred_test )
         
         
         
         
                
         
         st.subheader('Model Performance')
        
         
         st.write('Accuracy score in training data :', training_accuracy )
         st.write('Accuracy score in test data:',test_accuracy )
         st.write('Confusion matrix  :', confusion )
         st.write('Classification report   :', cls_report)
#-------------------------------------------------------------------------------------------------------------------------------------------#

def gb_clf_model(df):
     
     if name:
         x = df.drop(name,axis = 1)
         y = df[name]
         
         st.markdown('After splitting X and Y')
         st.write('Dependent Variable ( X )')
         st.info(list(x.columns))
         st.write('Independent Variable ( Y )')
         st.info(y.name)
     # splitting dependent and independent features
         x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2)
         
         scaling= st.radio('Scale the data points ?',('No','Yes'))
         if scaling=='Yes':
             scale = StandardScaler()
             x_train = scale.fit_transform(x_train)
             x_test = scale.transform(x_test)
         else:
             pass
         
          
           
         gb_cls = GradientBoostingClassifier( learning_rate= learning_rate, n_estimators= n_estimators,
                                             subsample = subsample , min_samples_split= min_samples_split,
                                             max_depth = max_depth)   
         gb_cls.fit(x_train,y_train)
         
         y_pred_train= gb_cls.predict(x_train)
         
         y_pred_test = gb_cls.predict(x_test)
         
         training_accuracy = accuracy_score(y_train,y_pred_train)
         test_accuracy = accuracy_score(y_test, y_pred_test )
         
         confusion = confusion_matrix(y_test, y_pred_test )
         
         cls_report = classification_report(y_test, y_pred_test )
         
         
         
         
                
         
         st.subheader('Model Performance')
        
         
         st.write('Accuracy score in training data :', training_accuracy )
         st.write('Accuracy score in test data:',test_accuracy )
         st.write('Confusion matrix  :', confusion )
         st.write('Classification report   :', cls_report)


#-------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------#

if selected == 'Model':

    with st.sidebar.header('Choose a csv file '):
        uploaded_file = st.file_uploader("Choose a file")
        
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_file)
        st.markdown('The file contains')
        st.write(df)
        
        
        st.write('The dataset has {} rows and {} columns'.format(df.shape[0],df.shape[1]))
        
        
        #name = st.text_input('Choose your dependent feature')
        
        algorithm = st.radio('Type of algoeithm to solve',('Classification','Regression'))
        
        name = st.text_input('Choose your dependent feature')
        
        if algorithm == 'Regression':
        
            option = st.selectbox( 'Select a model to Predict',('<select model>','Linear Regression','Lasso Regression','SVM Regression','KNN Regression','DecisionTree Regression','Random Forest Regression','XgBoost','Gradient Boosting'),help= 'model')
            
            if option =='Linear Regression':
                linear_model(df)
                
                
            if option =='KNN Regression':
                st.write('Enter K value')
                    
                with st.sidebar:
                    k_ip= st.number_input('n_neighbors',5, help =" Number of neighbors to use by default for kneighbors queries. ")
                    k = int(k_ip)
                        
                knn_model(df)
                
            if option =='Lasso Regression':
                st.write('Enter the alpha value')
                    
                with st.sidebar:
                    a_ip= st.number_input('Specify alpha value',1.0,help ="Constant that multiplies the L1 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf).When alpha = 0, the objective is equivalent to ordinary least squares, solved by the LinearRegression object. For numerical reasons, using alpha = 0 with the Lasso object is not advised. Instead, you should use the LinearRegression object.")
                    a = int(float(a_ip))
                lasso_model(df)
                
            if option == 'SVM Regression':
                st.write('Enter C and alpha values')
                    
                with st.sidebar:
                    c_ip = st.number_input('Specify c value',1.0, help ="Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.")
                    c = int(float(c_ip))
                    
                    e_ip = st.number_input('Specify epsilon value',0.0,1.0,0.2, help = "Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.")
                    e = int(float(e_ip))
                    
                svm_model(df)
                
                
            if option == 'DecisionTree Regression':
                
                    
                with st.sidebar:
                   # cri = st.text_input('Specify criterion','absolute_error')
                    #criterion = cri
                    
                    
                    max_ip = st.number_input('max_depth', 2,help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
                    max_depth1= int(float(max_ip))
                    
                    min_s_ip = st.number_input('min_samples_split', 2, help ="The minimum number of samples required to split an internal node")
                    min_samples_split = int(float(min_s_ip ))   
                    
                    min_le_ip = st.number_input('min_samples_leaf', 2,help ="The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression")
                    min_samples_leaf = int(float(min_le_ip))
                    
                    max_le_nod_ip = st.number_input('max_leaf_nodes', 2, help ="Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes")
                    max_leaf_nodes = int(max_le_nod_ip )
                    
                    
                dt_model(df)
                
            
            if option == 'Random Forest Regression':
                
                    
                with st.sidebar:
                   # cri = st.text_input('Specify criterion','absolute_error')
                    #criterion = cri
                    
                    
                    n_estimators = st.slider('n_estimators)', 0,1000,100, help ="The number of trees in the forest.")
                    
                    #criteria = st.radio('Performance metrics', ('mse','mae'),'mse')
                    
                    max_dpth = st.number_input('max_depth',100, help = "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
                    max_d = int(float   (max_dpth))
                    
                    min_sam_spl = st.number_input('min_samples_split', 2, help ="The minimum number of samples required to split an internal node")
                    min_samples_split = int(float(min_sam_spl))
                    
                    min_s_le = st.number_input('min_samples_leaf', 1, help = "The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.")
                    min_samples_leaf = int(float(min_s_le))
                    
                    bootstrap = st.radio('bootstrap',('True','False'),True,help = "Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.")
                    
                    oob_score = st.radio('oob_score',('True','False'),False, help = "Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True.")
                    
                    
                rf_model(df)
                
            if option == 'XgBoost':
                
                    
                with st.sidebar:
                   # cri = st.text_input('Specify criterion','absolute_error')
                    #criterion = cri
                    
                    
                    estimator = st.slider('n_estimators', 0,1000,100, help ="The number of trees in the ensemble, often increased until no further improvements are seen.")
                    
                    #criteria = st.radio('Performance metrics', ('mse','mae'),'mse')
                    
                    max_dpth = st.number_input('max_depth',10, help = "The maximum depth of each tree, often values are between 1 and 10.")
                    max_d = int(float(max_dpth))
                    
                    lr_1= st.number_input('eta',0.01, help = "The learning rate used to weight each model, often set to small values such as 0.3, 0.1, 0.01, or smaller.")
                    lr = float(lr_1)
                    
                    sub_smpl = st.number_input('subsample',0.0,1.0,1.0, help = "The number of samples (rows) used in each tree, set to a value between 0 and 1, often 1.0 to use all samples.")
                    ss = float(sub_smpl)
                    
                    col_smpl = st.number_input('colsample_bytree', 0.0,1.0,1.0, help = "Number of features (columns) used in each tree, set to a value between 0 and 1, often 1.0 to use all features.")
                    cs= float(col_smpl)
                    
                xgb_model(df)
                
            if option == 'Gradient Boosting':
                
                    
                with st.sidebar:
                  
                    
                    
                    estimator = st.slider('n_estimators', 2,1000,100, help ="the number of boosting stages that will be performed. Later, we will plot deviance against boosting iterations.")
                    
                    
                    max_dpth = st.number_input(' max_depth',4, help ="limits the number of nodes in the tree. The best value depends on the interaction of the input variables.")
                    max_d = int(float(max_dpth))
                    
                    
                    min_smpl_spli = st.number_input('min_samples_split',5, help =" the minimum number of samples required to split an internal node.")
                    mss= int(float(min_smpl_spli))
                    
                    ll =  st.number_input('learning_rate',0.0,1.0,0.01, help ="how much the contribution of each tree will shrink.")
                    lr1 = float(11)
                
                   
                    
                    
                    #ls = st.text_input('loss function to optimize(loss )','squared_error')
                   
                    
                gb_model(df)
                   
    #--------------------------------------------------------------------------------------------------------------------------------------------#
    
    #Classification algorithms
      
         
            
        if algorithm == 'Classification':
            option = st.selectbox( 'Select a model to Predict',('<select model>','Logistic Regression','SGD Classifier','GaussianNB','KNeighbors Classifier','SVC','DecisionTree Classifier','RandomForest Classifier','XGB Classifier','Gradient_Boosting Classifier'))
            if option =='Logistic Regression':
                with st.sidebar:
                    
                    
                    solvers = st.radio('solvers',('newton-cg', 'lbfgs', 'liblinear') , help ='''Algorithm to use in the optimization problem. Default is ‘lbfgs’. To choose a solver, you might want to consider the following aspects:
    
    1. For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;
    
    2. For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;
    
    3. ‘liblinear’ is limited to one-versus-rest schemes.
    
    Warning : The choice of the algorithm depends on the penalty chosen: Supported penalties by solver:
    ‘newton-cg’ - [‘l2’, ‘none’]
    
    ‘lbfgs’ - [‘l2’, ‘none’]
    
    ‘liblinear’ - [‘l1’, ‘l2’]
    
    ‘sag’ - [‘l2’, ‘none’]
    
    ‘saga’ - [‘elasticnet’, ‘l1’, ‘l2’, ‘none’]''')
                    
                    penaltys = st.radio(' penalty:',('l2', 'l1  ', 'elasticnet', 'none'), help ='''Specify the norm of the penalty:
    'none': no penalty is added;
    
    'l2': add a L2 penalty term and it is the default choice;
    
    'l1': add a L1 penalty term;
    
    'elasticnet': both L1 and L2 penalty terms are added.
    
    Warning : Some penalties may not work with some solvers. See the parameter solver below, to know the compatibility between the penalty and solver.''')
                    
    
                    c_ip = st.number_input('C',1.0, help ="Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.")
                    c = int(float(c_ip))
                log_model(df)
                     
                     
                        
                         
                
                     
                    
                
        if option =='SGD Classifier':
            
            with st.sidebar:
                
                
                
                
                losses = st.radio('loss',('hinge','log_loss', 'modified_huber'), help ='''The loss function to be used.
    
    * ‘hinge’ gives a linear SVM.
    
    * ‘log_loss’ gives logistic regression, a probabilistic classifier.
    
    * ‘modified_huber’ is another smooth loss that brings tolerance to
    outliers as well as probability estimates.
    
    * ‘squared_hinge’ is like hinge but is quadratically penalized.
    
    * ‘perceptron’ is the linear loss used by the perceptron algorithm.
    
    * The other losses, ‘squared_error’, ‘huber’, ‘epsilon_insensitive’ and ‘squared_epsilon_insensitive’ are designed for regression but can be useful in classification as well; see SGDRegressor for a description.''')
                
                penalties = st.radio('penalty',('l1', 'l2', 'elasticnet', 'none'), help ="The penalty (aka regularization term) to be used. Defaults to ‘l2’ which is the standard regularizer for linear SVM models. ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’.")
                
    
                max_i = st.number_input('max_iter',5, help ="The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit method. Values must be in the range [1, inf).The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit method. Values must be in the range [1, inf).")
               
               
                
            sgd_clf_model(df)
            
        
        if option =='DecisionTree Classifier':
            
            with st.sidebar:
            
                criter = st.radio('criterion',('gini','entropy','log_loss') , help = "The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain")
                
                max_ip = st.number_input('max_depth',1, help = "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples." )
                max_d= int((max_ip))
                
                min_s_ip = st.number_input('min_samples_split', 2, help = '''The minimum number of samples required to split an internal node:
    
    1. If int, then consider min_samples_split as the minimum number.
    
    2. If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.''')
                min_samples_split = int(float(min_s_ip ))
                
                min_le_ip = st.number_input('min_samples_leaf', 2, help = '''The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
    
    1. If int, then consider min_samples_leaf as the minimum number.
    
    2. If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.''')
                min_samples_leaf = int(float(min_le_ip))
                
                max_le_nod_ip = st.number_input('max_leaf_nodes', 2, help = "Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.")
                max_leaf_nodes = int(max_le_nod_ip )
            
            dt_clf_model(df)
            
        if option =='RandomForest Classifier':
            
            with st.sidebar:
                 
                 
                 n_estimators = st.slider('n_estimators)', 0,1000,100, help ="The number of trees in the forest.")
                 
                 #criteria = st.radio('Performance metrics', ('mse','mae'),'mse')
                 criter = st.radio('criterion',('gini','entropy','log_loss') , help ="The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain, see Mathematical formulation. Note: This parameter is tree-specific.")
                 
                 max_dpth = st.number_input('max_depth',100, help ="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
                 max_d = int(float   (max_dpth))
                 
                 min_sam_spl = st.number_input('min_samples_split', 2, help ='''The minimum number of samples required to split an internal node:
    
    1. If int, then consider min_samples_split as the minimum number.
    
    2. If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.''')
                 min_samples_split = int(float(min_sam_spl))
                 
                 min_s_le = st.number_input('min_samples_leaf', 1, help ='''The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
    
    1. If int, then consider min_samples_leaf as the minimum number.
    
    2. If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.''')
                 min_samples_leaf = int(float(min_s_le))
                 
                 bootstrap = st.radio('bootstrap',('True','False'),True, help ="Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.")
                 
                 oob_score = st.radio('oob_score',('True','False'),False, help ="Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True")
                 
            
            rf_clf_model(df)
            
            
        if option =='SVC':
            
            with st.sidebar:
                
                c_ip = st.number_input('C',1.0, help = "Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.")
                c = int(float(c_ip))
                
                kernel= st.radio('kernel',('rbf','linear','poly','sigmoid'), help ="Specifies the kernel type to be used in the algorithm. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples)."  )
                gamma = st.radio('gamma',('scale','auto'), help ='''Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    
    1. if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
    
    2. if ‘auto’, uses 1 / n_features.''')
               
                 
            
            svc_clf_model(df)
            
        if option =='GaussianNB':
            
            nb_clf_model(df)
            
        if option =='KNeighbors Classifier':
            with st.sidebar:
                
                k_ip= st.number_input('n_neighbours',2 , help = "Number of neighbors to use by default for kneighbors queries."  )
                k = int(k_ip)
                
                metric = st.radio('metrics',('euclidean', 'manhattan', 'minkowski'), help = '''The distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric. For a list of available metrics, see the documentation of DistanceMetric and the metrics listed in sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS. Note that the “cosine” metric uses cosine_distances. If metric is “precomputed”, X is assumed to be a distance matrix and must be square during fit. X may be a sparse graph, in which case only “nonzero” elements may be considered neighbors.''' )
                
                wghts=st.radio('weights',('uniform','distance'), help ='''Weight function used in prediction. Possible values:
    
    1. ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
    
    2. ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
    
    3. [callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.''')
            
            knn_clf_model(df)
            
            
        if option =='XGB Classifier':
            with st.sidebar:
                
                ets= st.number_input('Learning rate',0.01, help ="also called eta, it specifies how quickly the model fits the residual errors by using additional base learners." )
                lr = int(float(ets))
                
                gamma = st.number_input('gamma',0 , help ="regularization done by XGBoost - minimum loss reduction to create a new split, L1 reg on leaf weights, L2 reg leaf weights respectively. typical values for gamma: 0 - 0.5 but highly dependent on the data")
                
                reg_alpha =st.number_input('reg_alpha',0 , help ="regularization done by XGBoost - minimum loss reduction to create a new split, L1 reg on leaf weights, L2 reg leaf weights respectively. typical values for reg_alpha : 0 - 1 is a good starting point but again, depends on the data")
                
                reg_lambda= st.number_input('reg_lambda',0 , help ="regularization done by XGBoost - minimum loss reduction to create a new split, L1 reg on leaf weights, L2 reg leaf weights respectively. typical values for reg_lambda: 0 - 1 is a good starting point but again, depends on the data")
                
                max_depth=st.number_input('max_depth',1 , help ="how deep the tree's decision nodes can go. Must be a positive integer")
                           
                ss = st.number_input('subsample',0.5, help= "fraction of the training set that can be used to train each tree. If this value is low, it may lead to underfitting or if it is too high, it may lead to overfitting")
                subsample = int(float(ss))
                
                cb = st.number_input('colsample_bytree',0.5, help = "fraction of the features that can be used to train each tree. A large value means almost all features can be used to build the decision tree")
                colsample_bytree = int(float(cb))
    
            xgb_clf_model(df) 
            
            
        if option =='Gradient_Boosting Classifier':
                
            with st.sidebar:
                
               #loss=st.radio('Loss', ('log_loss','exponential'), help = "The loss function to be optimized. ‘log_loss’ refers to binomial and multinomial deviance, the same as used in logistic regression. It is a good choice for classification with probabilistic outputs. For loss ‘exponential’, gradient boosting recovers the AdaBoost algorithm")
                
                lr= st.number_input('learning_rate ', 0.0, help= "Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators. Values must be in the range (0.0, inf)." )
                learning_rate = float(lr)
                
                n_estimators = st.number_input('n_estimators ', 100, help ="The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance. Values must be in the range [1, inf).")
                
                ss = st.number_input('subset',1.0,help ="The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias. Values must be in the range (0.0, 1.0]")
                subsample = int(float(ss))
                
                #criterion = 'squared_error'
                
                msp = st.number_input('min_samples_split',1.0,help = "The minimum number of samples required to split an internal node:If int, values must be in the range [2, inf).If float, values must be in the range (0.0, 1.0] and min_samples_split will be ceil(min_samples_split * n_samples")
                min_samples_split = float(msp)
            
                    
                max_depth = st.number_input(' max_depth',3, help ="The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables. Values must be in the range [1, inf]")
                
            gb_clf_model(df)
           
           
#hide_streamlit_style = """
            #<style>
            #MainMenu {visibility: hidden;}
           # footer {visibility: hidden;}
            #</style>
            #"""
#st.markdown(hide_streamlit_style, unsafe_allow_html=True)            
            
        
                
               
                
                
                             
                 
                 
                 
                 
                 
                 
                 
                 
                 
            
        
            
            
            
            
           
            
           
            
            
            
            
        
        
             
                                           
                                           
                                           
                                         
    
    
          

          
          
          
 
 
 
           
