import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  #####
import mysql.connector
import streamlit as st

def main():
    st.header("Used Car Price Prediction Cardheko")
    st.write(" ")
    st.write(" ")

    model_name1 =st.sidebar.selectbox(
                "Select the option to check values:",
                ("Home","Predict Car Price","Cardheko Insights"),key=1,index=None)
    #=====================================================================================================

    mydb = mysql.connector.connect(host='localhost',
                                user='root',
                                password='Jesu@123',
                                database='Carsdheko')

    mycursor = mydb.cursor()

    #==========================================

    sql = "DESC car_details"
    mycursor.execute(sql)
    a=mycursor.fetchall()
    sql = "select * from car_details"
    mycursor.execute(sql)
    b=mycursor.fetchall()
    col=list(pd.DataFrame(a)[0])
    df=pd.DataFrame(data=b,columns=col)
    df
    test_df=df
#===============================================

    if model_name1 == "Cardheko Insights":

        Data_Insights()

    elif model_name1 == "Predict Car Price": 
        model_predict(test_df)   

    else:
        Home()

#---------------------------------------


 #\\\\\\\\\==========================================/////////////////   

def Data_Insights():
     st.header('Data_Insights')
     st.write("Data insights")


#++++++++++++++++++++++++++++++++++++++++++++++++++
    
def Home():

    st.write("""CarDekho is one of India's leading online platforms for buying and selling new and used cars. 
             It offers a wide range of services and features related to the automotive industry.""")

#===================================================    

def model_predict(test_df):

    cb=[]
    ms=list(test_df.columns)
    for i in ms:
        if i[:6]=='Brand_':
            cb.append(i[6:])
    cb=tuple(cb)

    br=st.selectbox("Choose Brand",cb,key=2,index=0)
        
    tm=[]
    for i in ms:
        if i[:6]=='Model_' and i[6:6+len(br)]==br:
            tm.append(i)

    mo=st.selectbox("Choose Model",tm,key=3,index=0)

    cv=[]
    t5=test_df[test_df[mo]==1]
    cv=t5['variantName'].unique()
    list(cv)

    va=st.selectbox("Choose Variant",cv,key=4,index=None)

    cm=[]
    ms=list(test_df.columns)
    for i in ms:
        if i[:6]=='Model_' and i[:10]!='Model_Year':
            cm.append(i[6:])

    cm=tuple(cm)        

    cy=[]
    cy=test_df['Model_Year'].unique()
    cy=tuple(cy)

    cl=[]
    dic1={'Delhi':5,'Chennai':4,'Bangalore':3,'Hyderabad':2,'Kolkata':1,'Jaipur':0}
    cl=list(dic1.keys())
    cl=tuple(cl)

    my=st.selectbox("Choose Model Year",cy,key=5,index=None)
    km=st.number_input("Enter KMs Driven",key=6)
    nw=st.number_input('Enter No of Owners', min_value=1, max_value=5, value=1, step=1,key=8)
    lo=st.selectbox("Choose Location",cl,key=7,index=None)

    if st.button('submit',key=9):


        d=test_df[(test_df['Brand_'+br]==1)&(test_df[mo]==1) & (test_df['variantName']==va)]
        var=list(d['Variant'])[0]
        fuel=list(d['Fuel_Type'])[0]
        trans=list(d['Transmission'])[0]
        eng=list(d['Engine_CC'])[0]
        feat=list(d['Features'])[0]
        #++++++++++++++++++++++++++++++++++++++++++++++++

        temp=list(test_df.columns)

        dic={}
        for i in temp:
            if i[0]=='L':
                dic[i]=2
            elif i=='Fuel_Type':
                dic[i]=3
            elif i[0]=='T':
                dic[i]=1
            elif i[0]=='N':
                dic[i]=1
            elif i== 'Model_Year':
                dic[i]=2012
            elif i=='Engine_CC':
                dic[i]=1000
            elif i[0] =='V':
                dic[i]=778
            elif i[0]=='F':
                dic[i]= 21
            elif i[0]=='B':
                dic[i]=False
            elif i[0]=='M':
                dic[i]=False

        dic['Brand_Tata']=True
        dic['Model_Tata Nexon']=True

        t1=pd.DataFrame(dic,index=[0])

    y=test_df['Price']
    X=test_df.drop(['Price','Car_Type','KMs_Driven','Seats','variantName'],axis=1)

    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

    best_algo=['name',0] #####
    models=[LinearRegression(),Lasso(alpha=0.01),RandomForestRegressor(),XGBRegressor()]
    for model in models:
        model.fit(x_train,y_train)

        train_pred=model.predict(x_train)
        test_pred=model.predict(x_test)
        
        predicted_value =model.predict(t1)##########

        if str(model)[:12] == "XGBRegressor":
            mod='XGBRegressor.'
        else:
            mod=model

        st.write(f"{str(mod)}")
        st.write('\n\n\n****Train****')
        st.write(f'Training Error:{mean_squared_error(y_train,train_pred)}')
        st.write(f'Training R2_Score:{r2_score(y_train,train_pred)}')
        
        st.write('****Test****')

        st.write(f'Test Error:{mean_squared_error(y_test,test_pred)}')
        st.write(f'Test R2_Score:{r2_score(y_test,test_pred)}')

        st.write("\n\nPredicted Value: ",2**predicted_value)
        
        st.write("_____________________________________________________\n")
        
        if best_algo[1] <= r2_score(y_test,test_pred):
            best_algo=[type(model).__name__,r2_score(y_test,test_pred)]

    st.write(best_algo)
   




if __name__ == "__main__":
    main()