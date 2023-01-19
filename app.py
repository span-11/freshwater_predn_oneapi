import pickle
import pandas as pd
import daal4py as d4p
from flask import Flask,render_template,request,redirect,url_for
import os 

# print(os.getcwd())
# print(os.listdir())
model=pickle.load(open('final_model_lgbm.pkl','rb'))

app=Flask(__name__)

prediction_features=['Chloride','Chlorine','Color','Copper','Fluoride','Iron','Manganese',
 'Nitrate','Odor','Sulfate', 'Total Dissolved Solids','Turbidity','Zinc','pH']

@app.route('/',methods=['GET','POST'])
def index():
        if request.method=='POST':
            val_nm=request.form.get('Entry_Type')
            if val_nm=='1':
                return redirect(url_for('manual_entry'))
            else:
                return redirect(url_for('user_file_upload'))
        else:
            return render_template('index.html')

@app.route('/file_upload',methods=['GET','POST'])
def user_file_upload():
    if request.method=='POST':
        uploaded_file=request.files["uploaded_file"]
        filename_file=uploaded_file.filename
        if '.csv' in filename_file:
            input_file=pd.read_csv(filename_file)
            if input_file.shape[0]!=0:
                if len(set(input_file.columns).intersection(prediction_features))==14:
                    input_file['Color_Recoded']=input_file['Color'].map({'Colorless':1,'Near Colorless':2,'Faint Yellow':3,'Light Yellow':4,'Yellow':5})
                    input_file['Color_Recoded'].fillna(5,inplace=True)
                    input_for_prediction=input_file[prediction_features]
                    input_for_prediction=input_for_prediction.apply(lambda x: pd.to_numeric(x,downcast='float',errors='coerce')).fillna(-1)
                    prediction=d4p.gbt_classification_prediction(nClasses=2,resultsToEvaluate="computeClassLabels|computeClassProbabilities",fptype='float').compute(input_for_prediction, model).prediction
                    input_file['Target']=prediction
                    input_file.drop('Color_Recoded',axis=1,inplace=True)
                    input_file.fillna('',inplace=True)
                    return input_file.to_html()
                else:
                    return str("Some feature/s are missing. Please ensure to include these features : Chloride,    Chlorine,   Color,  Copper, Fluoride,   Iron,   Manganese,  Nitrate,    Odor,   Sulfate,    Total Dissolved Solids, Turbidity,  Zinc,   pH")
            else:
                return str("The input data comprises of no records. Please upload records")
        else:
            return str("Please upload a csv file")
    else:
        return render_template('file_upload.html')

@app.route('/single_entry',methods=['GET','POST'])
def manual_entry():
    if request.method=='GET':
        return render_template('manual_entry.html')
    else:
        form_inputs=pd.DataFrame(request.form.to_dict(),index=[0])
        form_inputs=form_inputs.apply(lambda x:pd.to_numeric(x,downcast='float',errors='coerce')).fillna(-1)
        # return form_inputs.to_html()
        prediction=d4p.gbt_classification_prediction(nClasses=2,resultsToEvaluate="computeClassLabels|computeClassProbabilities",fptype='float').compute(form_inputs, model).prediction[0][0]
        return str("Target :")+str(prediction)
        # if prediction==1:
        #     return str("The source is likely to be a fresh-water")
        # else:
        #     return str("The source is not likely to be a freshwater")

# if __name__=="__main__":
#     app.run(debug=True)
