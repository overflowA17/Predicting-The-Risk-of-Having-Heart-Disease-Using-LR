from flask import Flask,render_template,request
import joblib
import numpy as np

model=joblib.load('heart_risk_prediction_regression_model.sav')

app=Flask(__name__) #application

@app.route('/')
def index():

	return render_template('webapp.html')

@app.route('/getresults',methods=['POST'])
def getresults():

	result=request.form 

	name=result['name']
	gender=float(result['gender'])
	age=float(result['age'])
	tc=float(result['tc'])
	hdl=float(result['hdl'])
	sbp=float(result['sbp'])
	smoke=float(result['smoke'])
	bpm=float(result['bpm'])
	diab=float(result['diab'])

	test_data=np.array([gender,age,tc,hdl,smoke,bpm,diab]).reshape(1,7)

	prediction=model.predict(test_data)[0]

	resultDict={"name":name,"risk":round(prediction,2)}

	return render_template('webappResults.html',results=resultDict)

app.run(debug=True)