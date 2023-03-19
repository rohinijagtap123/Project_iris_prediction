from flask import Flask,render_template,request,redirect
import pickle
import numpy as np

model=pickle.load(open("model.pkl","rb"))

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict_species():
    
    sepal_length=float(request.form.get("sepal_length"))
    sepal_width=float(request.form.get("sepal_width"))
    petal_length=float(request.form.get("petal_length"))
    petal_width=float(request.form.get("petal_width"))
    
    result=model.predict(np.array([[sepal_length,sepal_width,petal_length,petal_width]]))
    
    if result[0]=="setosa":
        return "<h1 style='color:green' 'text-align:center'>setosa</h1>"
    elif result[0]=="versicolor":
        return "<h1 style='color:red' 'text-align:center'>versicolor</h1>"
    else:
        return "<h1 style='color:blue' 'text-align:center'>virginica</h1>"
    
    
# @app.route("/predict",methods=["GET"])
# def predict_placement():
#     cgpa=float(request.args.get("cgpa"))
#     iq=float(request.args.get("iq"))
#     profile_score=float(request.args.get("profile_score"))
    
    
#     result=model.predict(np.array([[cgpa,iq,profile_score]]))
    
#     if result[0]==1:
#         return "<h1 style='color:green'>PLACED</h1>"
#     else:
#         return "<h1 style='color:red'>NOT PLACED</h1>"   

app.run(debug=True,port=5001)