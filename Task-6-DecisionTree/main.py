from flask import Flask,render_template,request
import pickle

app=Flask(__name__)

file=open("model.pkl","rb")
speciesTree=pickle.load(file)
file.close()

@app.route("/" , methods=["GET","POST"])
def home():
    if request.method == "POST":
        myDict=request.form
        sepal_length = float(myDict["sepal_length"])
        sepal_width = float(myDict["sepal_width"])
        petal_length = float(myDict["petal_length"])
        petal_width = float(myDict["petal_width"])
    
        pred = [sepal_length,sepal_width,petal_length,petal_width]
        iris_pred = speciesTree.predict([pred])[0]
        # print(iris_pred)
        # return render_template("result.html",age=age,bmi=bmi,insulin=insulin,glucose=glucose,BP=BP,thickness=thickness,res=round(Diab_pred))
        return render_template("result.html",sepal_length=sepal_length,sepal_width=sepal_width,petal_length=petal_length,petal_width=petal_width,res=iris_pred)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)