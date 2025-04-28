from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle


app = Flask(__name__)

dependencies = {
    'auc_roc': AUC
}

verbose_name = {
0: 'Normal',
1: 'Stroke',

}



class_labels = ["Normal","Stroke"]


models = load_model('stroke.h5')
model = load_model('strokes.h5')


def cnn(img_path):
	test_image = image.load_img(img_path, target_size=(224,224))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 224,224,3)

	predict_x=models.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	
	return  class_labels [classes_x[0]]


def predict_label(img_path):
  test_image = image.load_img(img_path, target_size=(224,224))
  test_image = image.img_to_array(test_image)/255.0
  test_image = test_image.reshape(-1,1, 224,224,3)

  predict_x=model.predict(test_image) 
  classes_x=np.argmax(predict_x,axis=1)
  
  return  class_labels [classes_x[0]]






@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')   
    
@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def submit():
	predict_result = None
	img_path = None
	model = None
	if request.method == 'POST':
		img = request.files['my_image']
		model = request.form['model']
		print(model)
	    # predict_result = "Prediction: Success" 
		img_path = "static/tests/" + img.filename	
		img.save(img_path)
		#plt.imshow(img)

		if model == 'cnn':

		     predict_result = cnn(img_path)
		elif model == 'lstm':
			 predict_result = predict_label(img_path)
			
 			  
	return render_template("prediction.html", prediction = predict_result, img_path = img_path, model = model)



@app.route("/performance")
def performance():
	return render_template('performance.html')   


@app.route("/chart")
def chart():
	return render_template('chart.html')   
    

	
if __name__ =='__main__':
	app.run(debug = True)


	

	


