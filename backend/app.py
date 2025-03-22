from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image_file = request.files['image']
        
        # Open the image
        image = Image.open(image_file)
        # Here you can add your image processing and prediction code
        # For example, preprocess the image and pass it to your model
        prediction = "fruit_name"  # Replace this with the actual model prediction
        return jsonify({"result": prediction})
    
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
