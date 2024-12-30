from flask import Flask, render_template, request, jsonify, send_file
from disease_predictor_new import predictDisease, get_all_symptoms
import csv
from io import StringIO
import datetime

app = Flask(__name__)

# Store predictions for CSV download
predictions_history = []

@app.route('/')
def home():
    symptoms = get_all_symptoms()
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')
    if not selected_symptoms:
        return jsonify({'error': 'Please select at least one symptom'})
    
    symptoms_string = ",".join(selected_symptoms)
    predictions = predictDisease(symptoms_string)
    
    # Store prediction for history
    predictions_history.append({
        'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'symptoms': symptoms_string,
        'prediction': predictions['final_prediction']
    })
    
    return jsonify(predictions)

@app.route('/download_history')
def download_history():
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Date', 'Symptoms', 'Prediction'])
    
    for pred in predictions_history:
        cw.writerow([pred['date'], pred['symptoms'], pred['prediction']])
    
    output = si.getvalue()
    return output, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename=predictions.csv'
    }

if __name__ == '__main__':
    app.run(debug=True)
