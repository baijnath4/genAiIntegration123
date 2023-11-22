from flask import Flask, request,jsonify
import os
import pandas as pd
import sys
from GenAIScript.IBM_GenAI import genAIFunction

app = Flask(__name__)
port=5000

excel_path = os.path.join(os.getcwd(), 'Data', 'test.xlsx')
print("excel path IBM gen ai:-", excel_path)
dataread = pd.read_excel(excel_path)
requiredData =  dataread.copy()

sourceDataJSON = requiredData.to_json(orient='records')
# Endpoint to show the source data


@app.route('/getData', methods=['GET'])
def get_data():
    return (sourceDataJSON)



@app.route('/askQuestion', methods=['POST'])
def askQuestion():
    req_data = request.get_json()
    question = req_data['Question']
    resut=genAIFunction(question)

    return (jsonify({'Question': question, 'Answer:':resut}))


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=port)
