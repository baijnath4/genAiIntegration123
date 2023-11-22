import pandas as pd
import numpy as np
import os
from GenAIScript.credential import API_KEY, PROJECT_ID
import sys


credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey":API_KEY
    }
project_id = PROJECT_ID

# -------------------IBM GEN AI ----------------------------
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
# print([model.name for model in ModelTypes])

model_id = ModelTypes.LLAMA_2_70B_CHAT

from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.STOP_SEQUENCES: ["<end·of·code>"]
}

from ibm_watson_machine_learning.foundation_models import Model

model = Model(
    model_id=model_id, 
    params=parameters, 
    credentials=credentials,
    project_id=project_id)

excel_path = os.path.join(os.getcwd(), 'Data', 'test.xlsx')
print("excel path IBM gen ai:-", excel_path)
dataread = pd.read_excel(excel_path)
df = dataread.copy()

# Generate code based on instruction

instruction = f"""
Answer the following question using only from the table. If there is no good answer in the article, say "I don't know".

Table Information:- 

MONTH=SLA month
YEAR=SLA year
ACCOUNT=List of accounts
METRIC_NAME=Different types of metric name
EXPECTED_TARGET=Expected target of each accounts
SLA_PERFORMANCE=Current SLA or service level agreement
SLA STATUS=SLA achived by the accounts
Comment=End user comment


table: 
###
{df}
###

Question: Which account has the highest expected target?
Answer: Accounts 'INTUIT', 'WEWORK', 'WEWORK', 'WEWORK', 'PAYPAL', 'INTUIT', 'INTUIT', 'WEWORK' has highest expected target.

Question: Which account has the lowest expected targets?
Answer: Accounts 'CIGNA EVERNORTH', 'CIGNA EVERNORTH' has the lowest expected target

Question: list of account whch sla performance is less then expected target?
Answer: The accounts CIGNA EVERNORTH', 'ADP', 'ADP', 'ADP' demonstrate SLA performance lower than the expected target.

Question: Which accounts have not met SLA status in year 2023?
Answer: Accounts ADP -   3,CIGNA EVERNORTH -   1 have not met the status in 2023.

Question: List number of accounts and their metrics which have not met SLA status in year 2023?
Answer: 3 Accounts of ADP with metric name “R2R8-% automated fixed asset” have not met SLA in year 2023.
        1 Accounts of CIGNA EVERNORTH with metric name “Net Revenue_MM Commercial -1” has not met SLA in year 2023.


"""



def genAIFunction(userQuestion,df=df):
    result = model.generate_text(" ".join([instruction, userQuestion]))
    finaResult = result.split('Question')[0]
    return finaResult
