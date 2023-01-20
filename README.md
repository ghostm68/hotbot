# huggingface_flask_chatbot
Inferences to AI models like Bloom, GPT Neo etc, from a web page in Flask

It needs a token that is retrieve from Hugging Face website ( sign up is required ). Once logged in to Huggingface, upper right, settings, access token-> new token


Installation ( Linux ):
=======================

git clone https://github.com/cf2018/huggingface_flask_chatbot.git  
python -m venv venv  
pip3 install -r requirements.txt  
cd huggingface_flask_chatbot  
flask run  

Open browser at http://127.0.0.1:8000  

Note:
=====
Token from Hugging Face needs to be configured in the app.py
