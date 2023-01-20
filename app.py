
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request, render_template
from huggingface_hub import notebook_login
from huggingface_hub import HfFolder
from huggingface_hub import InferenceApi
import time

#https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B
#inference = InferenceApi("bigscience/bloom",token='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')
#inference = InferenceApi("EleutherAI/gpt-j-6B",token='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')


app = Flask(__name__)


@app.route('/')
def chat():
    try:
        model = request.args.get('model')
    except:
        model = "EleutherAI/gpt-j-6B"
    return render_template('chat.html', model=model)

@app.route('/inference/', methods=['GET'])
def generar_codigo():
    # genera code en base a una serie de puntos
    input = request.args.get('input')
    model = request.args.get('model')
    response = infer_model(input,model)
    return str(response).replace(r'\n','<BR>')


def infer_model(prompt,model='EleutherAI/gpt-j-6B',
          max_length = 250,
          top_k = 0,
          num_beams = 0,
          no_repeat_ngram_size = 2,
          top_p = 0.9,
          seed=42,
          temperature=0.7,
          greedy_decoding = False,
          return_full_text = True):

    if model == None:
        return 'Please choose a model'

    inference = InferenceApi(model,token='hf_GET_THE_TOKEN_FROM_HUGGING_FACE_SIGN_UP')

    top_k = None if top_k == 0 else top_k
    do_sample = False if num_beams > 0 else not greedy_decoding
    num_beams = None if (greedy_decoding or num_beams == 0) else num_beams
    no_repeat_ngram_size = None if num_beams is None else no_repeat_ngram_size
    top_p = None if num_beams else top_p
    early_stopping = None if num_beams is None else num_beams > 0

    if 'gpt' not in model.lower() and 'flan' not in model.lower():
        params = {
        "max_new_tokens": max_length,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "do_sample": do_sample,
        "seed": seed,
        "early_stopping":early_stopping,
        "no_repeat_ngram_size":no_repeat_ngram_size,
        "num_beams":num_beams,
        "return_full_text":return_full_text
        }
    else:
        params = {
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "do_sample": do_sample,
        "early_stopping":early_stopping,
        "no_repeat_ngram_size":no_repeat_ngram_size,
        "num_beams":num_beams
        }

    #s = time.time()
    response = inference(prompt, params=params)
    #print(response)
    #proc_time = time.time()-s
    #print(f"Processing time was {proc_time} seconds")
    return response

