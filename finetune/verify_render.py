import json, requests
from common import TOOLS
from render import render
O='http://localhost:11434'
M='qwen2.5:7b'
def unload(): requests.post(O+'/api/generate', json={'model':M,'keep_alive':0}); 
def chat_count(msgs):
    unload()
    r=requests.post(O+'/api/chat', json={'model':M,'messages':msgs,'tools':TOOLS,'stream':False,'options':{'num_ctx':16384,'num_predict':1}}).json()
    return r['prompt_eval_count']
def raw_count(text):
    unload()
    r=requests.post(O+'/api/generate', json={'model':M,'prompt':text,'raw':True,'stream':False,'options':{'num_ctx':16384,'num_predict':1}}).json()
    return r['prompt_eval_count']
rows=[json.loads(l) for l in open('data/train.jsonl')]
ex=next(r for r in rows if r['family']=='fam_json_edit')
msgs=ex['messages'][:-1]
target=chat_count(msgs)
print('ollama chat tokens', target)
for args_sort in (True, False):
    for prop_sort in (True, False):
        text,_=render(msgs, TOOLS, add_generation_prompt=True, args_sort=args_sort, prop_sort=prop_sort)
        print('args_sort',args_sort,'prop_sort',prop_sort,'raw tokens',raw_count(text))
