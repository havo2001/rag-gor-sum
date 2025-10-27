import openai
import time


def get_llm_response_via_api(prompt, 
                             API_BASE="https://api.together.xyz",
                             API_KEY="[YOUR_API_KEY]",
                             LLM_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1",
                             TAU=1.0,
                             TOP_P=1.0,
                             N=1,
                             SEED=42,
                             MAX_TRIALS=5,
                             TIME_GAP=5):
    
    openai.api_base = API_BASE
    openai.api_key = API_KEY
    completion = None
    while MAX_TRIALS:
        MAX_TRIALS -= 1
        try:
            completion = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                n=N,
                temperature=TAU,
                top_p=TOP_P,
                seed=SEED,
            )
            break
        except Exception as e:
            print(e)
            print("Retrying...")
            time.sleep(TIME_GAP)

    if completion is None:
        raise Exception(f'Reach MAX_TRIALS={MAX_TRIALS}')
    contents = completion.choices
    if len(contents) == 1:
        return contents[0].message["content"]
    else:
        return [c.message["content"] for c in contents]
