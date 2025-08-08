import os, sys
import json, traceback, requests, time

def curl_url(url, post_data):
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Authorization": 'Bearer ' + os.getenv("API_KEY"),
        "X-DashScope-DataInspection": '{"input":"disable","output":"disable"}'
    }
    try:
        r = requests.post(url=url, data=json.dumps(post_data), headers=headers, verify=False)
        response_json = r.json()
        return response_json
    except Exception as e:
        traceback.print_exc()
        return None

def get_result(messages, max_retries=5, wait_time=1):
    for attempt in range(max_retries):
        try:
            responses = curl_url('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions', messages)
            if responses and "choices" in responses and responses["choices"]:
                return responses["choices"][0]["message"]["content"]
            else:
                print("No valid choices found in response. Attempt: {}. Response: {}".format(attempt + 1, responses))
        except Exception as e:
            print(f"Error when getting response: {e}")
            traceback.print_exc()

        time.sleep(wait_time)

    print("Max retries reached. No valid response received.")
    return None

def get_r1_result(messages, max_retries=5, wait_time=1):
    for attempt in range(max_retries):
        try:
            responses = curl_url('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions', messages)
            if responses and "choices" in responses and responses["choices"]:
                reasoning = responses["choices"][0]["message"]["reasoning_content"]
                answer = responses["choices"][0]["message"]["content"]
                return reasoning, answer
            else:
                print("No valid choices found in response. Attempt: {}. Response: {}".format(attempt + 1, responses))
        except Exception as e:
            print(f"Error when getting response: {e}")
            traceback.print_exc()

        time.sleep(wait_time)

    print("Max retries reached. No valid response received.")
    return None, None