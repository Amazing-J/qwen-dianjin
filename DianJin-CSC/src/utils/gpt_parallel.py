import os
import re
import sys
import time
import json
from functools import partial


from .curl_dashscope import get_result as get_qwen_result, get_r1_result

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_json(json_str):
    json_str = re.sub(r'\/\/.*', '', json_str)
    json_str = json_str.replace("```json", "```")
    pattern = r'```(.*?)```'
    matches = re.findall(pattern, json_str, re.DOTALL)
    if matches:
        item = matches[0]
        item = json.loads(item)
    else:
        # 尝试直接loads
        item = json.loads(json_str.strip())
    return item

def call_llm(params, model="qwen2.5-72b-instruct", temperature=0.8, max_retries=15, wait_time=1):
    idx, query_dict = params

    query = query_dict['query']
    metadata = query_dict['metadata']

    messages = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ],
        "temperature": temperature,
    }

    for _ in range(max_retries):
        try:
            if "deepseek-r1" in model.lower():
                res = get_r1_result(messages, max_retries=1, wait_time=0)
            else:
                res = get_qwen_result(messages, max_retries=1, wait_time=0)

            if res is None:
                continue

            if "deepseek-r1" in model.lower():
                reason = res[0]
                res = extract_json(res[1])
                out = {
                    'id': idx,
                    'model': model,
                    'query': query,
                    "reason": reason,
                    'response': res,
                    'metadata': metadata
                }
            else:
                res = extract_json(res)
                if "keys" in query_dict:
                    keys = query_dict["keys"]
                    for key in keys:
                        assert key in res
                out = {
                    'id': idx,
                    'model': model,
                    'query': query,
                    'response': res,
                    'metadata': metadata
                }

            return out
        except Exception as e:
            print(f"Error when calling llm: {e}")
            time.sleep(wait_time)

    print(f"Max Retries Exceeded.")
    return None

def get_finished_id(file_path):
    if not os.path.exists(file_path):
        return set()
    finished_id_set = set()
    with open(file_path) as f:
        cnt = 0
        for line in f:
            cnt += 1
            line = line.strip()
            item = json.loads(line)
            if item['response'] is None:
                continue
            finished_id_set.add(item['id'])
    return finished_id_set

def write_result(result_path, result_list):
    with open(result_path, 'a', encoding='utf-8') as f_out:
        for result in result_list:
            result_str = json.dumps(result, ensure_ascii=False)
            f_out.write(result_str + '\n')
        f_out.flush()

def run_gpt_parallel(querys, temperature=0.8, model="qwen2.5-72b-instruct", max_workers=2, result_path="result.jsonl", call_back=None):
    finised_id_set = get_finished_id(result_path)
    skip = len(finised_id_set)
    left = len(querys) - skip
    print(f"skip {skip}, left: {left}")
    failed = 0

    start_time = time.time()
    if call_back is None:
        call_back = partial(call_llm, temperature=temperature, model=model)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(call_back, (idx, query)) for idx, query in enumerate(querys) if
                   idx not in finised_id_set]
        result_list = list()
        try:
            for job in tqdm(as_completed(futures), desc="running", total=left):
                out_json = job.result(timeout=None)  # 默认timeout=None，不限时间等待结果
                if out_json is not None:
                    result_list.append(out_json)
                else:
                    failed += 1
        except:
            executor.shutdown(wait=False, cancel_futures=True)
            write_result(result_path, result_list)
            if failed > 0:
                print(f"{failed} failed. Run again!")
            sys.exit(0)

    write_result(result_path, result_list)
    if failed > 0:
        print(f"{failed} failed. Run again!")

    end_time = time.time()
    total_run_time = round(end_time - start_time, 3)
    print('Total_run_time: {} s'.format(total_run_time))

if __name__ == '__main__':
    query_list = [
        {
            "query": "1+1=?",
            "metadata": ""
        },
        {
            "query": "1+3=?",
            "metadata": ""
        },
    ]
    result_file="result.txt"
    run_gpt_parallel(query_list, max_workers=2, result_path=result_file)