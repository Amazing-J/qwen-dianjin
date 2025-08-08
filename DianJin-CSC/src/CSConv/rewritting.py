import json
from ..const import STRATEGY_DEFINITION

def read_jsonl(jsonl_path, template_path):
    with open(template_path) as f:
        template = f.read()

    query_list = list()
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line.strip())
            dialogue = item['metadata']['dialogue']
            query = template.replace("{conversation}", dialogue).replace("{strategy_definition}", STRATEGY_DEFINITION)
            query_list.append({
                'query': query,
                'metadata': item['metadata']
            })

    return query_list

def write_batch(query_list, jsonl_path):
    res = list()
    for idx, query in enumerate(query_list):
        custom_id = f"request_{idx}"
        q = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "deepseek-r1",
                "messages": [
                    {
                        "role": "user",
                        "content": query['query']
                    }
                ],
                # "temperature": 0.8,
            }
        }
        res.append(q)

    with open(jsonl_path, 'w') as f:
        for q in res:
            f.write(json.dumps(q, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    # 原始对话
    jsonl_path = "samples.jsonl"
    queries = read_jsonl(jsonl_path, "prompts/rewritting.txt")
    # 用于批处理调用模型
    write_batch(queries, "rewritten_dialogue.jsonl")
