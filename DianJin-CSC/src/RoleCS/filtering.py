import json

def filter_by_llm(item_list, prompt_path, result_path):
    with open(prompt_path) as f:
        template = f.read()
    query_list = list()
    for item in item_list:
        strategy_list = item['strategy_list']
        dialogue_list = item['dialogue_list']
        metadata = item['metadata']
        metadata["dialogue"] = {
            "strategy_list": strategy_list[:],
            "dialogue_list": dialogue_list[:],
        }
        conversations = list()
        parse_ok = True
        for strategy, dialogue in zip(strategy_list, dialogue_list):
            if dialogue.startswith("客服: "):
                text = dialogue[3:].strip()
                conversations.append(f"客服 ({strategy}): {text}")
            else:
                if not dialogue.startswith("客户: "):
                    parse_ok = False
                    break
                conversations.append(dialogue)
        if parse_ok:
            conversation = '\n'.join(conversations)
            query_list.append({
                "query": template.replace("{history}", conversation),
                "metadata": metadata,
            })

    # 调用qwen2.5-72b-instruct，结果写入到result_path中
    call_qwen_api(query_list, max_workers=32, result_path=result_path, model="qwen2.5-72b-instruct")

    ans = list()
    with open(result_path) as f:
        for index, line in enumerate(f):
            item = json.loads(line)
            if int(item['response']['score']) == 5:
                metadata = item['metadata']
                dialogue = metadata["dialogue"]

                ans.append({
                    "index": index,
                    "strategy_list": dialogue["strategy_list"],
                    "dialogue_list": dialogue["dialogue_list"],
                    "metadata": metadata
                })
    return ans