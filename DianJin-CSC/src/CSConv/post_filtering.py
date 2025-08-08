import re
import json
from collections import Counter

def merge_dialogue(dia_list):
    strategy_list = list()
    dialogue_list = list()
    for d in dia_list:
        speaker = d['speaker'].strip()
        if speaker == "客服":
            strategy = d['strategy'].strip()
            text = d['text'].strip()
            dialogue_list.append(f"{speaker}: {text}")
            strategy_list.append(strategy)
        else:
            text = d['text'].strip()
            dialogue_list.append(f"{speaker}: {text}")
            strategy_list.append("")

    return strategy_list, dialogue_list

def read_jsonl(jsonl_path):
    ans = list()
    with open(jsonl_path) as f:
        for index, line in enumerate(f):
            line = line.strip()
            item = json.loads(line)
            metadata = item['metadata']
            response = item['response']
            strategy_list, dialogue_list = merge_dialogue(response)
            reason = item['reason']
            ans.append({
                "index": index,
                "strategy_list": strategy_list,
                "dialogue_list": dialogue_list,
                "reason": reason,
                "metadata": metadata,
            })
    return ans

def try_remove_last_sent_in_dialogue(item_list):
    bad_word = set()
    bad_sent = list()
    for item in item_list:
        strategy_list = item['strategy_list']
        dialogue_list = item['dialogue_list']
        search_result = re.search(r"\(.*\)|（.*）", dialogue_list[-1])
        if search_result is not None:
            if "(^o^)" in dialogue_list[-1] or "(^_−)" in dialogue_list[-1]:
                continue
            bad_word.add(search_result.group(0))
            bad_sent.append(dialogue_list[-1])
            item['strategy_list'] = strategy_list[:-1]
            item['dialogue_list'] = dialogue_list[:-1]

    if bad_word:
        return True
    return False

def filter_by_llm(item_list, prompt_path, result_path):
    with open(prompt_path) as f:
        template = f.read()
    query_list = list()
    for item in item_list:
        strategy_list = item['strategy_list']
        dialogue_list = item['dialogue_list']
        metadata = item['metadata']
        metadata["dialogue_rewritten"] = {
            "strategy_list": strategy_list[:],
            "dialogue_list": dialogue_list[:],
            "reason": item['reason'],
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
                "query": template.replace("{conversation}", conversation),
                "metadata": metadata,
            })

    # 调用qwen2.5-72b-instruct，结果写入到result_path中
    call_qwen_api(query_list, max_workers=32, result_path=result_path, model="qwen2.5-72b-instruct")

    ans = list()
    with open(result_path) as f:
        for index, line in enumerate(f):
            item = json.loads(line)
            # 仅保留高质量对话
            if item['response']['quality'].lower() == "high":
                metadata = item['metadata']
                dialogue_rewritten = metadata["dialogue_rewritten"]

                ans.append({
                    "index": index,
                    "strategy_list": dialogue_rewritten["strategy_list"],
                    "dialogue_list": dialogue_rewritten["dialogue_list"],
                    "reason": dialogue_rewritten["reason"],
                    "metadata": metadata
                })
    return ans

def filter_by_utterance_number(item_list, min_utterance=10, max_utterance=50):
    ans_list = list()
    for item in item_list:
        utterance = len(item['dialogue_list'])
        if utterance < min_utterance and max_utterance > 50:
            continue
        ans_list.append(item)
    return ans_list

def map2strategy(strategy):
    ref_strategy_list = [
        "策略1: 礼貌问候",
        "策略2: 确认身份",
        "策略3: 重述或转述",
        "策略4: 细化问题",
        "策略5: 情感管理",
        "策略6: 提供建议",
        "策略7: 信息传达",
        "策略8: 解决实施",
        "策略9: 反馈请求",
        "策略10: 关系延续",
        "策略11: 感谢与告别",
        "策略12: 其它",
    ]
    strategy_dict = {st.split(":")[0].strip(): idx+1 for idx, st in enumerate(ref_strategy_list)}
    result1 = re.search(r"(策略\d+)", str(strategy))
    result2 = re.search(r"(\d+)", str(strategy))
    if result1 is not None:
        match = result1.group(0)
        return strategy_dict[match]
    elif result2:
        match = int(result2.group(0))
        return match
    else:
        raise RuntimeError(f"strategy[{strategy}] not matched.")

def filter_by_strategy(item_list):
    ans_list1 = list()
    # 排除多个策略的情况
    for item in item_list:
        strategy_list = item['strategy_list']
        ok = True
        for strategy in strategy_list:
            strategy = strategy.strip()
            if "+" in strategy:
                ok = False
                break
        if ok:
            ans_list1.append(item)
    # 确保对话包含策略1，2，11至少1次
    ans_list2 = list()
    for item in ans_list1:
        strategy_list = item['strategy_list']
        strategy_list = [map2strategy(strategy) for strategy in strategy_list if strategy != ""]
        c = Counter(strategy_list)
        if c[1] >= 1 and c[2] >= 1 and c[11] >= 1:
            ans_list2.append(item)

    return ans_list2

def filter_by_cross_existence(item_list):
    ans_list = list()
    for item in item_list:
        dialogue_list = item['dialogue_list']
        ok = True
        for idx, dig in enumerate(dialogue_list):
            if idx == 0:
                continue
            speaker_cur = dialogue_list[idx][:3]
            speaker_prev = dialogue_list[idx-1][:3]
            if speaker_cur == speaker_prev:
                ok = False
                break
        if ok:
            ans_list.append(item)
    return ans_list

def run_filter(jsonl_file="output/rewritten_dialogue.jsonl", result_path="output/rewritten_dialogue_filter.jsonl"):
    res = read_jsonl(jsonl_file)
    try_num = 0
    print(f"# filter1: 尝试移除系统提示信息")
    while try_remove_last_sent_in_dialogue(res):
        try_num += 1
        continue
    print(f"try {try_num} times.")

    print(f"# filter2: 根据LLM过滤")
    res_filter_by_llm = filter_by_llm(res, "prompts/post_filtering.txt", result_path=result_path)
    print(f"{len(res)} vs. {len(res_filter_by_llm)}")

    print(f"# filter3: 根据轮数过滤")
    res_filter_by_utterance = filter_by_utterance_number(res_filter_by_llm)
    print(f"{len(res_filter_by_llm)} vs. {len(res_filter_by_utterance)}")

    print(f"# filter4: 根据策略去过滤，确保策略完全匹配，确保仅包含策略1，2，11一次")
    res_filter_by_strategy = filter_by_strategy(res_filter_by_utterance)
    print(f"{len(res_filter_by_utterance)} vs. {len(res_filter_by_strategy)}")

    print(f"# filter5: 若客户与客服未交叉出现，则删除")
    res_filter_by_cross = filter_by_cross_existence(res_filter_by_strategy)
    print(f"{len(res_filter_by_strategy)} vs. {len(res_filter_by_cross)}")

    return res_filter_by_cross
