def normalize_turn(talks):
    new_talks = list()
    valid_talk_ids = list()
    metadata_list = list()

    for talk_id, talk_dict in enumerate(talks):
        talk = talk_dict['dialogue']
        metadata = talk_dict['metadata']
        dialogues = talk.split('\n')
        dialogue_history = list()
        user_turn_total = 0
        user_turn_valid = 0
        support_turn = 0

        for dialogue in dialogues:
            if dialogue_history:
                prev_start = dialogue_history[-1][0]
            else:
                prev_start = None
            if dialogue.startswith("客户"):
                user_turn_total += 1
                if len(dialogue[4:]) > 3:
                    user_turn_valid += 1
                cur_start = "客户"
            elif dialogue.startswith("客服"):
                support_turn += 1
                cur_start = "客服"
            else:
                print(talk)
                raise RuntimeError
            cur_extra = dialogue[3:].lstrip()
            if cur_start == prev_start:
                dialogue_history[-1][1] = dialogue_history[-1][1] + cur_extra
            else:
                dialogue_history.append([cur_start, cur_extra])

        new_talk_as_list = [d[0] + ': ' + d[1] for d in dialogue_history]
        new_talk_as_str = '\n'.join(new_talk_as_list)
        turn_num = len(new_talk_as_list) // 2
        # 过滤规则1: 轮数 < 3 或者 > 30
        if turn_num < 3 or turn_num > 30:
            continue
        # 过滤规则3: 客户与客服对话轮数相差过大
        if 2*user_turn_total <= support_turn:
            continue
        # 过滤规则4: 客户有效轮数过低
        if user_turn_total == 0 or 1.0 * user_turn_valid / user_turn_total < 0.7:
            continue
        # 过滤规则2: 单utterance超过500个字符
        too_long = False
        for t in new_talk_as_list:
            if len(t) > 503:
                too_long = True
                break
        if too_long:
            continue
        new_talks.append(new_talk_as_str)
        valid_talk_ids.append(talk_id)
        metadata_list.append(metadata)

    return new_talks, valid_talk_ids, metadata_list