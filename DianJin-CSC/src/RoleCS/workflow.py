import re
import os
import sys
import time
import json
import random
import chevron
from tqdm import tqdm
from ..const import STRATEGY_DEFINITION, STAGE_DEFINITION
from ..utils.curl_dashscope import get_result as get_qwen_result, get_r1_result
from ..utils.gpt_parallel import run_gpt_parallel

STRATEGY_MAP = {
    "礼貌问候": "用于对话初始阶段，通过标准化语句表达友好和尊重（区别要点：仅用于开场，不涉及后续内容交互）",
    "确认身份": "严格限定在礼貌问候之后，通过直接询问个人关键信息进行安全验证（区别要点：必须与身份验证或权限确认直接相关，而非其他信息收集）",
    "重述或转述": "主动复现用户的核心问题，使用简化语言澄清细节.（区别要点：仅用于确认理解准确性，不添加新信息或建议）",
    "细化问题": "通过具体、聚焦的追问挖掘深层需求（区别要点：要求明确的新信息，区别于重述或转述和政策解答）",
    "情感管理": "优先处理负面情绪表达，主动提供同理心响应并暂停其他流程（区别要点：仅当客户情绪明显负面（如投诉、愤怒）时适用，需立即响应而非后续处理）",
    "提供建议": "基于已确认的需求，提出主动解决方案或操作步骤建议（区别要点：必须给出具体行动方案，而非解释性内容或情感安抚）",
    "信息传达": "系统性解释规则、流程或条款（区别要点：仅传递非主观的客观规则，不可掺杂建议或解决措施。）",
    "解决实施": "启动已确定的解决方案，同步执行进展并确认客户知悉（区别要点：必须涉及实际操作的开始或进展更新，而非信息传达或反馈请求",
    "反馈请求": "在问题完全处理完毕后，明确邀请客户评估结果（区别要点：仅用于关闭流程前的最终确认，而非中间步骤。）",
    "关系延续": "主动引导客户关注未来服务或产品更新（区别要点：必须指向未来非紧急事项，区别于即时问题的解决或感谢。）",
    "感谢与告别": "以标准化结束语关闭对话（区别要点：仅在确认结束对话时使用，内容不包含信息或建议。）",
    "其它": "当且仅当前11项策略均不适用时触发，例如客户无回应、提问与业务无关、要求违反政策等，例如「您的问题超出服务范围，建议咨询官方客服」。（区别要点：作为兜底策略，排除其他所有情况后方可使用。）",
}

def extract_json(json_str):
    if isinstance(json_str, dict):
        return json_str
    json_str = re.sub(r'\/\/.*', '', json_str)
    json_str = json_str.replace("```json", "```")
    pattern = r'```(.*?)```'
    matches = re.findall(pattern, json_str, re.DOTALL)
    item = matches[0]
    item = json.loads(item)
    return item

def call_llm_and_extract_json(data, max_retries=1500, wait_time=1):
    model = data["model"]
    for _ in range(max_retries):
        try:
            if 'deepseek' in model.lower():
                response = get_r1_result(data, max_retries=1, wait_time=0)
            else:
                response = get_qwen_result(data, max_retries=1, wait_time=0)

            thinking = None
            if "deepseek" in model.lower():
                thinking = response[0]
                response = response[1]
            response = extract_json(response)
            return response, thinking
        except Exception as e:
            print(f"Error when calling llm and extract json: {e}")
            time.sleep(wait_time)

    return None, None

class Dialogue:
    # 专门管理agent交互中的对话历史
    def __init__(self, result_path=None):
        # 存储说话人信息
        self.speaker_list = list()
        # 存储客户与客服的对话历史
        self.dialogue_list = list()
        # 存储客服的策略，对应的客户位置保留为空
        self.strategy_list = list()
        # 存储客户的对话方向
        self.suggestion_list = list()
        # 存储思考过程
        self.strategy_thinking_list = list()
        self.role_thinking_list = list()
        # 以字符串形式的最终历史，包含策略
        self.dialogue_str = ""
        self.dialogue_finished = False
        self.reason = ""

        # 准备结果文件
        self.result_path = result_path
        self.result_f = None

    def add_dialogue(self, speaker, text, strategy="", direction="", strategy_thinking="", role_thinking=""):
        text = text.strip()
        if text[:2] in ("客户", "客服"):
            text = text[2:].strip()
        elif text[:3] in ("客户:", "客服:") :
            text = text[3:].strip()

        self.speaker_list.append(speaker)
        self.strategy_list.append(strategy)
        self.suggestion_list.append(direction)
        self.dialogue_list.append(text)
        self.strategy_thinking_list.append(strategy_thinking)
        self.role_thinking_list.append(role_thinking)

        if speaker == "客服":
            self.dialogue_str += f"{speaker}: [{strategy}] {text}\n"
        elif speaker == "客户":
            self.dialogue_str += f"{speaker}: {text}\n"
        else:
            raise ValueError(f"Invalid speaker: {speaker}")

    def get_dialogue(self):
        return self.dialogue_str

    def is_finished(self):
        utterance = len(self.speaker_list)
        if utterance >= 50:
            self.dialogue_finished = True
            self.reason = "对话长度超过50轮"

        return self.dialogue_finished

    def set_dialogue_finished(self, reason):
        self.dialogue_finished = True
        self.reason = reason

    def get_last_dialogue(self):

        speaker = self.speaker_list[-1]
        strategy = self.strategy_list[-1]
        suggestion = self.suggestion_list[-1]
        text = self.dialogue_list[-1]
        return {
            "id": len(self.speaker_list),
            "speaker": speaker,
            "strategy": strategy,
            "suggestion": suggestion,
            "text": text
        }

    def get_last_dialogue_with_thinking(self):

        speaker = self.speaker_list[-1]
        strategy = self.strategy_list[-1]
        suggestion = self.suggestion_list[-1]
        text = self.dialogue_list[-1]
        strategy_thinking = self.strategy_thinking_list[-1]
        role_thinking = self.role_thinking_list[-1]
        return {
            "speaker": speaker,
            "strategy": strategy,
            "suggestion": suggestion,
            "text": text,
            "strategy_thinking": strategy_thinking,
            "role_thinking": role_thinking
        }

    def open_result_path(self):
        if self.result_path is None:
            print(f"WARNING: result_path is None.")
            return
        self.result_f = open(self.result_path, 'w')

    def write_last_dialogue(self):
        if self.result_path is None:
            print(f"WARNING: result_path is None.")
            return
        last_dialogue = self.get_last_dialogue_with_thinking()
        last_dialogue_str = json.dumps(last_dialogue, ensure_ascii=False)
        self.result_f.write(last_dialogue_str + '\n')
        self.result_f.flush()

    def write_metadata(self, metadata):
        if self.result_path is None:
            print(f"WARNING: result_path is None.")
            return
        metadata_str = json.dumps(metadata, ensure_ascii=False)
        self.result_f.write(metadata_str + '\n')
        self.result_f.flush()

    def close_result_path(self):
        if self.result_path is None:
            print(f"WARNING: result_path is None.")
            return
        self.result_f.close()

class PlannerAgent:
    def __init__(self, planner_template_path="planner.txt", model="qwen2.5-72b-instruct"):
        project_dir = os.path.dirname(__file__)
        self.model = model
        self.planner_template_path = os.path.join(
            project_dir, planner_template_path
        )

        with open(self.planner_template_path) as f:
            self.planner_template = f.read()

        self.planner_prompt = ""
        self.dialogue_history = ""

    def set_prompt(self, **kwargs):
        value = {
            **kwargs
        }
        self.planner_prompt = chevron.render(self.planner_template, value, warn=True)

    def plan(self):
        query = self.planner_prompt

        messages = {
            "model": self.model,
            "messages": [
                {
                    'role': 'user',
                    'content': query,
                }
            ],
            "temperature": 0.8
        }
        item, thinking = call_llm_and_extract_json(messages)
        if item is None:
            return None, None

        return item['dialogue_scenario'], item["customer_goal"], thinking

class SupportAgent:
    # 模拟客服
    def __init__(self, support_template_path="support.txt", model="qwen2.5-72b-instruct"):
        project_dir = os.path.dirname(__file__)
        self.model = model
        self.support_template_path = os.path.join(
            project_dir, support_template_path
        )

        with open(self.support_template_path) as f:
            self.support_template = f.read()

        self.support_prompt = ""
        self.dialogue_history = ""

    def set_prompt(self, **kwargs):
        value = {
            **kwargs
        }
        self.support_prompt = chevron.render(self.support_template, value, warn=True)

    def response_to_user(self, strategy, dialogue):

        assert strategy in STRATEGY_MAP
        strategy = strategy + ": " + STRATEGY_MAP[strategy]

        query = self.support_prompt.replace("{strategy}", strategy).replace(
            "{history}", dialogue
        )

        messages = {
            "model": self.model,
            "messages": [
                {
                    'role': 'user',
                    'content': query,
                }
            ],
            "temperature": 0.8
        }
        item, thinking = call_llm_and_extract_json(messages)
        if item is None:
            return None, None

        return item['text'], thinking


class UserAgent:
    # 模拟客户
    def __init__(self, user_template_path="user3.txt", model="qwen2.5-72b-instruct"):
        project_dir = os.path.dirname(__file__)
        self.model = model
        self.user_template_path = os.path.join(
            project_dir, user_template_path
        )

        with open(self.user_template_path) as f:
            self.user_template = f.read()

        self.user_prompt = ""
        self.dialogue_history = ""

    def set_prompt(self, description, scenario):
        value = {
            "profile_description": description,
            "scenario": scenario,
        }
        self.user_prompt = chevron.render(self.user_template, value, warn=True)


    def response_to_support(self, direction, history):

        query = self.user_prompt.replace('{history}', history).replace("{direction}", direction)

        messages = {
            "model": self.model,
            "messages": [
                {
                    'role': 'user',
                    'content': query,
                }
            ],
            "temperature": 0.8
        }
        item, thinking = call_llm_and_extract_json(messages)
        if item is None:
            return None, None, None, None

        return item['text'], item['is_dialogue_finished'], item['reason'], thinking

class StrategyAgent:
    def __init__(self,
                 template_path="prompts/user_strategy.txt",
                 model="qwen2.5-72b-instruct",
                 temperature=0.8,
                 name="策略教练"
                 ):
        project_dir = os.path.dirname(__file__)
        self.model = model
        self.template_path = os.path.join(
            project_dir, template_path
        )

        with open(self.template_path) as f:
            self.strategy_template = f.read()

        self.strategy_prompt = self.strategy_template
        self.dialogue_history = ""
        self.name = name

        self.temperature = temperature


    def set_prompt(self, **kwargs):
        value = {
            **kwargs
        }
        self.strategy_prompt = chevron.render(self.strategy_template, value, warn=True)

    def get_response(self, history=""):
        query = self.strategy_prompt.replace("{history}", history)
        messages = {
            "model": self.model,
            "messages": [
                {
                    'role': 'user',
                    'content': query,
                }
            ],
            "temperature": self.temperature
        }

        item, thinking = call_llm_and_extract_json(messages)
        if item is None:
            return None, None

        return item, thinking

class Simulator:
    def __init__(self,
                 setting=None,
                 planner_template_path="prompts/planner.txt",
                 support_assistant_template_path="prompts/supporter_assistant.txt",
                 user_assistant_template_path="prompts/customer_assistant.txt",
                 support_template_path="prompts/supporter.txt",
                 user_template_path="prompts/customer.txt",
                 model="gpt-4o-mini",
                 ):

        self.persona = setting['persona']
        self.topic = setting['topic']

        self.planner = PlannerAgent(planner_template_path=planner_template_path, model=model)
        self.planner.set_prompt(profile_description=persona, topic=topic)

        self.support_agent = SupportAgent(support_template_path=support_template_path, model=model)

        self.support_strategy_agent = StrategyAgent(name="客服教练", template_path=support_assistant_template_path, model=model)

        self.user_agent = UserAgent(user_template_path=user_template_path, model=model)

        self.user_strategy_agent = StrategyAgent(name="客户教练", template_path=user_assistant_template_path, model=model)


    def plan(self):
        scenario, customer_goal = self.planner.plan()

        self.user_agent.set_prompt(self.persona, scenario)
        self.user_strategy_agent.set_prompt(scenario=scenario,
                                            customer_goal=customer_goal,
                                            profile_description=self.persona)
        self.support_agent.set_prompt(scenario=scenario)
        self.support_strategy_agent.set_prompt(
            strategy_definition = STRATEGY_DEFINITION,
            stage_definition = STAGE_DEFINITION
        )

    def auto_chat_by_support(self, result_path=None):
        dialogue_manager = Dialogue(result_path=result_path)
        dialogue_manager.open_result_path()
        while True:
            latest_dialogue = dialogue_manager.get_dialogue()
            strategy, strategy_thinking = self.support_strategy_agent.get_response(history=latest_dialogue)
            # 异常情况跳过
            if strategy is None:
                dialogue_manager.close_result_path()
                break

            strategy = strategy["strategy"]
            if strategy not in STRATEGY_MAP:
                strategy = "其它"
            if strategy == "感谢与告别" and not dialogue_manager.is_finished():
                dialogue_manager.set_dialogue_finished(reason="客服感谢与告别")

            # 2. 客服根据策略进行回复
            latest_dialogue = dialogue_manager.get_dialogue()
            support_response, role_thinking = self.support_agent.response_to_user(strategy, latest_dialogue)
            # 异常情况跳过
            if support_response is None:
                dialogue_manager.close_result_path()
                break

            dialogue_manager.add_dialogue("客服", support_response,
                                          strategy=strategy,
                                          strategy_thinking=strategy_thinking,
                                          role_thinking=role_thinking)
            d1 = dialogue_manager.get_last_dialogue()
            dialogue_manager.write_last_dialogue()
            if dialogue_manager.is_finished():
                metadata = {
                    "finished_reason": dialogue_manager.reason,
                }
                dialogue_manager.write_metadata(metadata)
                dialogue_manager.close_result_path()
                break

            # 3. 客户教练给出建议
            latest_dialogue = dialogue_manager.get_dialogue()
            direction, strategy_thinking = self.user_strategy_agent.get_response(latest_dialogue)
            # 异常情况跳过
            if direction is None:
                dialogue_manager.close_result_path()
                break

            direction = direction["direction"]
            # 4. 客户给出回复
            user_response, dialogue_finished, reason, role_thinking = self.user_agent.response_to_support(direction, latest_dialogue)
            # 异常情况跳过
            if user_response is None:
                dialogue_manager.close_result_path()
                break

            dialogue_manager.add_dialogue("客户", user_response,
                                          direction=direction,
                                          strategy_thinking=strategy_thinking,
                                          role_thinking=role_thinking)
            if dialogue_finished:
                dialogue_manager.set_dialogue_finished(reason="客户主动结束")

            d2 = dialogue_manager.get_last_dialogue()
            dialogue_manager.write_last_dialogue()

def get_setting_list(description_list):

    setting_list = list()
    topic_list = [
        "产品咨询",
        "营销与推广活动",
        "账户和交易管理",
        "财务咨询与规划",
        "风险管理与安全",
        "投诉与纠纷解决",
        "技术支持与在线服务",
    ]
    cnt = 0
    for topic in topic_list:
        for persona in description_list:
            cnt += 1
            setting_list.append((cnt, {
                "topic": topic,
                "persona": persona,
            }))

    return setting_list

def run_single_dialogue(param):
    idx, setting = param
    sm = Simulator(setting=setting, model=MODEL)
    sm.plan()
    sm.auto_chat_by_support(result_path=f"synth_new/{MODEL}-dig{idx}.jsonl")


def get_unfinished(lst):
    left = list()
    for idx, setting in enumerate(lst):
        file_path = f"synth_new/{MODEL}-dig{idx}.jsonl"
        if not os.path.exists(file_path):
            left.append((idx, setting))
            continue
        with open(file_path, encoding='utf-8') as f:
            try:
                lines = f.readlines()
            except:
                left.append((idx, setting))
                continue

            if not lines:
                left.append((idx, setting))
                continue
            try:
                target = json.loads(lines[-1].strip())
            except:
                left.append((idx, setting))
                continue
            if "finished_reason" in target:
                continue
            left.append((idx, setting))
    return left

def persona2text(persona_path, prompt_path):
    with open(prompt_path) as f:
        prompt_template = f.read()
    query_list = list()
    with open(persona_path) as f:
        for line in f:
            line = line.strip()
            persona = json.loads(line)
            query = prompt_template.replace("{character_profile}", persona)
            query_list.append({
                "query": query,
                "metadata": {
                    "persona": persona
                }
            })

    run_gpt_parallel(query_list, max_workers=128, result_path="description.jsonl")
    description_list = list()
    with open("description.jsonl") as f:
        for line in f:
            item = json.loads(line.strip())
            description = item["response"]
            description_list.append(description)

    return description_list

if __name__ == '__main__':
    random.seed(1234)
    MODEL = "deepseek-r1"
    # persona文件
    persona_path = "output/persona.jsonl"
    # 转换为描述性文本
    description_list = persona2text(persona_path)
    samples = get_setting_list(description_list)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [executor.submit(run_single_dialogue, (idx, st)) for idx, st in samples]
        try:
            for job in tqdm(as_completed(futures), desc="running", total=len(samples)):
                out_json = job.result(timeout=None)  # 默认timeout=None，不限时间等待结果
        except Exception as e:
            print(e)
            # 1. 关闭线程池
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(0)
