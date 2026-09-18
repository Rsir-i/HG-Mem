import os
from tqdm import tqdm
import json

# Resolve every path relative to this file's directory (data/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGIN_DIR = os.path.join(BASE_DIR, "origin_data")
PROCESS_DIR = os.path.join(BASE_DIR, "process_data")


def process_longmemeval():
    ## longmemeval
    # "question_id": "gpt4_2655b836",
    # "question_type": "temporal-reasoning",
    # "question": "What was the first issue I had with my new car after its first service?",
    # "answer": "GPS system not functioning correctly",
    # "question_date": "2023/04/10 (Mon) 23:07",
    # "haystack_dates": [
    # "haystack_session_ids": [
    # "haystack_sessions": [
    # "answer_session_ids": [

    in_data_path = os.path.join(ORIGIN_DIR, "longmemeval_s")
    if not os.path.exists(in_data_path):
        raise FileNotFoundError(f"longmemeval_s raw data not found: {in_data_path}")
    in_data = json.load(open(in_data_path))
    alldata = []

    for entry in tqdm(in_data, desc="Processing longmemeval"):
        question_id = entry['question_id']
        question_type = entry['question_type']
        question = entry['question']
        answer = entry['answer']
        question_date = entry['question_date']
        haystack_dates = entry['haystack_dates']
        haystack_session_ids = entry['haystack_session_ids']
        haystack_sessions = entry['haystack_sessions']
        answer_session_ids = []
        answer_turn_ids = []
        for cur_sess_id, sess_entry, ts in zip(entry['haystack_session_ids'], entry['haystack_sessions'], entry['haystack_dates']):
            for turn_id, turn in enumerate(sess_entry):
                if 'has_answer' in turn and turn['has_answer']==True:
                    clean_sess_id = cur_sess_id.replace('answer_', '')
                    answer_session_ids.append(clean_sess_id)
                    merged_turn = turn_id // 2 + 1
                    answer_turn_ids.append(f"{clean_sess_id}-turn_{merged_turn}")

        sessions = []
        for sess_entry in entry['haystack_sessions']:
            # new_session = [{k: v for k, v in item.items() if k != "has_answer"} for item in sess_entry]
            session = []
            for item in sess_entry:
                session.append(f"[{item['role']}]: {item['content']}")
            merged_session = []
            for i in range(0, len(session), 2):
                if i + 1 < len(session):
                    merged_session.append(session[i] + "\n" + session[i+1])
                else:
                    merged_session.append(session[i])
            if len(merged_session)==0:
                print(len(merged_session),len(session))
            sessions.append(merged_session)

        dataform = {
            'conversation_id':entry['question_id'],
            'qa':[
                {
                "question": entry['question'],
                "question_type": entry['question_type'],
                "question_date":entry['question_date'],
                "answer": entry['answer'],
                "answer_session_ids": answer_session_ids,
                "answer_turn_ids": answer_turn_ids,
            },
            ],
            'sessions_ids':[s.replace('answer_', '') for s in entry['haystack_session_ids']],
            'sessions_dates':entry['haystack_dates'],
            'sessions':sessions
            }
        alldata.append(dataform)

    out_path = os.path.join(PROCESS_DIR, "longmemeval_s.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(alldata, f, ensure_ascii=False, indent=4)
    print(f"[done] processed longmemeval_s → {out_path} ({len(alldata)} items)")


def process_locomo10():
    ## locomo10
    in_data_path = os.path.join(ORIGIN_DIR, "locomo10.json")
    if not os.path.exists(in_data_path):
        raise FileNotFoundError(f"locomo10 raw data not found: {in_data_path}")
    in_data = json.load(open(in_data_path))

    alldata = []

    for entry in tqdm(in_data, desc="Processing locomo10"):
        # dict_keys(['qa', 'conversation', 'event_summary', 'observation', 'session_summary', 'sample_id'])

        newqa = []
        for qaitem in entry['qa']:
            if 'adversarial_answer' in qaitem:
                answer = qaitem['adversarial_answer']
            else:
                answer = qaitem['answer']
            answer_session_ids = []
            answer_turn_ids = []
            for item in qaitem['evidence']:
                try:
                    turn_id = int(item.split(':')[1])
                except:
                    continue
                session_part = item.replace('D', 'session_').split(':')[0]  # e.g. session_1
                merged_turn = turn_id // 2 + 1
                answer_session_ids.append(session_part)
                answer_turn_ids.append(f"{session_part}-turn_{merged_turn}")

            newqa.append(
                {
                "question": qaitem['question'],
                "question_type": qaitem['category'],
                "question_date": None,
                "answer": answer,
                "answer_session_ids": answer_session_ids,
                "answer_turn_ids": answer_turn_ids,
            })

        conversation = entry['conversation']
        sessions_ids = []
        sessions_dates = []
        sessions = []
        for i in range(1000):
            if f'session_{i+1}' in conversation:
                sessions_ids.append(f'session_{i+1}')
                sessions_dates.append(conversation[f'session_{i+1}_date_time'])
                session = []
                for dialog in conversation[f'session_{i+1}']:
                    # Map speaker -> role and text -> content
                    if 'blip_caption' in dialog:
                        session.append(f"[{dialog['speaker']}]: {dialog['text']}\n The image Caption: {dialog['blip_caption']}")
                    else:
                        session.append(f"[{dialog['speaker']}]: {dialog['text']}")
                merged_session = []
                for i in range(0, len(session), 2):
                    if i + 1 < len(session):
                        merged_session.append(session[i] + "\n" + session[i+1])
                    else:
                        merged_session.append(session[i])

                sessions.append(merged_session)
        dataform = {
            'conversation_id':entry['sample_id'],
            'qa':newqa,
            'sessions_ids':sessions_ids,
            'sessions_dates':sessions_dates,
            'sessions':sessions
            }
        alldata.append(dataform)

    out_path = os.path.join(PROCESS_DIR, "locomo10.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(alldata, f, ensure_ascii=False, indent=4)
    print(f"[done] processed locomo10 → {out_path} ({len(alldata)} items)")


def process_LongMTBench():
    ## Long-MT-Bench+
    in_data_path = os.path.join(ORIGIN_DIR, "Long-MT-Bench-Plus.json")
    if not os.path.exists(in_data_path):
        print(f"[skip] Long-MT-Bench-Plus raw data not found: {in_data_path}")
        return
    in_data = json.load(open(in_data_path))
    alldata = []
    for entry in tqdm(in_data, desc="Processing Long-MT-Bench+"):
        # print(entry.keys()) #dict_keys(['sessions', 'questions', 'conversation_id', 'turns', 'answers'])
        newqa = []
        for q, a in zip(entry['questions'],entry['answers']):
            newqa.append(
                {
                "question": q,
                "question_type": None,
                "question_date": None,
                "answer": a,
                "answer_session_ids": None,
            })
        sessions_ids = []
        sessions_dates = []
        for i, session in enumerate(entry['sessions']):
            sessions_ids.append(f'session_{i+1}')
            sessions_dates.append(None)
        dataform = {
            'conversation_id':entry['conversation_id'],
            'qa':newqa,
            'sessions_ids':sessions_ids,
            'sessions_dates':sessions_dates,
            'sessions':entry['sessions']
            }
        alldata.append(dataform)
    out_path = os.path.join(PROCESS_DIR, "Long-MT-Bench-Plus.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(alldata, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    os.makedirs(PROCESS_DIR, exist_ok=True)
    process_longmemeval()
    process_locomo10()
    process_LongMTBench()
