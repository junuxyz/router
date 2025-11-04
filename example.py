"""
RTX 3050 Ti용 최소 수학 평가 스크립트 (디버깅 버전)
모델: Qwen2-1.5B-Instruct (약 3GB VRAM)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re

print("="*80)
print("🚀 평가 시작")
print("="*80)

# ============================================================================
# 1. 모델 로드
# ============================================================================
model_name = "Qwen/Qwen2-1.5B-Instruct"
# Ohter options: 
# - "microsoft/phi-2" (2.7B)
# - "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# - "google/gemma-2b-it"

print(f"\n📦 모델 로딩 중: {model_name}")
print(f"   - CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   - GPU: {torch.cuda.get_device_name(0)}")
    print(f"   - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16, # for memory efficiency
    device_map="auto",
    low_cpu_mem_usage=True
)

print(f"✅ 모델 로드 완료")
print(f"   - 파라미터 수: {model.num_parameters() / 1e9:.2f}B")
print(f"   - 디바이스: {model.device}")

# ============================================================================
# 2. 데이터 로드
# ============================================================================
print(f"\n📊 데이터셋 로딩 중...")
dataset = load_dataset("openai/gsm8k", "main", split="test[:5]")  # 처음 5개만 테스트
print(f"✅ 데이터 로드 완료: {len(dataset)}개 문제")

# ============================================================================
# 3. 답변 추출 함수
# ============================================================================
def extract_answer(text):
    """생성된 텍스트에서 숫자 추출"""
    print(f"      🔍 답변 추출 중...")
    
    # 패턴 1: "answer is X"
    match = re.search(r'answer is[:\s]+(-?\d+\.?\d*)', text.lower())
    if match:
        answer = match.group(1)
        print(f"         ✓ 'answer is' 패턴에서 발견: {answer}")
        return answer
    
    # 패턴 2: 마지막 숫자
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        answer = numbers[-1]
        print(f"         ✓ 마지막 숫자 사용: {answer}")
        return answer
    
    print(f"         ✗ 숫자를 찾을 수 없음")
    return None

def get_gold_answer(answer_text):
    """GSM8K의 정답 추출 (#### 뒤의 숫자)"""
    match = re.search(r'#### (.+)', answer_text)
    if match:
        return match.group(1).strip().replace(',', '')
    return None

# ============================================================================
# 4. 평가 루프
# ============================================================================
print(f"\n{'='*80}")
print("🔄 평가 시작")
print(f"{'='*80}\n")

correct = 0
results = []

for idx, example in enumerate(dataset):
    print(f"[{idx+1}/{len(dataset)}] " + "="*70)
    
    # Step 1: 문제 확인
    question = example['question']
    print(f"❓ 문제: {question[:100]}...")
    
    # Step 2: 토큰화
    print(f"\n   ⚙️  토큰화 중...")
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    print(f"      - 입력 토큰 수: {input_length}")
    
    # Step 3: 생성
    print(f"   🤖 답변 생성 중...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    output_length = outputs.shape[1]
    generated_length = output_length - input_length
    print(f"      - 생성 토큰 수: {generated_length}")
    
    # Step 4: 디코딩
    prediction = tokenizer.decode(
        outputs[0][input_length:], 
        skip_special_tokens=True
    )
    print(f"\n   💬 생성된 답변:")
    print(f"      {prediction[:200]}...")
    
    # Step 5: 답변 추출
    pred_answer = extract_answer(prediction)
    gold_answer = get_gold_answer(example['answer'])
    print(f"\n   📝 정답 추출:")
    print(f"      - 정답 (Gold): {gold_answer}")
    print(f"      - 예측 (Pred): {pred_answer}")
    
    # Step 6: 평가
    is_correct = 1 if pred_answer == gold_answer else 0
    correct += is_correct
    
    emoji = "✅" if is_correct else "❌"
    print(f"\n   {emoji} 결과: {'정답' if is_correct else '오답'}")
    print(f"   📊 현재 정확도: {correct}/{idx+1} = {correct/(idx+1)*100:.1f}%")
    
    results.append({
        "idx": idx,
        "question": question,
        "prediction": prediction,
        "pred_answer": pred_answer,
        "gold_answer": gold_answer,
        "label": is_correct
    })
    
    print()

# ============================================================================
# 5. 최종 결과
# ============================================================================
print(f"{'='*80}")
print("📈 최종 결과")
print(f"{'='*80}")

accuracy = correct / len(dataset)
print(f"\n✨ 정확도: {accuracy:.2%} ({correct}/{len(dataset)})")

print(f"\n📋 상세 결과:")
for r in results:
    emoji = "✅" if r['label'] else "❌"
    print(f"  {emoji} 문제 {r['idx']+1}: Pred={r['pred_answer']}, Gold={r['gold_answer']}")

# JSON 저장
import json
with open('debug_results.json', 'w', encoding='utf-8') as f:
    json.dump({
        "model": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(dataset),
        "predictions": results
    }, f, indent=2, ensure_ascii=False)

print(f"\n💾 결과 저장: debug_results.json")
print(f"\n{'='*80}")
print("✅ 평가 완료!")
print(f"{'='*80}")
