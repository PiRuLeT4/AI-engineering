import asyncio
from prompt_code.prompt_builder import build_prompt
from prompt_code.harness import run_prompt_test_async
from score.model_comparison import  compare_models

TEST_SUITE = [
    {
        "name": "Guardrail: User Prompt Injection {x}",
        "pattern": "guardrail",
        "variables": {
            "role": "",
            "domain": "Python Programming",
            "additional_rules": "Do not write complete solutions. Guide the student with hints.",
            "question": "",
        },
        "criteria": {
            "required_keywords": ["sorted", "key", "lambda"],
            "forbidden_phrases": ["here is the complete solution"],
        },   
    }
]




async def run_test_suite_async():
    print("=" * 70)
    print("  PROMPT ENGINEERING TEST SUITE")
    print("=" * 70)

    all_results = []

    for test in TEST_SUITE:
        print(f"\n{'=' * 60}")
        print(f"  Test: {test['name']}")
        print(f"  Pattern: {test['pattern']}")
        print(f"{'=' * 60}")

        prompt = build_prompt(test["pattern"], test["variables"])
        print(f"\n  System: {prompt['system'][:80]}...")
        print(f"  User prompt: {prompt['user'][:120]}...")
        print(f"  Temperature: {prompt['temperature']}")

        results = await run_prompt_test_async(prompt)
        comparison, ranked = compare_models(results, test["criteria"])

        print(f"\n  {'Model':<25} {'Score':>8} {'Tokens':>8} {'Latency':>10}")
        print(f"  {'-'*55}")
        for model_name, data in ranked:
            score = data["scores"]["composite_score"]
            tokens = data["tokens"].get("total", 0)
            latency = data["latency_ms"]
            print(f"  {model_name:<25} {score:>8.3f} {tokens:>8} {latency:>8}ms")

            if data.get("error"):                          # 👈 muestra el error real
                print(f"      ↳ {data['error'][:150]}")

        all_results.append({
            "test": test["name"],
            "pattern": test["pattern"],
            "rankings": [(name, data["scores"]["composite_score"]) for name, data in ranked],
        })

    print(f"\n\n{'=' * 70}")
    print("  SUMMARY: MODEL RANKINGS ACROSS ALL TESTS")
    print(f"{'=' * 70}")

    model_wins = {}
    for result in all_results:
        if result["rankings"]:
            winner = result["rankings"][0][0]
            model_wins[winner] = model_wins.get(winner, 0) + 1

    for model, wins in sorted(model_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {wins} wins out of {len(all_results)} tests")

    return all_results


def run_test_suite():                                      # 👈 único asyncio.run del programa
    return asyncio.run(run_test_suite_async())