#!/usr/bin/env python3
"""Select N SWE-bench Pro instances per language, weighted toward enterprise
app development: web/API, multi-user & auth, crypto, compliance, LLM use."""
import json, re, sys, collections

SRC = sys.argv[1]; PER_LANG = int(sys.argv[2]) if len(sys.argv) > 2 else 10

# issue_categories carry the strongest signal (human-labelled)
CAT_W = {
    "authentication_authorization_knowledge": 4.0,  # multiuser
    "security_knowledge": 3.5,                      # crypto / compliance
    "api_knowledge": 3.0,                           # web-based / fastapi-shaped
    "web_knowledge": 2.5,
    "ml_ai_knowledge": 4.0,                         # llm use
    "database_knowledge": 1.5,
    "back_end_knowledge": 1.0,
    "networking_knowledge": 1.0,
    "cloud_knowledge": 0.5,
    "full_stack_knowledge": 0.5,
    "ui_ux_knowledge": -1.0,                        # steer away from pure UI
    "desktop_knowledge": -2.0,
    "accessibility_knowledge": -1.0,
    "mobile_knowledge": -1.0,
}

# problem_statement keyword themes -> the user's stated requirements
THEMES = {
    "web/api":    r"\b(http|https|rest|api|endpoint|route|router|request|response|middleware|handler|cors|header|payload|webhook|graphql|openapi|swagger|fastapi|asgi|uvicorn|pydantic|flask|django)\b",
    "multiuser":  r"\b(auth|authn|authz|login|logout|session|token|jwt|oauth|sso|saml|permission|role|rbac|acl|tenant|multi-?user|account|credential|password|access control)\b",
    "crypto":     r"\b(crypt|encrypt|decrypt|cipher|aes|rsa|tls|ssl|certificate|x509|hash|hmac|signature|sign|verify|key ?pair|keystore|nonce|e2ee|pgp|secret)\b",
    "compliance": r"\b(audit|compliance|gdpr|hipaa|soc ?2|policy|retention|privacy|pii|licence|license|cve|vulnerab|scan|report|governance|regulat)\b",
    "llm":        r"\b(llm|language model|openai|anthropic|prompt|embedding|inference|completion|token(?:iz|s? limit)|ai model)\b",
}
THEME_W = {"web/api": 1.0, "multiuser": 2.0, "crypto": 2.0, "compliance": 1.5, "llm": 3.0}

rows = [json.loads(l) for l in open(SRC)]

def cats(d):
    c = d.get("issue_categories")
    if isinstance(c, str):
        try: c = json.loads(c)
        except Exception: c = [c]
    return c or []

scored = collections.defaultdict(list)
for d in rows:
    text = " ".join(str(d.get(k) or "") for k in ("problem_statement", "requirements", "interface")).lower()
    cat_score = sum(CAT_W.get(c, 0) for c in cats(d))
    hits, theme_score = {}, 0.0
    for name, pat in THEMES.items():
        n = len(re.findall(pat, text))
        if n:
            hits[name] = n
            theme_score += THEME_W[name] * min(n, 6) / 6.0 * 3.0   # saturating
    scored[d["repo_language"]].append({
        "score": round(cat_score + theme_score, 2),
        "instance_id": d["instance_id"], "repo": d["repo"],
        "themes": hits, "cats": [c.replace("_knowledge", "") for c in cats(d)],
        "title": (d.get("problem_statement") or "").strip().split("\n")[0][:95],
    })

picked = {}
for lang, items in scored.items():
    items.sort(key=lambda x: -x["score"])
    # cap 60% from any single repo so one codebase can't dominate a language
    nrepos = len({i["repo"] for i in items})
    cap = PER_LANG if nrepos < 2 else max(2, int(PER_LANG * 0.6))
    per_repo, out = collections.Counter(), []
    for it in items:
        if per_repo[it["repo"]] >= cap: continue
        per_repo[it["repo"]] += 1; out.append(it)
        if len(out) == PER_LANG: break
    picked[lang] = out

for lang in sorted(picked):
    print(f"\n===== {lang.upper()}  ({len(picked[lang])} selected) =====")
    for it in picked[lang]:
        th = ",".join(f"{k}:{v}" for k, v in sorted(it["themes"].items(), key=lambda x: -x[1]))
        print(f"  {it['score']:5.1f}  {it['repo']:28s} {th}")
        print(f"         {it['title']}")

ids = [it["instance_id"] for lang in sorted(picked) for it in picked[lang]]
json.dump(picked, open(SP_OUT := sys.argv[3] if len(sys.argv) > 3 else "selection.json", "w"), indent=1)
print(f"\nTOTAL: {len(ids)} instances -> {SP_OUT}")
