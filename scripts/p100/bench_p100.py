import json, time, urllib.request, threading, statistics

BASE = "http://localhost:8011/v1/completions"
MODEL = "meta-llama/Llama-3.1-8B"
PROMPT = ("Write a detailed technical explanation of how modern GPUs and AI "
          "accelerators schedule matrix multiplications across cores, including "
          "memory hierarchy considerations. Be thorough.\n\n")

def post(body, stream=False, timeout=300):
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def single_stream(max_tokens=200):
    body = {"model": MODEL, "prompt": PROMPT, "max_tokens": max_tokens,
            "temperature": 0, "stream": True}
    t0 = time.perf_counter(); ttft = None; ntok = 0; tlast = t0
    resp = post(body, stream=True)
    for raw in resp:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"): continue
        data = line[len("data:"):].strip()
        if data == "[DONE]": break
        obj = json.loads(data)
        txt = obj["choices"][0].get("text", "")
        if txt:
            now = time.perf_counter()
            if ttft is None: ttft = now - t0
            ntok += 1; tlast = now
    decode_s = tlast - (t0 + (ttft or 0))
    return ttft, ntok, decode_s

def one_blocking(max_tokens, out, idx):
    body = {"model": MODEL, "prompt": PROMPT, "max_tokens": max_tokens, "temperature": 0}
    t0 = time.perf_counter()
    r = post(body); d = json.loads(r.read())
    dt = time.perf_counter() - t0
    ct = d.get("usage", {}).get("completion_tokens")
    out[idx] = (ct, dt)

def concurrency(n, max_tokens=128):
    out = [None]*n; threads = []
    t0 = time.perf_counter()
    for i in range(n):
        th = threading.Thread(target=one_blocking, args=(max_tokens, out, i)); th.start(); threads.append(th)
    for th in threads: th.join()
    wall = time.perf_counter() - t0
    toks = sum(c for c,_ in out if c)
    return toks, wall, out

print("== single-stream (concurrency 1) ==")
res = [single_stream(200) for _ in range(3)]  # 1 warm + report best/median
for ttft, ntok, dec in res:
    tps = (ntok-1)/dec if dec>0 else 0
    print(f"  TTFT={ttft*1000:.0f} ms  out_tokens={ntok}  decode_tok/s={tps:.2f}")
med_tps = statistics.median([(n-1)/d for _,n,d in res if d>0])
print(f"  --> median single-user decode: {med_tps:.2f} tok/s/user")

for n in (2, 4, 8, 16):
    toks, wall, out = concurrency(n, 128)
    agg = toks/wall if wall>0 else 0
    ok = sum(1 for c,_ in out if c)
    print(f"== concurrency {n}: {ok}/{n} ok, {toks} out tokens in {wall:.2f}s -> aggregate {agg:.1f} tok/s ({agg/max(ok,1):.2f}/user) ==")
