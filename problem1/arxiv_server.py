
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote
import json
import os
import re
import sys
from datetime import datetime
from collections import Counter

DEFAULT_PORT = 8080
DATA_DIRS = [
    os.path.join(os.path.dirname(__file__), "sample_data"),
    os.path.dirname(__file__),
]

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def find_file(name):
    for d in DATA_DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None

def tokenize(text):
    if not text:
        return []
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())

def simple_abstract_stats(abstract):
    tokens = tokenize(abstract)
    sentences = [s for s in re.split(r"[.!?]+", abstract or "") if s.strip()]
    return {
        "total_words": len(tokens),
        "unique_words": len(set(tokens)),
        "total_sentences": len(sentences),
    }

def compute_corpus_stats(papers):
    total_papers = len(papers)
    word_counter = Counter()
    category_counter = Counter()

    for p in papers:
        word_counter.update(tokenize(p.get("abstract", "")))
        for c in p.get("categories", []):
            category_counter[c] += 1

    top_10 = [{"word": w, "frequency": n} for w, n in word_counter.most_common(10)]
    return {
        "total_papers": total_papers,
        "total_words": sum(word_counter.values()),
        "unique_words": len(word_counter),
        "top_10_words": top_10,
        "category_distribution": dict(category_counter),
    }

def load_data():
    papers_path = find_file("papers.json")
    papers = load_json(papers_path) or []
    for p in papers:
        p.setdefault("arxiv_id", p.get("id", ""))
        p.setdefault("title", "")
        p.setdefault("authors", [])
        p.setdefault("categories", [])
        if "abstract" not in p:
            p["abstract"] = p.get("summary", "")
        if "abstract_stats" not in p:
            p["abstract_stats"] = simple_abstract_stats(p.get("abstract", ""))

    corpus_path = find_file("corpus_analysis.json")
    corpus = load_json(corpus_path)
    if not corpus:
        corpus = compute_corpus_stats(papers)

    index = {p.get("arxiv_id"): p for p in papers if p.get("arxiv_id")}
    return papers, index, corpus

PAPERS, PAPER_INDEX, CORPUS = load_data()

def search_papers(query_text):
    terms = [t for t in tokenize(query_text) if t]
    if not terms:
        return []

    results = []
    for p in PAPERS:
        title_tokens = tokenize(p.get("title", ""))
        abs_tokens = tokenize(p.get("abstract", ""))
        pool = title_tokens + abs_tokens

        if all(t in pool for t in terms):
            cnt = Counter(pool)
            score = sum(cnt[t] for t in terms)

            where = []
            if any(t in title_tokens for t in terms):
                where.append("title")
            if any(t in abs_tokens for t in terms):
                where.append("abstract")

            results.append({
                "arxiv_id": p.get("arxiv_id"),
                "title": p.get("title"),
                "match_score": int(score),
                "matches_in": where or ["abstract"],
            })

    results.sort(key=lambda r: (-r["match_score"], r["title"].lower()))
    return results

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class Handler(BaseHTTPRequestHandler):
    last_count = None
    last_status = None

    def send_json(self, obj, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))
        self.last_status = f"{status} {'OK' if status == 200 else self.responses.get(status, [''])[0]}"

    def send_error_json(self, status, message):
        self.send_json({"error": {"status": status, "message": message}}, status=status)

    def log_message(self, fmt, *args):
        suffix = ""
        if self.last_count is not None:
            suffix = f" ({self.last_count} results)"
        print(f"[{now_str()}] {self.command} {self.path} - {self.last_status or '-'}{suffix}")

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            parts = [unquote(p) for p in path.split("/") if p]

            if path == "/papers":
                items = []
                for p in PAPERS:
                    items.append({
                        "arxiv_id": p.get("arxiv_id"),
                        "title": p.get("title"),
                        "authors": p.get("authors", []),
                        "categories": p.get("categories", []),
                    })
                self.last_count = len(items)
                self.send_json(items, 200)
                return

            if len(parts) == 2 and parts[0] == "papers":
                pid = parts[1]
                paper = PAPER_INDEX.get(pid)
                if not paper:
                    self.last_count = 0
                    self.send_error_json(404, "Paper not found")
                    return
                self.last_count = 1
                self.send_json(paper, 200)
                return

            if path == "/search":
                qs = parse_qs(parsed.query)
                q = (qs.get("q") or [""])[0].strip()
                if not q:
                    self.last_count = 0
                    self.send_error_json(400, "Missing query parameter 'q'")
                    return
                results = search_papers(q)
                payload = {"query": q, "results": results}
                self.last_count = len(results)
                self.send_json(payload, 200)
                return

            if path == "/stats":
                self.last_count = 1
                self.send_json(CORPUS, 200)
                return

            self.last_count = 0
            self.send_error_json(404, "Endpoint not found")

        except Exception:
            self.last_count = 0
            self.send_error_json(500, "Internal server error")

def main():
    port = DEFAULT_PORT
    if len(sys.argv) >= 2:
        if not re.fullmatch(r"\d+", sys.argv[1] or ""):
            print("Error: Port must be numeric", file=sys.stderr)
            sys.exit(1)
        port = int(sys.argv[1])
        if port < 1024 or port > 65535:
            print("Error: Port must be between 1024 and 65535", file=sys.stderr)
            sys.exit(1)

    httpd = ThreadingHTTPServer(("", port), Handler)

    print(f"ArXiv API server listening on port {port}")
    print("Available endpoints:")
    print("  GET /papers")
    print("  GET /papers/{arxiv_id}")
    print("  GET /search?q={query}")
    print("  GET /stats")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

if __name__ == "__main__":
    main()
