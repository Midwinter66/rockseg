import json

with open(r"0826_pdf_output.json", encoding="utf-8-sig") as f:
    d = json.load(f)

# Pages 5-9: method details, results, discussion
for i in range(4, d["page_count"]):
    p = d["pages"][i]
    print(f"\n{'='*80}")
    print(f"PAGE {p['page']} ({p['text_chars']} chars)")
    print(f"{'='*80}")
    print(p["text"][:3000])
