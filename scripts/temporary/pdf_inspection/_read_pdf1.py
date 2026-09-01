import json

with open(r"0826_pdf_output.json", encoding="utf-8-sig") as f:
    d = json.load(f)

print("TOTAL PAGES:", d["page_count"])
print("=" * 80)

# Pages 1-4: Title, abstract, intro, method
for i in range(min(4, d["page_count"])):
    p = d["pages"][i]
    print(f"\n{'='*80}")
    print(f"PAGE {p['page']} ({p['text_chars']} chars)")
    print(f"{'='*80}")
    print(p["text"][:2500])
