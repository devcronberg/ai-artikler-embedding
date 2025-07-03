import pdfplumber
from pathlib import Path
import re
import wordninja

pdf_file = "hp34c-ohpg-en.pdf"
output_txt = "hp34c-ohpg-en-full-clean.txt"

all_text = ""
total_words = 0
fixed_words = 0

with pdfplumber.open(pdf_file) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        words = page.extract_words(x_tolerance=1, y_tolerance=1)
        total_words += len(words)
        
        line_words = []
        for word in words:
            txt = word['text']
            if len(txt) > 20 and not re.match(r'^[\.\-_=]{5,}$', txt):
                split = wordninja.split(txt)
                fixed_words += 1
                print(f"⚠️ Page {page_num}: '{txt}' split into {split}")
                line_words.extend(split)
            else:
                line_words.append(txt)

        line = " ".join(line_words)
        all_text += line + "\n"

        if page_num % 5 == 0 or len(words) < 10:
            print(f"📄 Page {page_num}: {len(words)} words")
            print(f"    ➡️ '{line[:100]}...'")

print(f"\n✅ Done! Total {total_words} words processed, {fixed_words} long words split using wordninja.")
Path(output_txt).write_text(all_text, encoding="utf-8")
print(f"🚀 Cleaned text saved to: {output_txt}")
