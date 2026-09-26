# Trin 1: Hent teksten fra PDF-manualen, og gem den som en almindelig tekstfil.
# PDF'en skal ligge i den mappe, du kører programmet fra.
# Dette trin bruger ikke en AI-model. PDF'en skal indeholde læsbar tekst;
# billeder af tekst kræver en særskilt tekstgenkendelse (OCR).
import pdfplumber
from pathlib import Path
import re
import wordninja

# Filen, vi læser, og filen, som næste trin skal arbejde videre med.
pdf_file = "hp34c-ohpg-en.pdf"
output_txt = "hp34c-ohpg-en-full-clean.txt"

# Saml teksten og hold regnskab med ordene undervejs.
all_text = ""
total_words = 0
fixed_words = 0

# Åbn PDF'en, og behandl én side ad gangen. Filen lukkes automatisk bagefter.
with pdfplumber.open(pdf_file) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        # Tolerancerne styrer, hvor tæt bogstaver skal stå for at høre sammen.
        words = page.extract_words(x_tolerance=1, y_tolerance=1)
        total_words += len(words)
        
        line_words = []
        for word in words:
            txt = word['text']
            # PDF-udtræk kan klistre ord sammen. Ved mere end 20 tegn prøver
            # wordninja at gætte en opdeling ud fra engelske ord.
            # Mønsteret i re.match undtager rene linjer af punktum, bindestreg,
            # understregning og lighedstegn. Gættet kan stadig dele rigtige ord forkert.
            if len(txt) > 20 and not re.match(r'^[\.\-_=]{5,}$', txt):
                split = wordninja.split(txt)
                fixed_words += 1
                print(f"⚠️ Page {page_num}: '{txt}' split into {split}")
                line_words.extend(split)
            else:
                line_words.append(txt)

        # Gem hver sides ord som én linje; PDF'ens oprindelige layout bevares ikke.
        line = " ".join(line_words)
        all_text += line + "\n"

        # Vis et kort tekstudsnit hver femte side og på sider med meget lidt tekst.
        if page_num % 5 == 0 or len(words) < 10:
            print(f"📄 Page {page_num}: {len(words)} words")
            print(f"    ➡️ '{line[:100]}...'")

print(f"\n✅ Done! Total {total_words} words processed, {fixed_words} long words split using wordninja.")
# UTF-8 er tekstformatet, som også bruges, når filen læses i næste trin.
# En eksisterende fil med samme navn bliver overskrevet.
Path(output_txt).write_text(all_text, encoding="utf-8")
print(f"🚀 Cleaned text saved to: {output_txt}")
