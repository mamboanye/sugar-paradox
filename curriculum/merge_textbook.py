"""Merge 8 chapter HTML files into one unified textbook."""
import re
from pathlib import Path

CHAPTERS_DIR = Path("chapters")
OUTPUT = Path("textbook.html")

files = [
    "ch01_02.html", "ch03_04.html", "ch05_06.html", "ch07_08.html",
    "ch09_10.html", "ch11_12.html", "ch13_14.html", "ch15_epilogue.html",
]

def extract_body_content(html):
    """Extract content between <body> tags, stripping wrapper."""
    m = re.search(r'<body[^>]*>(.*)</body>', html, re.DOTALL)
    if m:
        content = m.group(1)
        # Remove outermost container div if present
        content = re.sub(r'^\s*<div[^>]*class="page"[^>]*>', '', content, count=1)
        content = re.sub(r'</div>\s*$', '', content, count=1)
        return content.strip()
    return html

def extract_styles(html):
    """Extract all <style> blocks."""
    return re.findall(r'<style[^>]*>(.*?)</style>', html, re.DOTALL)

def extract_scripts(html):
    """Extract all <script> blocks that aren't CDN links."""
    scripts = re.findall(r'<script(?! defer| src)[^>]*>(.*?)</script>', html, re.DOTALL)
    return [s for s in scripts if s.strip()]

# Collect all content
all_styles = []
all_bodies = []
all_scripts = []

for f in files:
    path = CHAPTERS_DIR / f
    html = path.read_text(encoding='utf-8')
    all_styles.extend(extract_styles(html))
    all_bodies.append((f, extract_body_content(html)))
    all_scripts.extend(extract_scripts(html))

# Deduplicate CSS (take unique rules)
# Just concatenate for now -- browsers handle duplicates fine
merged_css = "\n".join(all_styles)

# Count chapters and sections for TOC
print(f"Extracted {len(all_bodies)} files")
print(f"CSS blocks: {len(all_styles)}")
print(f"Script blocks: {len(all_scripts)}")
print(f"Total CSS chars: {len(merged_css)}")
print(f"Total content chars: {sum(len(b) for _, b in all_bodies)}")

# Write merged file
with open(OUTPUT, 'w', encoding='utf-8') as out:
    out.write("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Learning Regression Through the Sugar Paradox</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '\\\\(', right: '\\\\)', display: false}]})"></script>
  <style>
""")
    out.write(merged_css)
    out.write("""
    /* === UNIFIED NAVIGATION === */
    .textbook-nav {
      position: fixed; top: 0; left: 0; width: 260px; height: 100vh;
      background: #F2F0E8; border-right: 1px solid #E0DED6;
      overflow-y: auto; padding: 1.5rem 1rem; z-index: 100;
      font-family: 'Crimson Pro', Georgia, serif;
    }
    .textbook-nav h2 {
      font-size: 1rem; font-weight: 600; margin-bottom: 1rem;
      color: #2D2D2D; border: none; padding: 0;
    }
    .textbook-nav a {
      display: block; padding: 0.35rem 0.5rem; margin: 0.1rem 0;
      color: #7A7A6E; text-decoration: none; font-size: 0.82rem;
      border-radius: 3px; transition: all 0.15s; line-height: 1.3;
    }
    .textbook-nav a:hover { background: #E8E5DB; color: #2D2D2D; }
    .textbook-nav a.active { background: #C4843C; color: white; }
    .textbook-nav .nav-section {
      font-size: 0.68rem; font-weight: 600; color: #C4843C;
      text-transform: uppercase; letter-spacing: 0.08em;
      margin: 1rem 0 0.3rem 0.5rem;
    }
    .textbook-body {
      margin-left: 260px; max-width: 1100px; padding: 2rem 2rem 4rem;
    }

    /* === LANDING PAGE === */
    .landing { padding: 4rem 0 3rem; border-bottom: 2px solid #C4843C; margin-bottom: 3rem; }
    .landing h1 { font-family: 'Crimson Pro', serif; font-weight: 300; font-size: 2.6rem; letter-spacing: -0.02em; line-height: 1.15; max-width: 700px; }
    .landing .subtitle { font-size: 1.1rem; color: #7A7A6E; margin-top: 0.5rem; font-style: italic; }
    .landing-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 2.5rem 0; }
    .landing-card { background: #F2F0E8; padding: 1.3rem 1.5rem; border-radius: 6px; }
    .landing-card h3 { font-size: 1rem; margin: 0 0 0.5rem; font-weight: 600; }
    .landing-card p { font-size: 0.9rem; color: #7A7A6E; margin: 0; max-width: none; }
    .landing-stats { display: flex; gap: 2.5rem; margin: 2rem 0; }
    .landing-stat { text-align: center; }
    .landing-stat .num { font-size: 2.2rem; font-weight: 300; color: #C4843C; }
    .landing-stat .lbl { font-size: 0.75rem; color: #7A7A6E; text-transform: uppercase; letter-spacing: 0.06em; }

    @media (max-width: 900px) {
      .textbook-nav { display: none; }
      .textbook-body { margin-left: 0; }
      .landing-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>

<!-- SIDEBAR NAVIGATION -->
<nav class="textbook-nav">
  <h2>Sugar Paradox</h2>
  <a href="#landing">Home</a>
  <div class="nav-section">The Cross-Sectional Signal</div>
  <a href="#ch1">1. Vectors and Correlation</a>
  <a href="#ch2">2. Regression as Projection</a>
  <a href="#ch3">3. Partial Correlation</a>
  <a href="#ch4">4. The F-test</a>
  <div class="nav-section">The Within-Country Null</div>
  <a href="#ch5">5. Fixed Effects</a>
  <a href="#ch6">6. Variance Decomposition</a>
  <a href="#ch7">7. Serial Correlation</a>
  <a href="#ch8">8. Detrending</a>
  <div class="nav-section">The Diagnosis</div>
  <a href="#ch9">9. Mundlak/CRE</a>
  <a href="#ch10">10. Bootstrap &amp; Permutation</a>
  <a href="#ch11">11. GDP Positive Control</a>
  <a href="#ch12">12. Soybean Oil Placebo</a>
  <div class="nav-section">The Generalization</div>
  <a href="#ch13">13. Food-Group Cascade</a>
  <a href="#ch14">14. Diabetes Borderline</a>
  <a href="#ch15">15. Initial Conditions</a>
  <a href="#epilogue">Epilogue</a>
</nav>

<div class="textbook-body">

<!-- LANDING PAGE -->
<section class="landing" id="landing">
  <h1>Learning Regression Through the Sugar Paradox</h1>
  <p class="subtitle">An interactive textbook where every concept produces a real number from a real paper</p>

  <div class="landing-stats">
    <div class="landing-stat"><div class="num">15</div><div class="lbl">Chapters</div></div>
    <div class="landing-stat"><div class="num">37</div><div class="lbl">Countries</div></div>
    <div class="landing-stat"><div class="num">962</div><div class="lbl">Observations</div></div>
    <div class="landing-stat"><div class="num">1</div><div class="lbl">Paradox</div></div>
  </div>

  <div class="landing-grid">
    <div class="landing-card">
      <h3>The data</h3>
      <p>37 Sub-Saharan African countries, 2010-2022. Sugar supply, vegetable oil supply, obesity prevalence, diabetes prevalence, GDP per capita, urbanization. One row per country-year-sex. 962 observations total.</p>
    </div>
    <div class="landing-card">
      <h3>The paradox</h3>
      <p>Countries that eat more sugar are fatter (r = 0.68). But when you follow the same countries over time, sugar supply predicts nothing about obesity change. Six different within-country specifications all converge on the same answer: no signal.</p>
    </div>
    <div class="landing-card">
      <h3>The method</h3>
      <p>Each chapter teaches one regression concept from STAT 538. Each concept is illustrated by one computation on this data that produces one real number from the published paper. You learn the statistics by dismantling the paradox.</p>
    </div>
    <div class="landing-card">
      <h3>The journey</h3>
      <p>Start with a strong cross-sectional signal (t = 19.84). By Chapter 8, it's dead (t = -0.13). By Chapter 12, you understand why. By Chapter 15, you know what actually predicts obesity. The paper is the curriculum.</p>
    </div>
  </div>
</section>

""")

    # Write all chapter content
    for fname, body in all_bodies:
        out.write(f"\n<!-- ========== FROM {fname} ========== -->\n")
        out.write(body)
        out.write("\n\n")

    # Write all scripts
    for script in all_scripts:
        out.write(f"\n<script>\n{script}\n</script>\n")

    # Navigation highlight script
    out.write("""
<script>
// Highlight current chapter in sidebar
const navLinks = document.querySelectorAll('.textbook-nav a');
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      navLinks.forEach(l => l.classList.remove('active'));
      const id = entry.target.id;
      const link = document.querySelector(`.textbook-nav a[href="#${id}"]`);
      if (link) link.classList.add('active');
    }
  });
}, { rootMargin: '-20% 0px -70% 0px' });

document.querySelectorAll('section[id], .chapter[id], [id^="ch"], [id="landing"], [id="epilogue"]').forEach(s => observer.observe(s));
</script>
""")

    out.write("\n</div>\n</body>\n</html>")

print(f"\nWritten: {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
