from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


GROUPS = (
    ("mae", "Absolute MAE", "figure4_absolute_mae_tasks"),
    ("ratio", "Degradation Ratio", "figure4_ratio_tasks"),
    ("random", "Random-CV", "random_cv_mae"),
)


def build_gallery(figures_dir: str | Path, output_path: str | Path) -> dict[str, object]:
    figures = Path(figures_dir).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    groups: list[dict[str, object]] = []
    image_count = 0
    for key, label, directory_name in GROUPS:
        directory = figures / directory_name
        images = []
        if directory.exists():
            for path in sorted(directory.glob("*.png")):
                relative = os.path.relpath(path, output.parent).replace("\\", "/")
                images.append(
                    {
                        "name": _display_name(path.stem),
                        "src": relative,
                        "filename": path.name,
                    }
                )
        image_count += len(images)
        groups.append({"key": key, "label": label, "images": images})
    payload = json.dumps(groups, ensure_ascii=True, separators=(",", ":"))
    output.write_text(_page(payload, image_count), encoding="utf-8", newline="\n")
    return {"output": str(output), "image_count": image_count, "groups": groups}


def _display_name(stem: str) -> str:
    if stem == "random_cv_mae_all_tasks":
        return "All eight tasks"
    return " / ".join(part.replace("_", " ") for part in stem.split("__"))


def _page(payload: str, image_count: int) -> str:
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Figure 4 Image Gallery</title>
  <style>
    :root {{
      color-scheme: light;
      --paper: #f3f4f1;
      --surface: #ffffff;
      --ink: #18212a;
      --muted: #68737d;
      --line: #d7dde1;
      --blue: #356f9f;
      --coral: #d0525f;
      --green: #238b7e;
      --shadow: 0 8px 24px rgba(23, 34, 44, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    html {{ background: var(--paper); }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: Aptos, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    button, a {{ font: inherit; }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 20;
      display: grid;
      grid-template-columns: minmax(240px, 1fr) auto;
      align-items: center;
      gap: 24px;
      padding: 18px clamp(18px, 4vw, 56px) 14px;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(24px, 3vw, 38px);
      font-weight: 600;
      line-height: 1.05;
    }}
    .status {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 9px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    .status-dot {{
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 0 4px rgba(35, 139, 126, 0.12);
    }}
    .tabs {{
      display: flex;
      gap: 4px;
      padding: 0 clamp(18px, 4vw, 56px);
      background: var(--surface);
      border-bottom: 1px solid var(--line);
      overflow-x: auto;
    }}
    .tab {{
      appearance: none;
      min-height: 48px;
      padding: 0 16px;
      border: 0;
      border-bottom: 3px solid transparent;
      color: var(--muted);
      background: transparent;
      cursor: pointer;
      font-weight: 650;
    }}
    .tab[aria-selected="true"] {{ color: var(--ink); border-bottom-color: var(--blue); }}
    .tab-count {{ margin-left: 7px; color: var(--muted); font-weight: 500; }}
    main {{ padding: 28px clamp(18px, 4vw, 56px) 60px; }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 22px;
      max-width: 1800px;
      margin: 0 auto;
    }}
    .gallery.single {{ grid-template-columns: minmax(0, 1fr); max-width: 1500px; }}
    .figure-card {{
      min-width: 0;
      overflow: hidden;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: var(--shadow);
    }}
    .card-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 48px;
      padding: 11px 14px;
      border-bottom: 1px solid var(--line);
    }}
    .card-title {{ margin: 0; font-size: 14px; font-weight: 700; overflow-wrap: anywhere; }}
    .open-link {{ color: var(--blue); text-decoration: none; font-size: 12px; white-space: nowrap; }}
    .open-link:hover {{ text-decoration: underline; }}
    .figure-frame {{
      display: grid;
      place-items: center;
      width: 100%;
      min-height: 260px;
      aspect-ratio: 1.587;
      padding: 8px;
      background: #ffffff;
    }}
    .single .figure-frame {{ aspect-ratio: 2 / 1; }}
    .figure-image {{
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
      cursor: zoom-in;
    }}
    .empty {{ grid-column: 1 / -1; padding: 72px 24px; color: var(--muted); text-align: center; }}
    dialog {{
      width: min(96vw, 1800px);
      max-width: none;
      padding: 0;
      border: 1px solid #bfc7cd;
      border-radius: 6px;
      background: #ffffff;
      box-shadow: 0 24px 80px rgba(20, 28, 36, 0.24);
    }}
    dialog::backdrop {{ background: rgba(18, 24, 30, 0.74); }}
    .dialog-head {{ display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; border-bottom: 1px solid var(--line); }}
    .dialog-title {{ font-weight: 700; }}
    .close-button {{ width: 36px; height: 36px; border: 0; background: transparent; color: var(--ink); font-size: 25px; cursor: pointer; }}
    .dialog-image {{ display: block; width: 100%; max-height: calc(100vh - 90px); object-fit: contain; background: #ffffff; }}
    @media (max-width: 900px) {{
      .topbar {{ grid-template-columns: 1fr; gap: 8px; }}
      .status {{ justify-content: flex-start; }}
      .gallery {{ grid-template-columns: 1fr; }}
      .figure-frame {{ min-height: 200px; }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <h1>Figure 4 Image Gallery</h1>
    <div class="status"><span class="status-dot"></span><span id="statusText">{image_count} PNG files · live refresh</span></div>
  </header>
  <nav class="tabs" id="tabs" aria-label="Figure groups"></nav>
  <main><section class="gallery" id="gallery" aria-live="polite"></section></main>
  <dialog id="viewer">
    <div class="dialog-head"><span class="dialog-title" id="viewerTitle"></span><button class="close-button" id="closeViewer" aria-label="Close">×</button></div>
    <img class="dialog-image" id="viewerImage" alt="Expanded figure">
  </dialog>
  <script>
    const groups = {payload};
    const tabs = document.getElementById('tabs');
    const gallery = document.getElementById('gallery');
    const viewer = document.getElementById('viewer');
    const viewerImage = document.getElementById('viewerImage');
    const viewerTitle = document.getElementById('viewerTitle');
    const visibleImages = new Set();
    let activeKey = groups[0]?.key || '';

    const observer = new IntersectionObserver((entries) => {{
      entries.forEach((entry) => {{
        if (entry.isIntersecting) {{
          visibleImages.add(entry.target);
          refreshImage(entry.target);
        }} else {{
          visibleImages.delete(entry.target);
        }}
      }});
    }}, {{ rootMargin: '240px' }});

    function versioned(source) {{
      const separator = source.includes('?') ? '&' : '?';
      return `${{source}}${{separator}}v=${{Date.now()}}`;
    }}

    function refreshImage(image) {{
      if (!image?.dataset.source || image.dataset.loading === 'true') return;
      image.dataset.loading = 'true';
      const candidate = new Image();
      candidate.onload = () => {{
        image.src = candidate.src;
        image.dataset.loading = 'false';
      }};
      candidate.onerror = () => {{ image.dataset.loading = 'false'; }};
      candidate.src = versioned(image.dataset.source);
    }}

    function renderTabs() {{
      tabs.innerHTML = groups.map((group) => `
        <button class="tab" data-key="${{group.key}}" aria-selected="${{group.key === activeKey}}">
          ${{group.label}}<span class="tab-count">${{group.images.length}}</span>
        </button>`).join('');
      tabs.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => {{
        activeKey = tab.dataset.key;
        renderTabs();
        renderGallery();
      }}));
    }}

    function renderGallery() {{
      observer.disconnect();
      visibleImages.clear();
      const group = groups.find((item) => item.key === activeKey);
      gallery.classList.toggle('single', group?.key === 'random');
      if (!group || !group.images.length) {{
        gallery.innerHTML = '<div class="empty">No PNG files found.</div>';
        return;
      }}
      gallery.innerHTML = group.images.map((image) => `
        <article class="figure-card">
          <div class="card-head"><h2 class="card-title">${{image.name}}</h2><a class="open-link" href="${{image.src}}" target="_blank">Open PNG</a></div>
          <div class="figure-frame"><img class="figure-image" data-source="${{image.src}}" alt="${{image.name}}"></div>
        </article>`).join('');
      gallery.querySelectorAll('.figure-image').forEach((image) => {{
        observer.observe(image);
        image.addEventListener('click', () => {{
          viewerTitle.textContent = image.alt;
          viewerImage.src = versioned(image.dataset.source);
          viewer.showModal();
        }});
      }});
    }}

    document.getElementById('closeViewer').addEventListener('click', () => viewer.close());
    viewer.addEventListener('click', (event) => {{ if (event.target === viewer) viewer.close(); }});
    setInterval(() => visibleImages.forEach(refreshImage), 5000);
    renderTabs();
    renderGallery();
  </script>
</body>
</html>
'''


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static English Figure 4 PNG gallery.")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("output/training_size_ood/figures"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/training_size_ood/figure_gallery/index.html"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_gallery(args.figures_dir, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
