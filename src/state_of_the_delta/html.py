import base64
import mimetypes
from html import escape
from pathlib import Path

import markdown

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>Memo</title>
	<style>
	:root {{
		--background: #ffffff;
		--text-color: #37352f;
		--heading-color: #0A28A3;
		--control-border: #e5e5e5;
		--link-color: #0969da;
	}}
	body.darkmode {{
		--background: #1e1e1e;
		--text-color: #e6e6e6;
		--heading-color: #A1A2C7;
		--control-border: #444444;
		--link-color: #6cb6ff;
	}}
	body {{
		font-family: "Segoe UI", sans-serif;
		background-color: var(--background);
		color: var(--text-color);
		line-height: 1.6;
		margin: 0;
		padding: 40px 20px;
	}}
	.memo {{ max-width: 900px; margin: 0 auto; }}
	.memo-header, .memo-footer {{
		align-items: center;
		display: flex;
		justify-content: space-between;
	}}
	.memo-header {{ margin: 0 0 0.75rem; }}
	.memo-footer {{ margin: 2rem 0 0; }}
	.memo-header img, .memo-footer img {{
		display: block;
		height: auto;
		max-width: 240px;
	}}
	h1, h2, h3 {{ color: var(--heading-color); line-height: 1.3; margin: 1.5em 0 0.5em; }}
	h1 {{ margin-top: 0.5em; }}
	p, ul, ol {{ margin-bottom: 1em; }}
	a {{ color: var(--link-color); }}
	.theme-toggle {{
		align-items: center;
		background: transparent;
		border: 1px solid var(--control-border);
		border-radius: 50%;
		color: var(--text-color);
		cursor: pointer;
		display: inline-flex;
		font-size: 1.125rem;
		height: 2.25rem;
		justify-content: center;
		padding: 0;
		width: 2.25rem;
	}}
	.theme-toggle:hover {{ background: var(--control-border); }}
	.theme-icon-moon {{ display: none; }}
	body.darkmode .theme-icon-sun {{ display: none; }}
	body.darkmode .theme-icon-moon {{ display: inline; }}
	.memo-content img {{
		display: block;
		max-width: 100%;
		height: auto;
		margin: 1.5em 0;
		border-radius: 8px;
	}}
	.memo-ai-content-logo {{ margin: 2rem 0; }}
	.memo-ai-content-logo img {{ display: block; height: auto; max-width: 120px; }}
	@media (max-width: 480px) {{
		.memo-header img, .memo-footer img {{ max-width: 45%; }}
	}}
	</style>
</head>
<body>
	<main class="memo">
		<header class="memo-header">
			<button class="theme-toggle" type="button" aria-label="Toggle dark mode" title="Toggle dark mode">
				<span class="theme-icon-sun" aria-hidden="true">&#9788;</span>
				<span class="theme-icon-moon" aria-hidden="true">&#9790;</span>
			</button>
			<img src="{header_logo}" alt="Deltares">
		</header>
		<div class="memo-content">
		{html_content}
		</div>
		<footer class="memo-footer">
			<img src="{footer_logo}" alt="Enabling Delta Life">
			<img class="memo-ai-content-logo" src="{ai_content_logo}" alt="AI-generated content">
		</footer>
	</main>
	<script>
	const themeToggle = document.querySelector(".theme-toggle");
	themeToggle.addEventListener("click", () => {{
		document.body.classList.toggle("darkmode");
	}});
	</script>
</body>
</html>
"""


def image_data_uri(file_path: Path) -> str:
    """Return an image file as a base64 data URI."""
    mime_type = mimetypes.guess_type(file_path)[0] or "image/png"
    img_base64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{img_base64}"


def _logo_path(file_name: str) -> Path:
    """Return the path of a bundled Deltares logo asset."""
    return Path(__file__).parents[2] / "data" / "99_logos" / file_name


def render_memo_html(markdown_content: str) -> str:
    """Render Markdown memo content as a complete HTML document."""
    html_content = markdown.markdown(markdown_content, extensions=["extra"])

    return HTML_TEMPLATE.format(
        html_content=html_content,
        header_logo=image_data_uri(_logo_path("Deltares logo.svg")),
        ai_content_logo=image_data_uri(_logo_path("Deltares AI.svg")),
        footer_logo=image_data_uri(_logo_path("Deltares pay off.svg")),
    )


def render_memo_iframe(markdown_content: str) -> str:
	"""Render a memo in an isolated iframe for notebook display."""
	document = render_memo_html(markdown_content)
	return (
		'<iframe sandbox="allow-scripts" '
		'style="border: 0; height: 800px; width: 100%;" '
		f'srcdoc="{escape(document, quote=True)}" '
		'title="Memo preview"></iframe>'
	)
