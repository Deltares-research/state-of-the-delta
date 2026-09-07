import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import markdown
from dotenv import load_dotenv
from IPython.display import HTML, display
from openai import AzureOpenAI

from state_of_the_delta.rag import get_retriever

type MessagePart = dict[str, Any]
type ChatMessage = dict[str, Any]

DEFAULT_PROMPT = """
*General Instructions*
Generate a memo based on the provided template for the specified area of interest using the supplied figures and datasets.
Use Markdown formatting. Do not invent, estimate, or infer information that is not supported by the supplied data.

*Figure references*
Figures must be referenced as plain text using the format "Figure #".
For example, within the text write: "Figure 1 shows ..." or "... is shown in Figure 1."
Then place the figure placeholder on a separate line: [Figure 1]

*Datasets references*
Do not reference datasets in the memo. The datasets are here to provide additional context to support the memo generation.

*Quality Assurance*
Before returning the generated memo, ensure that:
- All figures are correctly referenced and included.
- The locations of interest are clearly described.
- The conclusion accurately reflects the observations from the figures.
- No unsupported information is added.
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memo</title>
    <style>
    body {{
        font-family: "Segoe UI", sans-serif;
        background-color: {background};
        color: {text_color};
        line-height: 1.6;
        margin: 0;
        padding: 40px 20px;
    }}
    .memo {{ max-width: 900px; margin: 0 auto; }}
    h1 {{ border-bottom: 3px solid {heading_border}; padding-bottom: 0.5em; }}
    h1, h2, h3 {{ line-height: 1.3; margin: 1.5em 0 0.5em; }}
    p, ul, ol {{ margin-bottom: 1em; }}
    a {{ color: {link_color}; }}
    img {{
        display: block;
        max-width: 100%;
        height: auto;
        margin: 1.5em 0;
        border-radius: 8px;
    }}
    </style>
</head>
<body>
    <main class="memo">
    {html_content}
    </main>
</body>
</html>"""


@dataclass(frozen=True)
class HtmlTheme:
    """Colors used by the standalone HTML renderer."""

    background: str
    text_color: str
    heading_border: str
    link_color: str


DARK_HTML_THEME = HtmlTheme(
    background="#1e1e1e",
    text_color="#e6e6e6",
    heading_border="#444444",
    link_color="#6cb6ff",
)
LIGHT_HTML_THEME = HtmlTheme(
    background="#ffffff",
    text_color="#37352f",
    heading_border="#e5e5e5",
    link_color="#0969da",
)


@dataclass
class Memo:
    """Generate, render, and export AI-assisted memos.

    Attributes:
        template: Optional Markdown template used as memo structure.
        figures: Mapping from figure placeholders, such as "Figure 1", to image paths.
        datasets: Additional structured or textual context for generation. Datasets are passed
            to the model but are not meant to be referenced in the final memo.
        chunks: Optional retrieved reference chunks used when RAG is enabled.
        content: Generated Markdown content. Populated by :meth:`generate_content`.
        prompt: Instruction prompt sent to Azure OpenAI.
    """

    template: Path | None = None
    figures: dict[str, Path] | None = None
    datasets: dict[str, Any] | None = None
    chunks: list[str] | None = None
    content: str | None = None
    prompt: str = DEFAULT_PROMPT

    @staticmethod
    def _required_env(name: str) -> str:
        """Return a required environment variable or raise a clear error if it is not set.

        Args:
            name: The name of the environment variable to retrieve.

        Returns:
            The value of the environment variable.
        """
        value = os.getenv(name)
        if value is None:
            raise ValueError(f"Environment variable {name} is not set.")
        return value

    def _get_client(self) -> AzureOpenAI:
        """Create an Azure OpenAI client from environment configuration.

        Returns:
            An instance of the AzureOpenAI client.
        """
        load_dotenv()

        return AzureOpenAI(
            api_key=self._required_env("AZURE_OPENAI_API_KEY"),
            azure_endpoint=self._required_env("AZURE_OPENAI_ENDPOINT"),
            api_version=self._required_env("AZURE_OPENAI_API_VERSION"),
        )

    def _get_messages(self) -> list[ChatMessage]:
        """Build the multimodal chat message payload for memo generation.

        Returns:
            A list of chat messages ready to be sent to the model.
        """
        if self.prompt is None:
            raise ValueError("Prompt is not set.")

        content: list[MessagePart] = [{"type": "text", "text": self.prompt}]

        if self.template is not None:
            content.append({"type": "text", "text": "Template:"})
            content.append({"type": "text", "text": self.template.read_text(encoding="utf-8")})

        if self.figures is not None:
            for ref, file_path_fig in self.figures.items():
                img_base64 = base64.b64encode(file_path_fig.read_bytes()).decode("utf-8")
                content.append({"type": "text", "text": f"Figure: {ref}"})
                content.append({"type": "image_url", "image_url": f"data:image/png;base64,{img_base64}"})

        if self.datasets is not None:
            for ref, dataset_content in self.datasets.items():
                content.append({"type": "text", "text": f"Dataset: {ref}"})
                content.append({"type": "text", "text": str(dataset_content)})

        if self.chunks:
            content.append({"type": "text", "text": "Retrieved reference context:"})
            for index, chunk in enumerate(self.chunks, start=1):
                content.append({"type": "text", "text": f"Reference {index}:\n{chunk}"})

        return [{"role": "user", "content": content}]

    def set_chunks(self, index_name: str) -> None:
        """Retrieve reference chunks for the current prompt and template."""
        retriever = get_retriever(
            index_name=index_name,
            embedding_model=self._required_env("AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS"),
        )

        query = self.prompt
        if self.template is not None:
            query += f"\nTemplate:\n{self.template.read_text(encoding='utf-8')}"

        results = retriever.invoke(query, top_k=5)

        self.chunks = [result["chunk"] for result in results]

    def generate_content(self, rag_index_name: str | None = None, max_completion_tokens: int = 2000) -> None:
        """Generate memo Markdown and store it in :attr:`content`.

        Args:
            rag_index_name: The name of the RAG index to use for retrieving reference chunks. Must be provided if rag is True.
        """
        client = self._get_client()
        if rag_index_name is not None:
            self.set_chunks(index_name=rag_index_name)

        response = client.chat.completions.create(
            model=self._required_env("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=self._get_messages(),
            max_completion_tokens=max_completion_tokens,
        )
        self.content = response.choices[0].message.content

    @staticmethod
    def _image_data_uri(file_path: Path) -> str:
        """Return an image file as a base64 data URI."""
        mime_type = mimetypes.guess_type(file_path)[0] or "image/png"
        img_base64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{img_base64}"

    def _get_content_with_figures(self) -> str:
        """Replace figure placeholders with embedded Markdown image links."""
        if self.content is None:
            raise ValueError("Memo content has not been generated.")

        content = self.content
        if self.figures is not None:
            for ref, file_path_fig in self.figures.items():
                image_src = self._image_data_uri(file_path_fig)
                content = content.replace(f"[{ref}]", f"![{ref}]({image_src})")
        return content

    def _get_content_as_html(self, darkmode: bool = True) -> str:
        """Render generated Markdown content as a complete HTML document.

        Args:
            darkmode: When true, render the HTML with a dark background and light text.

        Returns:
            A complete HTML document as a string.
        """
        html_content = markdown.markdown(self._get_content_with_figures(), extensions=["extra"])
        theme = DARK_HTML_THEME if darkmode else LIGHT_HTML_THEME

        return HTML_TEMPLATE.format(html_content=html_content, **theme.__dict__)

    def to_md(self, file_path: Path) -> None:
        """Write the generated memo as Markdown.

        Args:
            file_path: Path to the output Markdown file.
        """
        file_path.write_text(self._get_content_with_figures(), encoding="utf-8")

    def to_html(self, file_path: Path, darkmode: bool = True) -> None:
        """Write the generated memo as a standalone HTML document.

        Args:
            file_path: Path to the output HTML file.
            darkmode: When true, render the HTML with a dark background and light text.
        """
        file_path.write_text(self._get_content_as_html(darkmode=darkmode), encoding="utf-8")

    def show(self, darkmode: bool = True) -> None:
        """Display the generated memo inline in a notebook.

        Args:
            darkmode: When true, render the HTML with a dark background and light text.
        """
        display(HTML(self._get_content_as_html(darkmode=darkmode)))

    def __str__(self) -> str:
        """Return the generated memo as Markdown with figure links."""
        return self._get_content_with_figures()
