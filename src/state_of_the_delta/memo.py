import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv
from IPython.display import HTML, display
from openai import AzureOpenAI

from state_of_the_delta.html import image_data_uri, render_memo_html, render_memo_iframe
from state_of_the_delta.rag import get_chunks_from_deltares_kennisbank, get_retriever

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

DEFAULT_PROMPT_URI = "prompts:/sotc.rag.memo/5"


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

    prompt_uri: str = DEFAULT_PROMPT_URI
    template: Path | None = None
    figures: dict[str, Path] | None = None
    datasets: dict[str, Any] | None = None
    chunks: list[str] | None = None
    content: str | None = None

    @staticmethod
    def _get_env_variable(name: str) -> str:
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
            api_key=self._get_env_variable("AZURE_OPENAI_API_KEY"),
            azure_endpoint=self._get_env_variable("AZURE_OPENAI_ENDPOINT"),
            api_version=self._get_env_variable("AZURE_OPENAI_API_VERSION"),
        )

    def _get_prompt(
        self,
    ) -> mlflow.entities.model_registry.prompt_version.PromptVersion:
        """Get the instruction prompt for memo generation.

        Returns:
            The loaded instruction prompt.
        """

        if self.prompt_uri is None:
            raise ValueError("Prompt URI is not set.")

        prompt = mlflow.genai.load_prompt(self.prompt_uri)

        return prompt

    def _get_prompt_content(self) -> list[dict[str, Any]]:
        """Get the structured content for memo generation.

        Returns:
            A list of chat messages ready to be sent to the model.
        """

        content: list[dict[str, Any]] = []

        if self.template is not None:
            content.append({"type": "text", "text": "Template:"})
            content.append({
                "type": "text",
                "text": self.template.read_text(encoding="utf-8"),
            })

        if self.figures is not None:
            for ref, file_path_fig in self.figures.items():
                img_base64 = base64.b64encode(file_path_fig.read_bytes()).decode("utf-8")
                content.append({"type": "text", "text": f"Figure: {ref}"})
                content.append({
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{img_base64}",
                })

        if self.datasets is not None:
            for ref, dataset_content in self.datasets.items():
                content.append({"type": "text", "text": f"Dataset: {ref}"})
                content.append({"type": "text", "text": str(dataset_content)})

        if self.chunks:
            content.append({"type": "text", "text": "Retrieved reference context:"})
            for index, chunk in enumerate(self.chunks, start=1):
                content.append({"type": "text", "text": f"Reference {index}:\n{chunk}"})

        return content

    def _get_prompt_messages(self) -> list[dict[str, Any]]:
        """Get the chat messages for memo generation.

        Returns:
            A list of chat messages ready to be sent to the model.
        """
        prompt = self._get_prompt()

        content = self._get_prompt_content()

        messages = [prompt.template[0], {"role": "user", "content": content}]

        return messages

    def _get_prompt_query(self) -> str:
        """Get the query string for retrieving reference chunks."""
        prompt = self._get_prompt()

        query = [prompt.template[0]["content"]]

        if self.template is not None:
            query.append("Template:")
            query.append(self.template.read_text(encoding="utf-8"))

        query = "\n".join(query)

        return query

    def set_chunks(self, index_name: str) -> None:
        """Retrieve reference chunks for the current prompt and template."""
        retriever = get_retriever(
            index_name=index_name,
            embedding_model=self._get_env_variable("AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS"),
        )

        query = self._get_prompt_query()

        results = retriever.invoke(query, top_k=5)

        self.chunks = [result["chunk"] for result in results]

    def set_chunks_deltares_kennisbank(self, index_name: str) -> None:
        query = self._get_prompt_query()

        self.chunks = get_chunks_from_deltares_kennisbank(query=query, index_name=index_name)

    def generate_content(self, rag_index_name: str | None = None, max_completion_tokens: int = 2000) -> None:
        """Generate memo Markdown and store it in :attr:`content`.

        Args:
            rag_index_name: The name of the RAG index to use for retrieving reference chunks. Must be provided if rag is True.
        """
        client = self._get_client()
        if rag_index_name is not None and rag_index_name != "kennisbank-vector":
            self.set_chunks(index_name=rag_index_name)
        elif rag_index_name == "kennisbank-vector":
            self.set_chunks_deltares_kennisbank(index_name=rag_index_name)

        response = client.chat.completions.create(
            model=self._get_env_variable("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=self._get_prompt_messages(),
            max_completion_tokens=max_completion_tokens,
        )
        self.content = response.choices[0].message.content

    def _get_content_with_figures(self) -> str:
        """Replace figure placeholders with embedded Markdown image links."""
        if self.content is None:
            raise ValueError("Memo content has not been generated.")

        content = self.content
        if self.figures is not None:
            for ref, file_path_fig in self.figures.items():
                image_src = image_data_uri(file_path_fig)
                content = content.replace(f"[{ref}]", f"![{ref}]({image_src})")
        return content

    def _get_content_as_html(self) -> str:
        """Render generated Markdown content as a complete HTML document."""
        return render_memo_html(self._get_content_with_figures())

    def to_md(self, file_path: Path) -> None:
        """Write the generated memo as Markdown.

        Args:
            file_path: Path to the output Markdown file.
        """
        file_path.write_text(self._get_content_with_figures(), encoding="utf-8")

    def to_html(self, file_path: Path) -> None:
        """Write the generated memo as a standalone HTML document.

        Args:
            file_path: Path to the output HTML file.
        """
        file_path.write_text(self._get_content_as_html(), encoding="utf-8")

    def show(self) -> None:
        """Display the generated memo inline in a notebook."""
        display(HTML(render_memo_iframe(self._get_content_with_figures())))

    def __str__(self) -> str:
        """Return the generated memo as Markdown with figure links."""
        return self._get_content_with_figures()
