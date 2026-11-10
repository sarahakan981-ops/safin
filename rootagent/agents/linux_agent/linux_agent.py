"""Linux Terminal Agent — full access to every Linux tool via the terminal."""

from typing import TYPE_CHECKING, List, Optional

from ..base_agent import BaseAgent
from ..prompts import load_prompt

if TYPE_CHECKING:
    from ...knowledge import RAGEngine
    from ...llm import LLM
    from ...runtime import Runtime
    from ...tools import Tool

# Pre-loaded template (module-level, same pattern as pa_agent)
linux_agent = load_prompt("linux_agent")


class LinuxTerminalAgent(BaseAgent):
    """A Linux-first agent with unrestricted terminal access.

    Unlike :class:`RootAgentAgent`, this agent is terminal-centric: it is
    prompted to use *any* binary installed on the Linux system through the
    ``linux_exec`` / ``terminal`` tools, to discover available tools with
    ``list_installed_tools`` / ``which_tool``, and to install missing ones
    with ``linux_install`` — without waiting for a pre-built tool wrapper.
    """

    def __init__(
        self,
        llm: "LLM",
        tools: List["Tool"],
        runtime: "Runtime",
        target: Optional[str] = None,
        scope: Optional[List[str]] = None,
        rag_engine: Optional["RAGEngine"] = None,
        **kwargs,
    ):
        """
        Initialize the LinuxTerminalAgent.

        Args:
            llm: The LLM instance for generating responses
            tools: List of tools available to the agent
            runtime: The runtime environment for tool execution
            target: Primary target (host/IP/URL) if any
            scope: List of in-scope targets/networks
            rag_engine: RAG engine for knowledge retrieval
            **kwargs: Additional arguments passed to BaseAgent
        """
        super().__init__(llm, tools, runtime, **kwargs)
        self.target = target
        self.scope = scope or []
        self.rag_engine = rag_engine
        #: Cache of every executable found in PATH (built on first prompt)
        self._path_tools: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Tool discovery helpers
    # ------------------------------------------------------------------
    def enumerate_path_tools(self, refresh: bool = False) -> List[str]:
        """List every executable binary found in $PATH on this machine.

        Returns:
            Sorted list of executable names (possibly thousands)
        """
        import os

        if self._path_tools is not None and not refresh:
            return self._path_tools

        found = set()
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            if not directory or not os.path.isdir(directory):
                continue
            try:
                for entry in os.scandir(directory):
                    if entry.is_file() and os.access(entry.path, os.X_OK):
                        found.add(entry.name)
            except OSError:
                continue

        self._path_tools = sorted(found)
        return self._path_tools

    def get_system_prompt(self, mode: str = "agent") -> str:  # noqa: ARG002
        """Build the terminal-first system prompt.

        Args:
            mode: unused for this agent (retained for BaseAgent
                compatibility); the terminal-first prompt is always used.
        """
        env = self.runtime.environment

        # Limit the full inventory echoed into the prompt (huge on real
        # systems); the agent can always call `list_installed_tools`.
        path_tools = self.enumerate_path_tools()
        path_tools_preview = path_tools[:300]

        # RAG context from the latest user message (same approach as
        # RootAgentAgent)
        rag_context = ""
        if self.rag_engine and self.conversation_history:
            last_msg = self.conversation_history[-1].content
            if isinstance(last_msg, list):
                last_msg = " ".join(
                    str(part.get("text", ""))
                    for part in last_msg
                    if isinstance(part, dict)
                )
            if last_msg:
                try:
                    relevant = self.rag_engine.search(last_msg)
                    if relevant:
                        rag_context = "\n\n".join(relevant)
                except Exception:  # pragma: no cover - defensive
                    rag_context = ""

        api_tools = [t for t in self.tools if t.enabled]

        return linux_agent.render(
            target=self.target,
            scope=self.scope,
            environment=env,
            path_tools=path_tools_preview,
            rag_context=rag_context,
            tools=api_tools,
            plan=self._task_plan,
        )
