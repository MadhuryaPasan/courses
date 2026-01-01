# Updated LangChain Course Notes

This notebook is a refreshed version of the original `note.ipynb`, updated to reflect the latest standards and best practices in LangChain as of 2024. We will go through several common use cases, demonstrating the power and flexibility of the LangChain Expression Language (LCEL).

## 1. Environment Setup

First, let's install the necessary libraries. This notebook assumes you are running a local LLM with an OpenAI-compatible API, such as Ollama.

```python
%pip install -qU langchain langchain-openai langchain-core langchain-community langsmith scikit-image matplotlib tqdm
```

### 1.1 Imports and LLM Configuration

```python
import os
from getpass import getpass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from pydantic import BaseModel, Field

from IPython.display import display, Markdown

import textwrap

# For local LLM through Ollama (or other service with OpenAI-compatible API)
# Set a dummy API key as it's required by the library, but not used for local models.
os.environ["OPENAI_API_KEY"] = "none"

MODEL = "qwen2:1.5b"
BASE_URL = "http://localhost:11434/v1"

# We will define two LLM instances:
# - llm: for tasks requiring factual, consistent output.
# - creative_llm: for tasks benefiting from more diverse, imaginative output.
llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, temperature=0.0)
creative_llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, temperature=0.9)
```

### 1.2 Article Content

This is the draft article we will be working with throughout the notebook.

```python
article = """
We believe AI's short—to mid-term future belongs to agents and that the long-term future of *AGI* may evolve from agentic systems. Our definition of agents covers any neuro-symbolic system in which we merge neural AI (such as an LLM) with semi-traditional software.

With agents, we allow LLMs to integrate with code — allowing AI to search the web, perform math, and essentially integrate into anything we can build with code. It should be clear the scope of use cases is phenomenal where AI can integrate with the broader world of software.

In this introduction to AI agents, we will cover the essential concepts that make them what they are and why that will make them the core of real-world AI in the years to come.

---\n
## Neuro-Symbolic Systems

Neuro-symbolic systems consist of both neural and symbolic computation, where:

- Neural refers to LLMs, embedding models, or other neural network-based models.
- Symbolic refers to logic containing symbolic logic, such as code.

Both neural and symbolic AI originate from the early philosophical approaches to AI: connectionism (now neural) and symbolism. Symbolic AI is the more traditional AI. Diehard symbolists believed they could achieve true AGI via written rules, ontologies, and other logical functions.

The other camp were the connectionists. Connectionism emerged in 1943 with a theoretical neural circuit but truly kicked off with Rosenblatt's perceptron paper in 1958 [1][2]. Both of these approaches to AI are fascinating but deserve more time than we can give them here, so we will leave further exploration of these concepts for a future chapter.

Most important to us is understanding where symbolic logic outperforms neural-based compute and vice-versa.

| Neural | Symbolic |
| --- | --- |
| Flexible, learned logic that can cover a huge range of potential scenarios. | Mostly hand-written rules which can be very granular and fine-tuned but hard to scale. |
| Hard to interpret why a neural system does what it does. Very difficult or even impossible to predict behavior. | Rules are written and can be understood. When unsure why a particular ouput was produced we can look at the rules / logic to understand. |
| Requires huge amount of data and compute to train state-of-the-art neural models, making it hard to add new abilities or update with new information. | Code is relatively cheap to write, it can be updated with new features easily, and latest information can often be added often instantaneously. |
| When trained on broad datasets can often lack performance when exposed to unique scenarios that are not well represented in the training data. | Easily customized to unique scenarios. |
| Struggles with complex computations such as mathematical operations. | Perform complex computations very quickly and accurately. |

Pure neural architectures struggle with many seemingly simple tasks. For example, an LLM *cannot* provide an accurate answer if we ask it for today's date.

Retrieval Augmented Generation (RAG) is commonly used to provide LLMs with up-to-date knowledge on a particular subject or access to proprietary knowledge.

### Giving LLMs Superpowers

By 2020, it was becoming clear that neural AI systems could not perform tasks symbolic systems typically excelled in, such as arithmetic, accessing structured DB data, or making API calls. These tasks require discrete input parameters that allow us to process them reliably according to strict written logic.

In 2022, researchers at AI21 developed Jurassic-X, an LLM-based "neuro-symbolic architecture." Neuro-symbolic refers to merging the "neural computation" of large language models (LLMs) with more traditional (i.e. symbolic) computation of code.

Jurassic-X used the Modular Reasoning, Knowledge, and Language (MRKL) system [3]. The researchers developed MRKL to solve the limitations of LLMs, namely:

- Lack of up-to-date knowledge, whether that is the latest in AI or something as simple as today's date.
- Lack of proprietary knowledge, such as internal company docs or your calendar bookings.
- Lack of reasoning, i.e. the inability to perform operations that traditional software is good at, like running complex mathematical operations.
- Lack of ability to generalize. Back in 2022, most LLMs had to be fine-tuned to perform well in a specific domain. This problem is still present today but far less prominent as the SotA models generalize much better and, in the case of MRKL, are able to use tools relatively well (although we could certainly take the MRKL solution to improve tool use performance even today).

MRKL represents one of the earliest forms of what we would now call an agent; it is an LLM (neural computation) paired with executable code (symbolic computation).

## ReAct and Tools

There is a misconception in the broader industry that an AI agent is an LLM contained within some looping logic that can generate inputs for and execute code functions. This definition of agents originates from the huge popularity of the ReAct agent framework and the adoption of a similar structure with function/tool calling by LLM providers such as OpenAI, Anthropic, and Ollama.

Our "neuro-symbolic" definition is much broader but certainly does include ReAct agents and LLMs paired with tools. This agent type is the most common for now, so it\'s worth understanding the basic concept behind it.

The **Re**ason **Act**ion (ReAct) method encourages LLMs to generate iterative *reasoning* and *action* steps. During *reasoning,* the LLM describes what steps are to be taken to answer the user\'s query. Then, the LLM generates an *action,* which we parse into an input to some executable code, which we typically describe as a tool/function call.

Following the reason and action steps, our action tool call returns an observation. The logic returns the observation to the LLM, which is then used to generate subsequent reasoning and action steps.

The ReAct loop continues until the LLM has enough information to answer the original input. Once the LLM reaches this state, it calls a special *answer* action with the generated answer for the user.

## Not only LLMs and Tool Calls

LLMs paired with tool calling are powerful but far from the only approach to building agents. Using the definition of neuro-symbolic, we cover architectures such as:

- Multi-agent workflows that involve multiple LLM-tool (or other agent structure) combinations.
- More deterministic workflows where we may have set neural model-tool paths that may fork or merge as the use case requires.
- Embedding models that can detect user intents and decide tool-use or LLM selection-based selection in vector space.

These are just a few high-level examples of alternative agent structures. Far from being designed for niche use cases, we find these alternative options to frequently perform better than the more common ReAct or Tool agents. We will cover all of these examples and more in future chapters.

---

Agents are fundamental to the future of AI, but that doesn\'t mean we should expect that future to come from agents in their most popular form today. ReAct and Tool agents are great and handle many simple use cases well, but the scope of agents is much broader, and we believe thinking beyond ReAct and Tools is key to building future AI.

---

You can sign up for the [Aurelio AI newsletter](https://b0fcw9ec53w.typeform.com/to/w2BDHVK7) to stay updated on future releases in our comprehensive course on agents.

---

## References

[1] The curious case of Connectionism (2019) [https://www.degruyter.com/document/doi/10.1515/opphil-2019-0018/html](https://www.degruyter.com/document/doi/10.1515/opphil-2019-0018/html)

[2] F. Rosenblatt, [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf) (1958), Psychological Review

[3] E. Karpas et al. [MRKL Systems: A Modular, Neuro-Symbolic Architecture That Combines Large Language Models, External Knowledge Sources and Discrete Reasoning](https://arxiv.org/abs/2205.00445) (2022), AI21 Labs
"""
```

## 2. Article Content Generation with LCEL

We'll use LangChain Expression Language (LCEL) to create chains that generate a title and description for our article.

### 2.1 Generating an Article Title

```python
prompt_one = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant called {name} that helps generate article titles."),
        (
            "human",
            """You are tasked with creating a name for an article.\nThe article is here for you to examine:\n---\n{article}\n---\nOnly output the article name, no other explanation or text can be provided and it needs to be simple and short."""
        )
    ]
)

chain_one = (
    prompt_one
    | creative_llm
    | StrOutputParser()
)

article_title = chain_one.invoke({"article": article, "name": "Title Finder"})

print(f"Generated Title:\n\n{article_title}")
```

### 2.2 Generating an Article Description

Now let's create a short, SEO-friendly description for the article.

```python
prompt_two = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant that helps build good articles."),
        (
            "human",
            """You are tasked with creating a description for the article. The article is here for you to examine:\n---\n{article}\n---\nHere is the article title: '{article_title}'.\nOutput an SEO-friendly article description. Make sure it does not exceed 120 characters.\nDo not output anything other than the description."""
        )
    ]
)

chain_two = prompt_two | llm | StrOutputParser()

article_description = chain_two.invoke({"article": article, "article_title": article_title})

print(f"Generated Description:\n\n{article_description}")
```

### 2.3 Providing Editorial Advice with Structured Output

A powerful feature of LangChain is the ability to request structured output (like JSON) from the LLM. We can define a Pydantic model to specify the exact format we need.

Here, we'll ask the LLM to choose a paragraph from our article, suggest an improved version, and provide constructive feedback.

```python
from pydantic import BaseModel, Field

class Paragraph(BaseModel):
    """A Pydantic model to structure the LLM's editorial feedback."""
    original_paragraph: str = Field(description="The original paragraph from the article.")
    edited_paragraph: str = Field(description="The improved, edited version of the paragraph.")
    feedback: str = Field(description="Constructive feedback on why the edits were made.")

prompt_three = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert editor providing feedback on an article."),
        (
            "human",
            """Please review the following article:\n---\n{article}\n---Choose one paragraph to review and edit. Provide the original paragraph, your edited version, and constructive feedback explaining your changes."""
        )
    ]
)

structured_llm = creative_llm.with_structured_output(Paragraph)

chain_three = prompt_three | structured_llm

editorial_feedback = chain_three.invoke({"article": article})

print("--- Original Paragraph ---")
print(textwrap.fill(editorial_feedback.original_paragraph, width=100))
print("\n--- Edited Paragraph ---")
print(textwrap.fill(editorial_feedback.edited_paragraph, width=100))
print("\n--- Feedback ---")
print(textwrap.fill(editorial_feedback.feedback, width=100))
```

### 2.4 Image Generation

We can also use LangChain to generate a prompt for an image generator (like DALL-E 3) and then call the API to create a thumbnail for our article.

```python
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate

# 1. Create a prompt to generate the image description
image_prompt_template = PromptTemplate(
    input_variables=["article"],
    template=(
        "Generate a prompt with less than 500 characters to generate an image "
        "based on the following article: {article}"
    ),
)

# 2. Define a function to call the image generator
# Note: This requires a valid OpenAI API key with DALL-E access enabled
def generate_image(image_prompt_str):
    return DallEAPIWrapper(model="dall-e-3").run(image_prompt_str)

# 3. Create the chain
image_chain = (
    {"article": RunnablePassthrough()}
    | image_prompt_template
    | creative_llm
    | StrOutputParser()
    | RunnableLambda(generate_image)
)

# Uncomment the lines below to run (this will incur costs on your OpenAI account)
# image_url = image_chain.invoke(article)
# print(f"Generated Image URL: {image_url}")
```

## 3. LangSmith Tracing

LangSmith is an invaluable tool for debugging, monitoring, and evaluating your LLM applications. Let's set it up to trace the execution of our chains.

```python
from langsmith import traceable
import random
import time
from tqdm.auto import tqdm

# Set up LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
try:
    os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGCHAIN_API_KEY"] or getpass("Enter LangSmith API Key: ")
except (TypeError, KeyError):
     os.environ["LANGCHAIN_API_KEY"] = getpass("Enter LangSmith API Key: ")
os.environ["LANGCHAIN_PROJECT"] = "Updated LangChain Course"

print("LangSmith is set up. Subsequent runs will be traced.")
```

### 3.1 Tracing a Chain

Now that LangSmith is configured, let's re-run our title generation chain. You can then view the trace in your LangSmith project.

```python
traced_title = chain_one.invoke({"article": article, "name": "Traced Title Finder"})
print(f"Generated Title (Traced):\n\n{traced_title}")
```

### 3.2 Tracing Standard Python Functions

The `@traceable` decorator allows you to trace any Python function, not just LangChain objects.

```python
@traceable(name="Random Number Generator")
def generate_random_number():
    return random.randint(0, 100)

@traceable
def generate_string_with_delay(input_str: str):
    number = random.randint(1, 3)
    time.sleep(number)
    return f"{input_str} (delayed by {number}s)"

print("Running traceable functions...")
for i in tqdm(range(3)):
    generate_random_number()
    generate_string_with_delay("Hello LangSmith!")
print("Finished. Check your LangSmith project for the traces.")
```

## 4. Advanced Prompting Techniques

### 4.1 RAG-style Prompting

This demonstrates a basic Retrieval-Augmented Generation (RAG) setup where we provide external context to the LLM to answer a question.

```python
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the user\'s query based on the context below. If you cannot answer, say 'I don\'t know'.\n\nContext: {context}"),
        ("user", "{query}"),
    ]
)

rag_chain = rag_prompt | llm | StrOutputParser()

rag_context = """Aurelio AI is an AI company developing tooling for AI engineers. Their focus is on language AI with a team that has strong expertise in building AI agents and a background in information retrieval. They are the company behind Semantic Router and Semantic Chunkers."""

rag_query = "What does Aurelio AI do?"

response = rag_chain.invoke({"query": rag_query, "context": rag_context})
print(response)
```

### 4.2 Few-Shot Prompting

Few-shot prompting guides the model's behavior by providing it with examples of desired input-output pairs. This is highly effective for controlling the output format.

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {
        "input": "Can you explain gravity?",
        "output": """
## Gravity

Gravity is a fundamental force of the universe.

### Key Points

* Discovered by Sir Isaac Newton.
* Described as the curvature of spacetime in General Relativity.
* The hypothetical particle for gravity is the graviton."""
    },
    {
        "input": "What is the capital of France?",
        "output": """
## France

The capital of France is Paris.

### Key Points

* Origins of the name Paris are from the Celtic Parisii tribe.
* It is a major global center for art, fashion, and culture."""
    },
]

example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])

few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's query based on the context. Always answer in markdown format with headers and bullet points."),
    few_shot_prompt,
    ("user", "{query}\n\nContext: {context}"),
])

few_shot_chain = final_prompt | llm | StrOutputParser()

response = few_shot_chain.invoke({"query": rag_query, "context": rag_context})

print(response)
```

### 4.3 Streaming

LangChain supports streaming, which allows you to receive the LLM's response token by token. This is crucial for creating responsive, real-time user experiences.

```python
print("Streaming response:")
full_response = ""
for chunk in few_shot_chain.stream({"query": rag_query, "context": rag_context}):
    print(chunk, end="", flush=True)
    full_response += chunk
```

### 4.4 Chain-of-Thought (CoT) Prompting

Chain-of-Thought prompting encourages the model to 'think step by step', which can significantly improve its performance on reasoning tasks.

```python
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "Be a helpful assistant and answer the user's question.\n\nTo answer the question, you must:\n- List systematically and in precise detail all sub problems that need to be solved.\n- Solve each sub problem individually and in sequence.\n- Finally, use everything you have worked through to provide the final answer."),
    ("user", "{query}")
])

cot_chain = cot_prompt | llm | StrOutputParser()
cot_query = "How many keystrokes are needed to type the numbers from 1 to 500?"

response = cot_chain.invoke({"query": cot_query})
print(response)
```

## 5. Chat Memory

### 5.1 Legacy Method (ConversationChain)

*Note: This method is deprecated in favor of `RunnableWithMessageHistory` but is included here for reference.*

The older approach relied on `ConversationBufferMemory` to store the conversation history and `ConversationChain` to manage the interaction loop.

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 1. Initialize Memory
memory = ConversationBufferMemory(return_messages=True)

# 2. Add some initial context manually (simulating a conversation)
memory.save_context(
    {"input": "Hi, my name is James"},
    {"output": "Hey James, what's up? I'm an AI model called Zeta."}
)
memory.save_context(
    {"input": "I'm researching the different types of conversational memory."},
    {"output": "That's interesting, what are some examples?"}
)

# 3. Initialize the Chain
chain = ConversationChain(llm=llm, memory=memory, verbose=True)

# 4. Run the chain with a new user input
response = chain.invoke(input="What is my name again?")
print(f"Response: {response['response']}")
```

### 5.2 Modern Method (RunnableWithMessageHistory)

To build a chatbot, you need to manage the conversation history. The modern, recommended approach is to use `RunnableWithMessageHistory`.

```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant called Zeta."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{query}"),
])

memory_chain = memory_prompt | llm | StrOutputParser()

chat_history_store = {}

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_history_store:
        chat_history_store[session_id] = InMemoryChatMessageHistory()
    return chat_history_store[session_id]

chain_with_history = RunnableWithMessageHistory(
    memory_chain,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
)

print("Chatbot initialized. Use a unique session_id for each conversation.")

session_id = "my_chat_session_01"

# First message
response_1 = chain_with_history.invoke(
    {"query": "Hi, my name is James"}, 
    config={"configurable": {"session_id": session_id}}
)
print(f"Zeta: {response_1}")

# Second message
response_2 = chain_with_history.invoke(
    {"query": "What is my name?"}, 
    config={"configurable": {"session_id": session_id}}
)
print(f"Zeta: {response_2}")
```

### 5.3 Specialized Memory Types (Modern Implementation)

Modern LangChain allows you to implement specialized memory logic by creating custom `BaseChatMessageHistory` classes.

#### 5.3.1 ConversationBufferWindowMemory (Last k Messages)

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

class BufferWindowMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    k: int = Field(default=4)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)
        self.messages = self.messages[-self.k :]

    def clear(self) -> None:
        self.messages = []

# Usage: Update get_chat_history to return BufferWindowMessageHistory(k=4)
```

#### 5.3.2 ConversationSummaryMemory (Summarized History)

```python
from langchain_core.messages import SystemMessage

class ConversationSummaryMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    llm: ChatOpenAI

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a new summary of the conversation given the existing summary and new messages."),
            ("human", "Existing summary:\n{existing_summary}\n\nNew messages:\n{new_messages}")
        ])
        new_summary = self.llm.invoke(summary_prompt.format_messages(
            existing_summary=self.messages,
            new_messages=[m.content for m in messages]
        ))
        self.messages = [SystemMessage(content=new_summary.content)]

    def clear(self) -> None:
        self.messages = []
```

#### 5.3.3 ConversationSummaryBufferMemory (Hybrid)

```python
class ConversationSummaryBufferMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    llm: ChatOpenAI
    k: int = Field(default=4)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        # 1. Pop existing summary if present
        existing_summary = self.messages.pop(0) if self.messages and isinstance(self.messages[0], SystemMessage) else None
        
        # 2. Add new messages
        self.messages.extend(messages)
        
        # 3. If over limit, summarize the oldest messages
        if len(self.messages) > self.k:
            old_messages = self.messages[:-self.k]
            self.messages = self.messages[-self.k:]
            
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "Summarize the following old messages and merge with existing summary."),
                ("human", "Existing Summary:\n{existing_summary}\n\nOld Messages:\n{old_messages}")
            ])
            new_summary = self.llm.invoke(summary_prompt.format_messages(
                existing_summary=existing_summary.content if existing_summary else "None",
                old_messages=[m.content for m in old_messages]
            ))
            self.messages.insert(0, SystemMessage(content=new_summary.content))

    def clear(self) -> None:
        self.messages = []
```

## 6. Agents and Tools

Agents use an LLM to decide which actions to take. Actions are defined by **Tools**.

### 6.1 Defining Custom Tools

```python
from langchain_core.tools import tool

@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y

@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' and 'y'."""
    return x * y

@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the power of 'y'."""
    return x**y

tools = [add, multiply, exponentiate]
```

### 6.2 Creating a Tool-Calling Agent

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 1. Define Prompt with Scratchpad
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful mathematical assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 2. Construct Agent
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# 3. Create Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. Invoke Agent
# response = agent_executor.invoke({"input": "What is 10.7 multiplied by 7.68?"})
```

## 7. Conclusion

This document has covered the modern fundamentals of working with LangChain, including:
- Setting up your environment for local and external services.
- Building chains using the LangChain Expression Language (LCEL).
- Enforcing structured output with Pydantic.
- Tracing and debugging with LangSmith.
- Advanced prompting techniques like Few-Shot and Chain-of-Thought.
- Modern implementation of various memory strategies (Buffer, Window, Summary, Hybrid).
- Creating custom tools and tool-calling agents.

```