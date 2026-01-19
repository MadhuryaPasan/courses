# =================LLM=================
import reflex as rx
import asyncio
from openai import AsyncOpenAI



class State(rx.State):
    # user question
    question: str

    # chat history (question, answer)
    chat_history: list[tuple[str, str]]

    @rx.event
    async def answer(self):
        # Our chatbot has some brains now!
        client = AsyncOpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1"
        )

        session = await client.chat.completions.create(
            model="qwen3:1.7b-q8_0",
            messages=[
                {"role": "user", "content": self.question}
            ],
            stop=None,
            temperature=0.7,
            stream=True,
        )

        # Add to the answer as the chatbot responds.
        answer = ""
        self.chat_history.append((self.question, answer))

        # Clear the question input.
        self.question = ""
        # Yield here to clear the frontend input before continuing.
        yield

        async for item in session:
            if hasattr(item.choices[0].delta, "content"):
                if item.choices[0].delta.content is None:
                    # presence of 'None' indicates the end of the response
                    break
                answer += item.choices[0].delta.content
                self.chat_history[-1] = (
                    self.chat_history[-1][0],
                    answer,
                )
                yield









# # =================streaming=================
# import reflex as rx
# import asyncio



# class State(rx.State):
#     # user question
#     question: str

#     # chat history (question, answer)
#     chat_history: list[tuple[str, str]]

#     @rx.event
#     async def answer(self):
#         answer = "I don't know!"
#         self.chat_history.append((self.question, ""))
#         # clear the question input.
#         self.question = ""

#         # Yield here to clear the frontend input before continuing.
#         yield

#         for i in range(len(answer)):
#             # Pause to show the streaming effect.
#             await asyncio.sleep(0.1)

#             # add one letter at a time to the output
#             self.chat_history[-1] = (self.chat_history[-1][0], answer[: i + 1])

#             yield


# =====================original===================
# import reflex as rx

# class State(rx.State):
#     # user question
#     question: str

#     # chat history (question, answer)
#     chat_history: list[tuple[str,str]]

#     @rx.event
#     def answer(self):
#         answer = "I don't know!"
#         self.chat_history.append((self.question, answer))
#         self.question = ""
