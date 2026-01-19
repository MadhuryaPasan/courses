# ================= version 4 state (check state.py) ==============
# chatapp.py
import reflex as rx

from chatapp_tutorial import style

from chatapp_tutorial.state import State


def qa(question: str, answer: str) -> rx.Component:
    return rx.box(
        rx.box(
            rx.text(question, style=style.question_style),
            text_align="right",
        ),
        rx.box(
            rx.text(answer, style=style.answer_style),
            text_align="left",
        ),
        margin_y="1em",
        width="100%",
    )


def chat() -> rx.Component:
    return rx.box(
        rx.foreach(State.chat_history, lambda messages: qa(messages[0], messages[1]))
    )


def action_bar() -> rx.Component:
    return rx.hstack(
        rx.input(
            value=State.question,
            placeholder="Ask a question",
            on_change=State.set_question,
            style=style.input_style,
        ),
        rx.button("Ask", on_click=State.answer, style=style.button_style),
    )


def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            chat(),
            action_bar(),
            align="center",
        )
    )


app = rx.App()
app.add_page(index)


# # ================= version 3 using styling (style.py)==============
# # chatapp.py
# import reflex as rx

# from chatapp_tutorial import style


# def qa(question: str, answer: str) -> rx.Component:
#     return rx.box(
#         rx.box(
#             rx.text(question, style=style.question_style),
#             text_align="right",
#         ),
#         rx.box(
#             rx.text(answer, style=style.answer_style),
#             text_align="left",
#         ),
#         margin_y="1em",
#         width="100%",
#     )


# def chat() -> rx.Component:
#     qa_pairs = [
#         (
#             "What is Reflex?",
#             "A way to build web apps in pure Python!",
#         ),
#         (
#             "What can I make with it?",
#             "Anything from a simple website to a complex web app!",
#         ),
#     ]
#     return rx.box(
#         *[
#             qa(question, answer)
#             for question, answer in qa_pairs
#         ]
#     )


# def action_bar() -> rx.Component:
#     return rx.hstack(
#         rx.input(
#             placeholder="Ask a question",
#             style=style.input_style,
#         ),
#         rx.button("Ask", style=style.button_style),
#     )


# def index() -> rx.Component:
#     return rx.center(
#         rx.vstack(
#             chat(),
#             action_bar(),
#             align="center",
#         )
#     )


# app = rx.App()
# app.add_page(index)


# # ================= version 2 reusable components==============
# import reflex as rx


# # component
# def qa(question: str, answer: str) -> rx.Component:
#     return rx.box(
#         rx.box(question, text_align="right"),
#         rx.box(answer, text_align="left"),
#         margin_y="1em",
#     )


# # component
# def chat() -> rx.Component:
#     """
#     reusing components
#     used components qa()
#     """

#     qa_pairs = [
#         (
#             "What is Reflex?",
#             "A way to build web apps in pure Python!",
#         ),
#         (
#             "What can I make with it?",
#             "Anything from a simple website to a complex web app!",
#         ),
#     ]

#     return rx.box(*[qa(question, answer) for question, answer in qa_pairs])


# def action_bar() -> rx.Component:
#     return rx.hstack(
#         rx.input(placeholder="Ask a question"),
#         rx.button("Ask"),
#     )


# def index() -> rx.Component:
#     return rx.container(chat(), action_bar())


# app = rx.App()
# app.add_page(index)


# ======================= version 1 ==========================

# import reflex as rx

# def index() -> rx.Component:

#     return rx.container(
#         rx.box(
#             "What is Reflex?",
#             text_align = "right",
#             class_name="text-lg text-sky-700"
#         ),
#         rx.box(
#             "A way to build web app in pure Python!",
#             text_align = "left"
#         )

#     )

# app = rx.App()
# app.add_page(index)


# =================== original===================================

# """Welcome to Reflex! This file outlines the steps to create a basic app."""

# import reflex as rx

# from rxconfig import config


# class State(rx.State):
#     """The app state."""


# def index() -> rx.Component:
#     # Welcome Page (Index)
#     return rx.container(
#         rx.color_mode.button(position="top-right"),
#         rx.vstack(
#             rx.heading("Welcome to Reflex!", size="9"),
#             rx.text(
#                 "Get started by editing ",
#                 rx.code(f"{config.app_name}/{config.app_name}.py"),
#                 size="5",
#             ),
#             rx.link(
#                 rx.button("Check out our docs!"),
#                 href="https://reflex.dev/docs/getting-started/introduction/",
#                 is_external=True,
#             ),
#             spacing="5",
#             justify="center",
#             min_height="85vh",
#         ),
#     )


# app = rx.App()
# app.add_page(index)
