import reflex as rx

config = rx.Config(
    app_name="chatapp_tutorial",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)