import gradio as gr

import ui.app as ui_app


def test_gradio_interface_constructs():
    """Ensure the Gradio interface object can be created without launching."""
    assert isinstance(ui_app.iface, gr.Interface)

