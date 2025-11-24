"""
Gradio UI for EcoRenoAdvisor.
"""

import gradio as gr
from pathlib import Path
from agent.agent import agent


def run_agent(
    query: str,
    category: str,
    budget: float,
    eco: float,
    voc_level: str,
) -> str:
    """
    Run the agent with user inputs.

    Args:
        query: User question
        category: Material category filter
        budget: Maximum price per m2
        eco: Minimum eco score (0.0-1.0)
        voc_level: VOC level filter

    Returns:
        Agent response
    """
    if not query.strip():
        return "Please enter a question."

    voc_mapping = {
        "Any": None,
        "Zero": 0,
        "Low": 1,
        "Medium": 2,
        "High": 3,
    }
    voc_num = voc_mapping.get(voc_level, None)

    filters = {
        "category": category if category.strip() else None,
        "max_price": budget if budget > 0 else None,
        "min_eco": eco if eco > 0 else None,
        "voc": voc_num,
    }
    filters = {k: v for k, v in filters.items() if v is not None}

    try:
        response = agent(query, filters)
        return response
    except Exception as e:
        return f"Error: {str(e)}"


# Theme: red primary button, neutral background
custom_theme = gr.themes.Soft(
    primary_hue="red",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="#8B1A1A",
    button_primary_background_fill_hover="#A52A2A",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#6B1A1A",
    button_secondary_background_fill_hover="#7B2A2A",
    button_secondary_text_color="#FFFFFF",
    background_fill_primary="#F5F5F5",
    background_fill_secondary="#FFFFFF",
    block_background_fill="#FFFFFF",
    input_background_fill="#FFFFFF",
    input_border_color="#1E88E5",
    input_border_color_focus="#1976D2",
    slider_color="#1E88E5",
    checkbox_background_color_selected="#1E88E5",
    checkbox_border_color_selected="#1976D2",
    body_text_color="#333333",
    link_text_color="#8B1A1A",
    link_text_color_hover="#A52A2A",
    border_color_primary="#8B1A1A",
    border_color_accent="#A52A2A",
)

# Custom CSS
custom_css = """
/* Recommendation box styling */
.recommendation-box textarea {
    border-color: #8B1A1A !important;
    background-color: #FFF5F5 !important;
}
.recommendation-box textarea:focus {
    border-color: #A52A2A !important;
    box-shadow: 0 0 0 2px rgba(139, 26, 26, 0.2) !important;
}

/* Keep markdown headers black */
h1, h2, h3 {
    color: #000000 !important;
}

/* ---- BLUE LABELS ON WHITE FOR INPUT FIELDS ---- */
.category label,
.category label span,
.price label,
.price label span,
.eco label,
.eco label span,
.query label,
.query label span {
    color: #1565C0 !important;
    background-color: #FFFFFF !important;
    border-radius: 4px !important;
    padding: 2px 0 !important;
}

/* Manual VOC label (Markdown) */
.voc-label p {
    color: #1565C0 !important;
    margin-bottom: 4px !important;
    font-weight: 500 !important;
}

/* ---- WHITE INPUT BOXES WITH BLUE BORDER ---- */
.category textarea,
.category input,
.price input,
.eco input,
.query textarea {
    background-color: #FFFFFF !important;
    border: 2px solid #1E88E5 !important;
    color: #000000 !important;
}

/* VOC dropdown box */
.voc select {
    background-color: #FFFFFF !important;
    border: 2px solid #1E88E5 !important;
    color: #000000 !important;
}

/* Focus states */
.category textarea:focus,
.category input:focus,
.price input:focus,
.eco input:focus,
.query textarea:focus,
.voc select:focus {
    border-color: #1976D2 !important;
    box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.2) !important;
}
"""

# Get image path
image_path = Path(__file__).parent.parent / "images" / "ssty_01.jpg"
image_path_str = str(image_path) if image_path.exists() else None

# Create Gradio interface
with gr.Blocks(theme=custom_theme, title="EcoRenoAdvisor", css=custom_css) as iface:
    # Header
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            if image_path_str:
                gr.Image(
                    value=image_path_str,
                    show_label=False,
                    show_download_button=False,
                    height=120,
                    width=200,
                    container=True,
                    elem_classes="header-image",
                )
        with gr.Column(scale=4):
            gr.Markdown(
                """
                # EcoRenoAdvisor
                ### Personalized, Eco-Friendly Renovation Advice

                Get AI-powered recommendations based on sustainable materials and expert documents.
                """,
                elem_classes="header-text",
            )

    gr.Markdown("---")

    # Main content
    with gr.Row():
        # Filters column
        with gr.Column(scale=1):
            gr.Markdown("### Filters")
            category_input = gr.Textbox(
                label="Category (optional)",
                placeholder="e.g., flooring, insulation, paint",
                lines=1,
                elem_classes="category",
            )
            budget_input = gr.Slider(
                minimum=0,
                maximum=200,
                value=100,
                step=5,
                label="Max Price per m² ($)",
                elem_classes="price",
            )
            eco_input = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.1,
                label="Min Eco Score",
                elem_classes="eco",
            )
            # Manual label + dropdown, to avoid the red pill style
            gr.Markdown("**Max VOC Level**", elem_classes="voc-label")
            voc_input = gr.Dropdown(
                choices=["Any", "Zero", "Low", "Medium", "High"],
                value="Any",
                label="",           # no auto label
                show_label=False,
                elem_classes="voc",
            )

        # Question and answer column
        with gr.Column(scale=2):
            gr.Markdown("### Ask Your Question")
            query_input = gr.Textbox(
                label="Your Question",
                placeholder=(
                    "e.g., For a 12 m² kids room, find two low VOC flooring options, "
                    "mid price, eco_score above 0.7. Compare pros and cons."
                ),
                lines=4,
                elem_classes="query",
            )
            submit_btn = gr.Button("Get Recommendation", variant="primary", size="lg")

            gr.Markdown("### Recommendation")
            output = gr.Textbox(
                label="",
                lines=12,
                show_label=False,
                elem_classes="recommendation-box",
            )

    # Examples
    gr.Markdown("---")
    gr.Markdown("### Example Questions")
    gr.Examples(
        examples=[
            [
                "Find low VOC flooring options for a kids room, mid price range, eco score above 0.7",
                "flooring",
                80,
                0.7,
                "Low",
            ],
            [
                "Compare insulation materials for bedrooms",
                "insulation",
                150,
                0.5,
                "Any",
            ],
            [
                "What are the most sustainable paint options within my budget?",
                "paint",
                50,
                0.8,
                "Low",
            ],
        ],
        inputs=[query_input, category_input, budget_input, eco_input, voc_input],
        label="Click an example to try it:",
    )

    # Button and enter key
    submit_btn.click(
        fn=run_agent,
        inputs=[query_input, category_input, budget_input, eco_input, voc_input],
        outputs=output,
    )
    query_input.submit(
        fn=run_agent,
        inputs=[query_input, category_input, budget_input, eco_input, voc_input],
        outputs=output,
    )


if __name__ == "__main__":
    import os
    import socket

    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

    port = 7860
    for p in range(7860, 7870):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", p))
        sock.close()
        if result != 0:
            port = p
            if p != 7860:
                print(f"Port 7860 in use, using port {port} instead")
            break
    else:
        print("Warning: Could not find available port in range 7860-7869, trying 7860 anyway")
        port = 7860

    iface.launch(server_name="0.0.0.0", server_port=port, share=False, show_error=False)
