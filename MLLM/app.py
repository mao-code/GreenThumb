import gradio as gr
from mllm import analyse_plant

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 🌿 GreenThumb (Plant Care Assistant \nUpload a plant photo and get tailored care advice.)")
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(label="Plant Photo", type="pil", sources=["upload", "clipboard"])
                notes_in = gr.Textbox(label="Additional notes (optional)", placeholder="e.g., Yellow leaves, 25°C room temperature, watered last Sunday…")
                btn = gr.Button("Analyse & advise", variant="primary")

        with gr.Row():
            result_md = gr.Markdown()

        btn.click(analyse_plant, inputs=[img_in, notes_in], outputs=result_md)
    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, show_api=False, inbrowser=True)
