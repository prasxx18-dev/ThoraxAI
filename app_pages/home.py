"""Landing page: product-focused, one primary action (Analyze an X-ray)."""

import streamlit as st

from src.ui import feature_card, feature_grid, footer, icon, page_setup, render, section_title

page_setup("Home")

# An abstract, label-free illustration of the workspace. It deliberately contains
# no numbers, finding names or scores, so it cannot be mistaken for a result.
PREVIEW_SVG = """
<svg class="tx-preview" viewBox="0 0 560 372" xmlns="http://www.w3.org/2000/svg" role="img"
aria-label="Illustration of the ThoraxAI analysis workspace">
<defs>
<radialGradient id="glow" cx="50%" cy="50%" r="50%">
<stop offset="0%" stop-color="#2DD4BF" stop-opacity=".75"/><stop offset="55%" stop-color="#F59E0B" stop-opacity=".28"/>
<stop offset="100%" stop-color="#F59E0B" stop-opacity="0"/></radialGradient>
<linearGradient id="film" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#1E293B"/><stop offset="1" stop-color="#0B1220"/></linearGradient>
</defs>
<rect width="560" height="372" rx="18" fill="#FFFFFF"/>
<rect x=".5" y=".5" width="559" height="371" rx="17.5" fill="none" stroke="#E2E8F0"/>
<rect x="0" y="0" width="560" height="44" rx="18" fill="#F8FAFC"/><rect x="0" y="26" width="560" height="18" fill="#F8FAFC"/>
<line x1="0" y1="44" x2="560" y2="44" stroke="#E2E8F0"/>
<circle cx="24" cy="22" r="5" fill="#E2E8F0"/><circle cx="40" cy="22" r="5" fill="#E2E8F0"/><circle cx="56" cy="22" r="5" fill="#E2E8F0"/>
<rect x="210" y="16" width="140" height="12" rx="6" fill="#E2E8F0"/>
<rect x="20" y="62" width="250" height="290" rx="12" fill="url(#film)"/>
<g fill="none" stroke="#CBD5E1" stroke-opacity=".55" stroke-width="3" stroke-linecap="round">
<path d="M145 92v70"/>
<path d="M132 104c-38 10-58 52-58 104 0 34 16 52 36 46 26-8 30-40 30-80v-50"/>
<path d="M158 104c38 10 58 52 58 104 0 34-16 52-36 46-26-8-30-40-30-80v-50"/>
</g>
<g fill="none" stroke="#94A3B8" stroke-opacity=".35" stroke-width="2" stroke-linecap="round">
<path d="M88 150c20-10 36-12 48-8"/><path d="M84 180c22-10 40-12 52-8"/><path d="M84 210c22-8 40-10 52-6"/>
<path d="M202 150c-20-10-36-12-48-8"/><path d="M206 180c-22-10-40-12-52-8"/><path d="M206 210c-22-8-40-10-52-6"/>
</g>
<circle cx="190" cy="214" r="58" fill="url(#glow)"/>
<rect x="290" y="62" width="250" height="290" rx="12" fill="#FFFFFF" stroke="#E2E8F0"/>
<rect x="308" y="80" width="96" height="10" rx="5" fill="#0B1F3A" fill-opacity=".85"/>
<rect x="308" y="98" width="150" height="8" rx="4" fill="#E2E8F0"/>
<g>
<rect x="308" y="130" width="70" height="8" rx="4" fill="#CBD5E1"/><rect x="390" y="130" width="132" height="8" rx="4" fill="#F1F5F9"/><rect x="390" y="130" width="104" height="8" rx="4" fill="#0D9488"/>
<rect x="308" y="160" width="58" height="8" rx="4" fill="#CBD5E1"/><rect x="390" y="160" width="132" height="8" rx="4" fill="#F1F5F9"/><rect x="390" y="160" width="82" height="8" rx="4" fill="#0D9488"/>
<rect x="308" y="190" width="76" height="8" rx="4" fill="#CBD5E1"/><rect x="390" y="190" width="132" height="8" rx="4" fill="#F1F5F9"/><rect x="390" y="190" width="50" height="8" rx="4" fill="#CBD5E1"/>
<rect x="308" y="220" width="52" height="8" rx="4" fill="#CBD5E1"/><rect x="390" y="220" width="132" height="8" rx="4" fill="#F1F5F9"/><rect x="390" y="220" width="36" height="8" rx="4" fill="#CBD5E1"/>
<rect x="308" y="250" width="66" height="8" rx="4" fill="#CBD5E1"/><rect x="390" y="250" width="132" height="8" rx="4" fill="#F1F5F9"/><rect x="390" y="250" width="24" height="8" rx="4" fill="#CBD5E1"/>
<rect x="308" y="280" width="60" height="8" rx="4" fill="#CBD5E1"/><rect x="390" y="280" width="132" height="8" rx="4" fill="#F1F5F9"/><rect x="390" y="280" width="16" height="8" rx="4" fill="#CBD5E1"/>
</g>
<g fill="#1E3A5F"><rect x="470" y="126" width="2" height="16" rx="1"/><rect x="452" y="156" width="2" height="16" rx="1"/>
<rect x="462" y="186" width="2" height="16" rx="1"/><rect x="448" y="216" width="2" height="16" rx="1"/>
<rect x="476" y="246" width="2" height="16" rx="1"/><rect x="458" y="276" width="2" height="16" rx="1"/></g>
<rect x="308" y="312" width="214" height="24" rx="8" fill="#F0FDFA" stroke="#CCFBF1"/>
<rect x="320" y="320" width="96" height="8" rx="4" fill="#0F766E" fill-opacity=".55"/>
</svg>
"""

left, right = st.columns([1.05, 1], gap="large", vertical_alignment="center")
with left:
    render(
        '<div class="tx-eyebrow">Chest X-ray image analysis</div>'
        '<div class="tx-hero-title">Intelligent insights<br>from <span>chest X-rays</span></div>'
        '<p class="tx-hero-lead">ThoraxAI scores a frontal chest radiograph for 14 thoracic findings at once and '
        "shows which image regions influenced each score, so every output can be inspected rather than taken on "
        "trust.</p>"
    )
    b1, b2, _ = st.columns([1.15, 1, 0.6])
    if b1.button("Analyze an X-ray", type="primary", icon=":material/arrow_forward:", icon_position="right",
                 width="stretch"):
        st.switch_page("app_pages/analyze.py")
    if b2.button("How it works", width="stretch"):
        st.switch_page("app_pages/about.py")
    render(
        '<div class="tx-hero-meta">'
        f'<span>{icon("lock", 16)}Processed locally in your session</span>'
        f'<span>{icon("eye", 16)}Explainable by design</span>'
        "</div>"
    )
with right:
    render(PREVIEW_SVG + '<div class="tx-preview-caption">Illustration of the analysis workspace</div>')

section_title("One workspace, three ways to look at an image",
              "Built for transparent review of model behaviour on chest radiographs.")
feature_grid([
    feature_card(
        "layers", "Multi-label analysis",
        "Each finding receives its own independent score, so co-occurring findings are surfaced together "
        "instead of being forced into a single label."),
    feature_card(
        "eye", "Visual explainability",
        "Grad-CAM heatmaps highlight the regions that most influenced a selected score, side by side with the "
        "original image."),
    feature_card(
        "chart", "Model evaluation",
        "Performance on held-out data is documented with ROC and precision-recall curves, confusion matrices "
        "and per-finding metrics."),
])

section_title("How it works")
render(
    '<div class="tx-steps">'
    '<div class="tx-step"><span class="n">1</span><div><h4>Upload</h4>'
    "<p>Add a frontal chest X-ray in PNG or JPEG format.</p></div></div>"
    '<div class="tx-step"><span class="n">2</span><div><h4>Analyze</h4>'
    "<p>A DenseNet-based model scores 14 findings in seconds.</p></div></div>"
    '<div class="tx-step"><span class="n">3</span><div><h4>Explain</h4>'
    "<p>Open a heatmap for any finding and download the visualization.</p></div></div>"
    "</div>"
)

footer()
