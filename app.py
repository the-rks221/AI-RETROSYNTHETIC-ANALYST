import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw
import os
import json
import re
import sqlite3
import hashlib
from PIL import Image

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & SECURE SETUP
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Retrosynthetic Analyst",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {
        color: #00FFCC;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. SQLITE DATABASE SETUP FOR USER AUTHENTICATION
# ------------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    if data and data[0] == make_hash(password):
        return True
    return False

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users(username, password) VALUES (?, ?)', (username, make_hash(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

init_db()

# Session State Initialization for Login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

# ------------------------------------------------------------------------------
# 3. AUTHENTICATION SCREEN (LOGIN / SIGN UP)
# ------------------------------------------------------------------------------
if not st.session_state['logged_in']:
    st.markdown("<h1 class='main-title'>🔐 AI Retrosynthetic Analyst Portal</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])
        
        with tab_login:
            st.subheader("Welcome Back!")
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                if check_user(login_user, login_pass):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = login_user
                    st.success("Login successful! Loading dashboard...")
                    st.rerun()
                else:
                
                    st.error("Invalid username or password.")
                    
        with tab_signup:
            st.subheader("Create a New Account")
            new_user = st.text_input("Choose Username", key="signup_user")
            new_pass = st.text_input("Choose Password", type="password", key="signup_pass")
            if st.button("Register"):
                if new_user.strip() and new_pass.strip():
                    if add_user(new_user, new_pass):
                        st.success("Account created successfully! Please switch to the Login tab.")
                    else:
                        st.error("Username already exists. Choose another.")
                else:
                    st.warning("Please fill in all fields.")
    st.stop()  # Stop execution here until user logs in

# ------------------------------------------------------------------------------
# 4. MAIN APPLICATION (AUTHENTICATED USERS ONLY)
# ------------------------------------------------------------------------------
st.markdown("<h1 class='main-title'>🧪 AI Retrosynthetic Analyst Web Dashboard</h1>", unsafe_allow_html=True)

# Sidebar with User Info & Logout
with st.sidebar:
    st.write(f"👤 Logged in as: **{st.session_state['username']}**")
    if st.button("🚪 Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""
        st.rerun()
        
    st.markdown("---")
    st.header("⚙️ Configuration")
    selected_model = st.selectbox("Select Gemini Model", ["gemini-3.8-flash", "Gemini 3.8 Live Extended Thinking", "Antigravity"])
    st.markdown("---")
    st.markdown("Designed for Pharmaceutical Chemists & Computational Retrosynthesis.")

# Retrieve Shared API Key Securely from Streamlit Secrets or Environment Variables
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

# Helper Functions
def generate_disconnection_gif(target_smiles, synthon_smiles, output_gif_path):
    try:
        target_mol = Chem.MolFromSmiles(target_smiles)
        synthon_mol = Chem.MolFromSmiles(synthon_smiles)
        if not target_mol or not synthon_mol:
            return None
            
        frames = []
        img_target = Draw.MolToImage(target_mol, size=(350, 350))
        for _ in range(4): frames.append(img_target)
        
        img_highlight = Draw.MolToImage(target_mol, size=(350, 350), highlightAtoms=range(target_mol.GetNumAtoms()), highlightColor=(1, 0.4, 0.4))
        for _ in range(2): frames.append(img_highlight)
        
        img_synthons = Draw.MolToImage(synthon_mol, size=(350, 350))
        for _ in range(5): frames.append(img_synthons)
        
        frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], duration=500, loop=0)
        return output_gif_path
    except Exception as e:
        st.error(f"Error rendering animation: {e}")
        return None

def extract_text_and_json(raw_response):
    if isinstance(raw_response, list):
        parts = [block['text'] if isinstance(block, dict) and 'text' in block else str(block) for block in raw_response]
        text_result = "\n".join(parts)
    else:
        text_result = str(raw_response)
        
    json_str = "[]"
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text_result)
    if json_match:
        json_str = json_match.group(1).strip()
        text_result = text_result.replace(json_match.group(0), "").strip()

    return text_result, json_str

def get_multi_pathway_retrosynthesis(molecule_name, smiles, model_name):
    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("Server API Key is not configured in Streamlit Secrets.")
        
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)

    template = """
    You are a world-class synthetic organic chemist. Analyze the target molecule:
    Name: {molecule}
    SMILES: {smiles}

    Provide a thorough report covering THREE distinct retrosynthetic routes:
    - **Pathway A**: Classical/Industrial Route (highest commercial feasibility)
    - **Pathway B**: Alternative Synthon/Green Chemistry Route (mild conditions/eco-friendly)
    - **Pathway C**: Convergent or Novel Functional Group Interconversion (FGI) Route

    Structure your report strictly as follows:

    ## 1. Pathway A: Classical/Industrial Route
    - **Disconnection Strategy**: Explain key bond cleavage.
    - **Forward Steps**: Step-by-step reagents, solvents, temperature, and expected yields.

    ## 2. Pathway B: Alternative Green Chemistry Route
    - **Disconnection Strategy**: Explain key bond cleavage.
    - **Forward Steps**: Step-by-step reagents, solvents, temperature, and expected yields.

    ## 3. Pathway C: Convergent/Novel FGI Route
    - **Disconnection Strategy**: Explain key bond cleavage.
    - **Forward Steps**: Step-by-step reagents, solvents, temperature, and expected yields.

    ## 4. Comprehensive Feasibility Comparison Table
    Create a Markdown table comparing Pathway A, B, and C across these metrics:
    | Metric | Pathway A | Pathway B | Pathway C |
    |---|---|---|---|
    | Number of Synthetic Steps | ... | ... | ... |
    | Commercial Availability of Starting Materials | ... | ... | ... |
    | Atom Economy / Green Metrics | ... | ... | ... |
    | Total Estimated Yield | ... | ... | ... |
    | Overall Feasibility Score (Out of 10) | ... | ... | ... |
    | Best Suited Application | ... | ... | ... |

    CRITICAL: At the VERY END of your response, output a single JSON block containing valid SMILES for each disconnection step across all 3 pathways formatted exactly like this:
    ```json
    [
      {{
        "pathway_name": "Pathway A: Industrial Route",
        "steps": [
          {{
            "step": 1,
            "title": "Disconnection title",
            "target_smiles": "SMILES_HERE",
            "synthon_smiles": "SYNTHON_SMILES_HERE"
          }}
        ]
      }},
      {{
        "pathway_name": "Pathway B: Alternative Route",
        "steps": [
          {{
            "step": 1,
            "title": "Disconnection title",
            "target_smiles": "SMILES_HERE",
            "synthon_smiles": "SYNTHON_SMILES_HERE"
          }}
        ]
      }},
      {{
        "pathway_name": "Pathway C: Novel Route",
        "steps": [
          {{
            "step": 1,
            "title": "Disconnection title",
            "target_smiles": "SMILES_HERE",
            "synthon_smiles": "SYNTHON_SMILES_HERE"
          }}
        ]
      }}
    ]
    ```
    """

    prompt = PromptTemplate(input_variables=["molecule", "smiles"], template=template)
    chain = prompt | llm | StrOutputParser()
    raw_result = chain.invoke({"molecule": molecule_name, "smiles": smiles})
    return extract_text_and_json(raw_result)

# Main UI Execution
col1, col2 = st.columns([3, 1])
with col1:
    target_input = st.text_input("Enter Target Organic Molecule:", value="Paracetamol", placeholder="e.g., Ibuprofen, Aspirin, Caffeine")
with col2:
    st.write(" ")
    st.write(" ")
    analyze_btn = st.button("🚀 Analyze 3 Pathways")

if analyze_btn:
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("⚠️ Server API Key is missing. Please configure it in Streamlit Cloud Secrets.")
    else:
        with st.spinner(f"🔍 Searching PubChem and generating structural models for '{target_input}'..."):
            try:
                compounds = pcp.get_compounds(target_input, 'name')
                if not compounds:
                    st.error("❌ Compound not found in PubChem database.")
                else:
                    target = compounds[0]
                    smiles = target.smiles if hasattr(target, 'smiles') else target.isomeric_smiles
                    
                    st.subheader(f"Molecule Identified: {target.iupac_name}")
                    st.code(f"SMILES: {smiles}")

                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        img = Draw.MolToImage(mol, size=(300, 300))
                        st.image(img, caption="Target 2D Structure", width=250)

                    with st.spinner("⚙️ Gemini is evaluating 3 retrosynthetic pathways and calculating disconnections..."):
                        report_text, json_str = get_multi_pathway_retrosynthesis(target_input, smiles, selected_model)

                    st.markdown("---")
                    st.markdown(report_text)

                    st.markdown("---")
                    st.header("🎬 Animated Disconnection Simulations")
                    
                    try:
                        pathways = json.loads(json_str)
                        for path_idx, pathway in enumerate(pathways, 1):
                            st.subheader(f"📌 {pathway.get('pathway_name', f'Pathway {path_idx}')}")
                            steps = pathway.get("steps", [])
                            
                            cols = st.columns(len(steps) if len(steps) > 0 else 1)
                            for idx, step_info in enumerate(steps):
                                with cols[idx % len(cols)]:
                                    st.write(f"**Step {step_info.get('step', idx+1)}:** {step_info.get('title', '')}")
                                    t_smi = step_info.get("target_smiles", "")
                                    s_smi = step_info.get("synthon_smiles", "")
                                    
                                    if t_smi and s_smi:
                                        gif_path = f"temp_path{path_idx}_step{idx}.gif"
                                        out_gif = generate_disconnection_gif(t_smi, s_smi, gif_path)
                                        if out_gif:
                                            st.image(out_gif)
                    except Exception as json_err:
                        st.warning(f"Could not render animations for step disconnections: {json_err}")

                    st.markdown("---")
                    safe_name = "".join([c for c in target_input if c.isalnum() or c==' ']).strip().replace(' ', '_')
                    st.download_button(
                        label="📥 Download Full Synthesis Report (.md)",
                        data=report_text,
                        file_name=f"{safe_name}_3_Pathways_Retrosynthesis.md",
                        mime="text/markdown"
                    )

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
