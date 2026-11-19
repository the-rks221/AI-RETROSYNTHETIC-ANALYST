import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import pubchempy as pcp
import streamlit as st
import wikipedia
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
from supabase import Client, create_client
from wikipedia.exceptions import DisambiguationError, PageError

try:
    import py3Dmol
    from streamlit.components.v1 import html as st_html

    HAS_PY3DMOL = True
except ImportError:
    HAS_PY3DMOL = False

# ------------------------------------------------------------------------------
# 1. CONFIGURATION & ADMIN SETUP
# ------------------------------------------------------------------------------
ADMIN_USERNAME = "admin"

st.set_page_config(
    page_title="AI Retrosynthetic & Solubility Analyst",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 2. SUPABASE CLOUD DATABASE AUTHENTICATION
# ------------------------------------------------------------------------------
@st.cache_resource
def init_supabase() -> Client:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except KeyError:
        st.error("⚠️ Supabase credentials missing from Streamlit Secrets.")
        st.stop()


supabase: Client = init_supabase()


def make_hash(password: str) -> str:
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_user(username: str, password: str) -> bool:
    try:
        response = supabase.table("users").select("password").eq("username", username).execute()
        if len(response.data) > 0 and response.data[0]["password"] == make_hash(password):
            return True
        return False
    except Exception as e:
        st.error(f"Login error: {e}")
        return False


def add_user(username: str, password: str) -> bool:
    try:
        existing = supabase.table("users").select("username").eq("username", username).execute()
        if len(existing.data) > 0:
            return False
        supabase.table("users").insert(
            {"username": username, "password": make_hash(password)}
        ).execute()
        return True
    except Exception as e:
        st.error(f"Registration error: {e}")
        return False


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "saved_molecules" not in st.session_state:
    st.session_state["saved_molecules"] = {}
if "active_mol_query" not in st.session_state:
    st.session_state["active_mol_query"] = "Paracetamol"

# ------------------------------------------------------------------------------
# 3. AUTHENTICATION SCREEN (LOGIN / SIGN UP)
# ------------------------------------------------------------------------------
if not st.session_state["logged_in"]:
    st.markdown("<h1 class='main-title'>🔐 AI Retrosynthetic Analyst Portal</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])

        with tab_login:
            st.subheader("Welcome Back!")
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", key="btn_login"):
                if check_user(login_user, login_pass):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = login_user
                    st.success("Login successful! Loading dashboard...")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tab_signup:
            st.subheader("Create a New Account")
            new_user = st.text_input("Choose Username", key="signup_user")
            new_pass = st.text_input("Choose Password", type="password", key="signup_pass")
            if st.button("Register", key="btn_register"):
                if new_user.strip() and new_pass.strip():
                    if add_user(new_user, new_pass):
                        st.success("Account created successfully! Please switch to the Login tab.")
                    else:
                        st.error("Username already exists. Choose another.")
                else:
                    st.warning("Please fill in all fields.")
    st.stop()

# ------------------------------------------------------------------------------
# 4. GREEN CHEMISTRY RUBRIC, MECHANISM ANCHORS & OFFLINE SWAPS
# ------------------------------------------------------------------------------
CANONICAL_MECHANISMS = [
    "Esterification",
    "Amidation",
    "Oxidation",
    "Reduction",
    "Suzuki Coupling",
    "Buchwald-Hartwig",
]

SCORE_CAPS: Dict[str, float] = {
    "catalysis_atom": 30.0,
    "hazard_guide": 25.0,
    "mildness": 20.0,
    "selectivity": 15.0,
    "precedent": 10.0,
}

RUBRIC_MARKDOWN = """
### 📊 Green Chemistry Scoring Rubric (0–100 Scale)
* **Catalysis & Atom / E-Factor Profile (0–30 pts):** Catalytic turnover vs. stoichiometric loading, atom economy, and renewable/bio-derived feedstock potential.
* **Solvent & Reagent Hazard / Green Guides (0–25 pts):** Alignment with ACS GCI / CHEM21 solvent tiers and PubChem/GHS hazard profiles (CMR, aquatic toxicity, flammability/peroxides).
* **Reaction Conditions & Process Mildness (0–20 pts):** Mild temperature/pressure, avoiding cryogenic or harsh anhydrous/high-T extremes, and simplified workup.
* **Mechanistic & Chemoselectivity Fit (0–15 pts):** Match to transformation class and tolerance of competing functional groups.
* **Literature & Precedent Strength (0–10 pts):** Documented reaction precedent vs. heuristic transfer.

**Priority Tiers:**
* **Priority 1 (75 or more pts + Literature-Supported):** Catalytic / renewable / lower-hazard option with documented precedent.
* **Priority 2 (55–74.9 pts):** Milder conditions, safer solvent, or lower-hazard reagent with reasonable practicality.
* **Priority 3 (less than 55 pts or weak precedent):** Conditional swap requiring reaction optimization or bench validation.
"""

OFFLINE_GREEN_SWAPS: Dict[str, Dict[str, Any]] = {
    "esterification": {
        "mechanism_type": "Esterification",
        "mechanism_rationale": (
            "Replaces stoichiometric coupling agents or corrosive acyl chlorides/mineral acids "
            "with catalytic Brønsted/Lewis acid or mild dehydrative/transesterification pathways "
            "that shift equilibrium (e.g., water/alcohol removal or excess green alcohol/ester)."
        ),
        "candidates": [
            {
                "name": "Catalytic Solid/Heterogeneous Acid in Bio-Derived Solvent",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Proton/Lewis acid activation of carboxylic acid carbonyl for nucleophilic attack by alcohol",
                "reagents_conditions": "Amberlyst-15 or H2SO4 (1–5 mol%), 2-MeTHF or EtOAc, reflux with Dean-Stark / molecular sieves",
                "solvent_tier": "CHEM21 Recommended (2-MeTHF / EtOAc) over chlorinated or excess toxic solvents",
                "scores": {"catalysis_atom": 24.0, "hazard_guide": 20.0, "mildness": 14.0, "selectivity": 11.0, "precedent": 9.0},
                "source_ref": "ACS GCI Solvent Selection Guide; CHEM21 solvent guide; classic Fischer esterification precedent",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Avoid with acid-sensitive protecting groups, tertiary/hindered alcohols, or acid-labile heterocycles.",
            },
            {
                "name": "Enzymatic / Lipase-Catalyzed Esterification or Transesterification",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Acyl-enzyme intermediate formation with high chemo-, regio-, and enantioselectivity",
                "reagents_conditions": "Immobilized Candida antarctica lipase B (Novozym 435), tert-butanol or 2-MeTHF, 30–50 °C",
                "solvent_tier": "CHEM21 Recommended / Mild aqueous-organic or t-BuOH media",
                "scores": {"catalysis_atom": 26.0, "hazard_guide": 22.0, "mildness": 18.0, "selectivity": 14.0, "precedent": 7.0},
                "source_ref": "ACS GCI Biocatalysis & green esterification literature benchmarks",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "Medium",
                "caveat": "Sensitive to substrate steric bulk, water activity, and enzyme cost/stability at larger scales.",
            },
            {
                "name": "Boronic Acid-Catalyzed Direct Dehydrative Esterification",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Transient acyloxyboronate formation activating carboxylic acid toward alcohol",
                "reagents_conditions": "Arylboronic acid (2–5 mol%), 4 Å molecular sieves, toluene or 2-MeTHF, 60–80 °C",
                "solvent_tier": "CHEM21 Problematic (toluene) / Recommended (2-MeTHF swap)",
                "scores": {"catalysis_atom": 22.0, "hazard_guide": 16.0, "mildness": 14.0, "selectivity": 11.0, "precedent": 6.0},
                "source_ref": "Organoboron catalytic amidation/esterification literature & green metrics heuristics",
                "evidence_level": "HEURISTIC",
                "confidence": "Medium",
                "caveat": "Requires moisture control (sieves) and experimental screening for sterically hindered substrates.",
            },
        ],
    },
    "amidation": {
        "mechanism_type": "Amidation",
        "mechanism_rationale": (
            "Avoids high-MW coupling reagents (e.g., HATU/HOBt) that generate stoichiometric "
            "byproduct waste and hazardous aprotic dipolar solvents (DMF/DMAc), favoring "
            "lower-waste activating agents or catalytic boron routes in greener media."
        ),
        "candidates": [
            {
                "name": "Propylphosphonic Anhydride (T3P) Coupling in EtOAc or 2-MeTHF",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Cyclic anhydride activation of carboxylate yielding water-soluble phosphate byproducts",
                "reagents_conditions": "T3P (50 wt% in EtOAc/2-MeTHF), base (e.g., NMM or DIPEA), 0–25 °C",
                "solvent_tier": "ACS GCI Preferred / CHEM21 Recommended (EtOAc or 2-MeTHF vs. DMF/DCM)",
                "scores": {"catalysis_atom": 20.0, "hazard_guide": 20.0, "mildness": 17.0, "selectivity": 12.0, "precedent": 9.0},
                "source_ref": "ACS GCI Pharmaceutical Roundtable reagent guide; T3P green amidation precedents",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Check epimerization risk on alpha-chiral carboxylic acids and manage exothermic addition.",
            },
            {
                "name": "Catalytic Boronic Acid / Borinic Acid Direct Amidation",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Boron-mediated carboxylate activation enabling direct condensation with amines",
                "reagents_conditions": "2–10 mol% arylboronic/borinic catalyst, molecular sieves, 2-MeTHF or CPME, 50–80 °C",
                "solvent_tier": "CHEM21 Recommended (2-MeTHF) / Problematic (CPME peroxide risk)",
                "scores": {"catalysis_atom": 25.0, "hazard_guide": 18.0, "mildness": 14.0, "selectivity": 11.0, "precedent": 6.0},
                "source_ref": "Catalytic direct amidation literature & green chemistry atom-economy principles",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "Medium",
                "caveat": "Less reactive with weakly nucleophilic anilines or bulky amines; needs substrate optimization.",
            },
            {
                "name": "Amide Formation via Esters in Ethanol / Aqueous-Alcohol Media",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Direct aminolysis or catalytic/alkoxide-assisted acyl transfer from ester precursor",
                "reagents_conditions": "Esters + amine in EtOH or i-PrOH, optional catalytic base/La(OTf)3 or TBD, 25–70 °C",
                "solvent_tier": "CHEM21 Recommended (EtOH / i-PrOH; avoids DMF/NMP)",
                "scores": {"catalysis_atom": 21.0, "hazard_guide": 21.0, "mildness": 15.0, "selectivity": 10.0, "precedent": 6.0},
                "source_ref": "CHEM21 solvent guide & ester aminolysis heuristics",
                "evidence_level": "HEURISTIC",
                "confidence": "Medium",
                "caveat": "Often slow for hindered substrates; verify no competing transesterification if mixed alcohols are used.",
            },
        ],
    },
    "oxidation": {
        "mechanism_type": "Oxidation",
        "mechanism_rationale": (
            "Replaces stoichiometric toxic heavy-metal oxidants (e.g., Cr(VI), MnO2) or "
            "chlorinated solvents with catalytic aerobic or hypochlorite/peroxide pathways "
            "matched to the target oxidation state."
        ),
        "candidates": [
            {
                "name": "Catalytic Cu / TEMPO Aerobic Alcohol Oxidation",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Cu(I)/TEMPO redox cycle using O2/air as terminal oxidant with water as sole byproduct",
                "reagents_conditions": "1–5 mol% Cu salt, ligand (e.g., bpy/DMAP), 1–5 mol% TEMPO, air/O2, MeCN or 2-MeTHF, RT",
                "solvent_tier": "CHEM21 Recommended (2-MeTHF) / Problematic (MeCN) + benign O2 oxidant",
                "scores": {"catalysis_atom": 28.0, "hazard_guide": 21.0, "mildness": 17.0, "selectivity": 12.0, "precedent": 9.0},
                "source_ref": "ACS GCI oxidation principles; aerobic Cu/TEMPO literature precedents",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Check oxygen-solvent flammability limits in headspace and sulfur/amine catalyst poisoning.",
            },
            {
                "name": "Catalytic TEMPO / NaOCl (Household Bleach) Oxidation",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Oxoammonium cation generated catalytically from TEMPO oxidizes alcohol selectively",
                "reagents_conditions": "0.5–2 mol% TEMPO, cat. KBr/NaHCO3, aq. NaOCl, EtOAc or i-PrOAc, 0–15 °C",
                "solvent_tier": "CHEM21 Recommended (EtOAc / i-PrOAc) + aqueous phase",
                "scores": {"catalysis_atom": 23.0, "hazard_guide": 18.0, "mildness": 17.0, "selectivity": 12.0, "precedent": 9.0},
                "source_ref": "Anelli oxidation precedent; green solvent replacement of DCM with EtOAc",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Control pH (~8.5–9.5) and temperature to prevent over-oxidation to carboxylic acid or chlorination.",
            },
            {
                "name": "Catalytic H2O2 Oxidation with Safe Polyoxometalate / Mn / Fe Catalyst",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Metal-peroxo/oxo oxygen atom transfer using hydrogen peroxide (water byproduct)",
                "reagents_conditions": "0.5–3 mol% catalyst, 30% aq. H2O2, EtOH or t-BuOH, 20–60 °C",
                "solvent_tier": "CHEM21 Recommended (EtOH / t-BuOH) + H2O2",
                "scores": {"catalysis_atom": 25.0, "hazard_guide": 18.0, "mildness": 14.0, "selectivity": 9.0, "precedent": 6.0},
                "source_ref": "Peroxide oxidation literature & green chemistry oxidant heuristics",
                "evidence_level": "HEURISTIC",
                "confidence": "Medium",
                "caveat": "Peroxide decomposition/exotherm risk; screen catalyst selectivity on sensitive alkenes/heterocycles.",
            },
        ],
    },
    "reduction": {
        "mechanism_type": "Carbonyl / Nitro Reduction",
        "mechanism_rationale": (
            "Balances hydride delivery, catalytic hydrogenation, or catalytic transfer hydrogenation "
            "against competing reducible groups (alkenes, aryl halides, nitriles, esters) while "
            "avoiding pyrophoric hydrides (LiAlH4), stoichiometric heavy metals, or toxic chlorinated/dipolar solvents."
        ),
        "candidates": [
            {
                "name": "Catalytic Transfer Hydrogenation (CTH) in Alcohol Donor Media",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Metal-catalyzed hydrogen transfer from an alcohol/formate donor to C=O or -NO2 without pressurized H2 gas",
                "reagents_conditions": "Pd/C, Ru, or Ir catalyst + HCOONH4 / HCOONa or i-PrOH, EtOH or 2-MeTHF, 40–80 °C",
                "solvent_tier": "CHEM21 Recommended (EtOH, i-PrOH, 2-MeTHF) avoiding pressurized H2 cylinders",
                "scores": {"catalysis_atom": 24.0, "hazard_guide": 21.0, "mildness": 16.0, "selectivity": 11.0, "precedent": 9.0},
                "source_ref": "ACS GCI reduction/hydrogenation guidance; CHEM21 solvent guide; transfer hydrogenation literature",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Check for competing dehalogenation on Ar-Br/Ar-I or alkene saturation; catalyst-free/chemo-tuned conditions may be needed.",
            },
            {
                "name": "NaBH4 Reduction in Protic Alcohol / Aqueous-Alcohol Medium",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Nucleophilic hydride transfer to polarized carbonyls (aldehydes/ketones) under mild conditions",
                "reagents_conditions": "NaBH4 (0.3–1.0 equiv), MeOH or EtOH (or THF/H2O), 0–25 °C",
                "solvent_tier": "CHEM21 Recommended (EtOH) / Problematic (MeOH) — safer than LiAlH4 in Et2O/THF",
                "scores": {"catalysis_atom": 18.0, "hazard_guide": 19.0, "mildness": 18.0, "selectivity": 12.0, "precedent": 9.0},
                "source_ref": "Standard carbonyl reduction precedent; ACS GCI / CHEM21 solvent selection guides",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Does not typically reduce esters, amides, or standard nitro groups directly; watch for exotherms/H2 evolution in protic media.",
            },
            {
                "name": "Aqueous / Mild Iron-Mediated Nitro-to-Aniline Reduction",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Single-electron / proton-coupled reduction of nitroaromatics to anilines with reduced heavy-metal waste",
                "reagents_conditions": "Fe powder or catalytic Fe/additive, aq. EtOH or aq. NH4Cl, 50–80 °C",
                "solvent_tier": "CHEM21 Recommended (EtOH / H2O) replacing SnCl2 or high-pressure H2 where appropriate",
                "scores": {"catalysis_atom": 20.0, "hazard_guide": 20.0, "mildness": 15.0, "selectivity": 11.0, "precedent": 8.0},
                "source_ref": "Nitro reduction literature & green chemistry reagent alternatives",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "Medium",
                "caveat": "Iron sludge handling and chemoselectivity near reducible alkenes, ketones, or labile groups require bench verification.",
            },
        ],
    },
    "suzuki coupling": {
        "mechanism_type": "C-C Cross-Coupling (Suzuki-Miyaura)",
        "mechanism_rationale": (
            "Preserves Pd-catalyzed oxidative addition / transmetalation / reductive elimination "
            "while shifting away from hazardous dipolar aprotic solvents (DMF/DMAc/NMP) and "
            "excessive Pd loadings toward aqueous-organic or greener ether/alcohol media."
        ),
        "candidates": [
            {
                "name": "Aqueous Ethanol / 2-MeTHF Suzuki-Miyaura with Low-Loading Pd",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Pd(0)/Pd(II) catalytic cycle with organoboron transmetalation under mild base",
                "reagents_conditions": "0.5–2 mol% Pd catalyst + ligand, K2CO3 or K3PO4, EtOH/H2O or 2-MeTHF/H2O, 50–80 °C",
                "solvent_tier": "CHEM21 Recommended (EtOH, 2-MeTHF, H2O) replacing DMF/toluene",
                "scores": {"catalysis_atom": 24.0, "hazard_guide": 21.0, "mildness": 16.0, "selectivity": 12.0, "precedent": 9.0},
                "source_ref": "ACS GCI Suzuki-Miyaura solvent guide; CHEM21 solvent selection guide",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Aryl halides/boronic acids with low aqueous solubility may need cosolvent or phase-transfer tuning.",
            },
            {
                "name": "Micellar / Surfactant-Enabled Aqueous Suzuki Coupling",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Pd catalysis inside surfactant nanomicelles in water, minimizing organic solvent volume",
                "reagents_conditions": "1–2 mol% Pd catalyst/ligand, ~2 wt% aqueous surfactant (e.g., TPGS-750-M), base, RT–45 °C",
                "solvent_tier": "Water-dominant medium (minimal organic cosolvent)",
                "scores": {"catalysis_atom": 25.0, "hazard_guide": 22.0, "mildness": 16.0, "selectivity": 10.0, "precedent": 6.0},
                "source_ref": "Lipshutz micellar catalysis literature & green cross-coupling heuristics",
                "evidence_level": "HEURISTIC",
                "confidence": "Medium",
                "caveat": "Emulsion workup and surfactant compatibility should be tested for your specific substrate.",
            },
            {
                "name": "Boronate Ester (Bpin) Swap to Curb Hydrolytic Protodeboronation",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Stabilizes boron species against premature cleavage during slow cross-coupling",
                "reagents_conditions": "Ar-Bpin + Ar'-X, Pd catalyst, mild base, 2-MeTHF/H2O, 60–80 °C",
                "solvent_tier": "CHEM21 Recommended (2-MeTHF) + aqueous base",
                "scores": {"catalysis_atom": 20.0, "hazard_guide": 19.0, "mildness": 14.0, "selectivity": 12.0, "precedent": 7.0},
                "source_ref": "Cross-coupling selectivity heuristics & boron reagent stability guidelines",
                "evidence_level": "HEURISTIC",
                "confidence": "Medium",
                "caveat": "Bpin has lower atom economy than free boronic acid; use mainly when free acid protodeboronates fast.",
            },
        ],
    },
    "buchwald-hartwig": {
        "mechanism_type": "C-N Cross-Coupling (Buchwald-Hartwig Amination)",
        "mechanism_rationale": (
            "Forms aryl/heteroaryl C–N bonds via Pd-catalyzed oxidative addition, amine coordination/deprotonation, "
            "and reductive elimination while shifting away from hazardous dipolar aprotic solvents (DMF/NMP/dioxane) "
            "and high catalyst loadings toward greener ethers or aqueous/micellar media."
        ),
        "candidates": [
            {
                "name": "Low-Loading Pd/Ligand Coupling in 2-MeTHF or CPME",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Pd(0)/Pd(II) catalytic C–N bond formation using bulky/biaryl phosphine ligands and mild base",
                "reagents_conditions": "0.5–2 mol% Pd source + ligand, Cs2CO3 or K3PO4 (or NaOtBu), 2-MeTHF or CPME, 60–90 °C",
                "solvent_tier": "CHEM21 Recommended (2-MeTHF) / Problematic (CPME) — replaces dioxane/DMF",
                "scores": {"catalysis_atom": 23.0, "hazard_guide": 19.0, "mildness": 15.0, "selectivity": 12.0, "precedent": 9.0},
                "source_ref": "ACS GCI / CHEM21 ether solvent guidance; Buchwald-Hartwig cross-coupling literature",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Strong alkoxide bases (e.g., NaOtBu) can cleave base-sensitive esters/nitriles or epimerize chiral centers; test K3PO4/Cs2CO3.",
            },
            {
                "name": "Aqueous / Micellar Surfactant-Enabled Buchwald-Hartwig Coupling",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Pd-catalyzed C–N coupling within aqueous surfactant nanoreactors with minimal organic cosolvent",
                "reagents_conditions": "1–2 mol% Pd catalyst/ligand, aqueous surfactant (e.g., TPGS-750-M or water/t-BuOH), base, 25–60 °C",
                "solvent_tier": "Water-dominant / low-organic medium reducing solvent waste and thermal load",
                "scores": {"catalysis_atom": 25.0, "hazard_guide": 21.0, "mildness": 16.0, "selectivity": 10.0, "precedent": 6.0},
                "source_ref": "Micellar aqueous Buchwald-Hartwig literature & green chemistry solvent reduction",
                "evidence_level": "HEURISTIC",
                "confidence": "Medium",
                "caveat": "Substrate solubility in micellar phases, amine basicity/nucleophilicity, and catalyst/surfactant removal need experimental validation.",
            },
            {
                "name": "Cu-Catalyzed Ullmann-Type C-N Coupling in Greener Solvent",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Copper-mediated C–N bond formation offering a lower-cost metal alternative to palladium",
                "reagents_conditions": "CuI or Cu2O (2–10 mol%), diamine/amino-acid ligand, K2CO3 or K3PO4, 2-MeTHF or EtOH/H2O, 70–100 °C",
                "solvent_tier": "CHEM21 Recommended (2-MeTHF / EtOH) with lower-cost Cu catalyst vs. Pd",
                "scores": {"catalysis_atom": 20.0, "hazard_guide": 18.0, "mildness": 12.0, "selectivity": 10.0, "precedent": 7.0},
                "source_ref": "Copper-catalyzed C–N coupling literature & green metal substitution heuristics",
                "evidence_level": "HEURISTIC",
                "confidence": "Low",
                "caveat": "Often requires higher temperatures and higher Cu loading than Pd; aryl iodides/bromides react faster than aryl chlorides.",
            },
        ],
    },
}

KNOWN_DRUG_FALLBACKS: Dict[str, Dict[str, Any]] = {
    "paracetamol": {
        "cid": 1983,
        "iupac_name": "N-(4-hydroxyphenyl)acetamide",
        "molecular_weight": 151.163,
        "xlogp": 0.46,
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "synonyms": ["paracetamol", "acetaminophen"],
    },
    "acetaminophen": {
        "cid": 1983,
        "iupac_name": "N-(4-hydroxyphenyl)acetamide",
        "molecular_weight": 151.163,
        "xlogp": 0.46,
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "synonyms": ["acetaminophen", "paracetamol"],
    },
    "ibuprofen": {
        "cid": 3672,
        "iupac_name": "2-[4-(2-methylpropyl)phenyl]propanoic acid",
        "molecular_weight": 206.28,
        "xlogp": 3.5,
        "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "synonyms": ["ibuprofen"],
    },
    "aspirin": {
        "cid": 2244,
        "iupac_name": "2-acetyloxybenzoic acid",
        "molecular_weight": 180.16,
        "xlogp": 1.2,
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "synonyms": ["aspirin", "acetylsalicylic acid"],
    },
}


def clamp_scores(raw_scores: Optional[Dict[str, Any]]) -> Dict[str, float]:
    src = raw_scores or {}
    clamped: Dict[str, float] = {}
    for key, cap in SCORE_CAPS.items():
        try:
            val = float(src.get(key, 0.0))
        except (TypeError, ValueError):
            val = 0.0
        if np.isnan(val) or np.isinf(val):
            val = 0.0
        clamped[key] = round(max(0.0, min(cap, val)), 1)
    return clamped


@dataclass
class GreenCandidate:
    name: str
    origin: str
    mechanism_role: str
    reagents_conditions: str
    solvent_tier: str
    scores: Dict[str, float]
    source_ref: str
    evidence_level: str
    confidence: str
    caveat: str
    hard_block: Optional[str] = None

    def __post_init__(self):
        self.scores = clamp_scores(self.scores)

    @property
    def total_score(self) -> float:
        return round(sum(self.scores.values()), 1)

    @property
    def priority_tier(self) -> str:
        if self.hard_block:
            return f"Blocked ({self.hard_block})"
        s = self.total_score
        if s >= 75.0 and self.evidence_level == "LITERATURE-SUPPORTED" and self.confidence in ("High", "Medium"):
            return "Priority 1"
        if s >= 55.0:
            return "Priority 2"
        return "Priority 3"

    @property
    def badge(self) -> str:
        return f"**[{self.evidence_level} | Confidence: {self.confidence}]**"


def normalize_reaction_key(mech_or_step: str) -> Optional[str]:
    text = (mech_or_step or "").lower()
    if any(k in text for k in ["buchwald", "hartwig", "c-n coupling", "c–n coupling", "arylation", "aminat"]):
        return "buchwald-hartwig"
    if any(k in text for k in ["reduct", "hydrogenat", "transfer hydrogenation", "nitro", "carbonyl"]):
        return "reduction"
    if any(k in text for k in ["suzuki", "cross-coupling", "c-c coupling", "biaryl"]):
        return "suzuki coupling"
    if any(k in text for k in ["esterif", "ester"]):
        return "esterification"
    if any(k in text for k in ["amidat", "amide", "peptide coupling"]):
        return "amidation"
    if any(k in text for k in ["oxidat", "oxidize", "dehydrogenat"]):
        return "oxidation"
    return None


def get_rdkit_profile(mol) -> Dict[str, Any]:
    smarts_map = {
        "ester": "[CX3](=O)[OX2][#6]",
        "amide": "[CX3](=O)[NX3]",
        "carbonyl": "[CX3]=[OX1]",
        "nitro": "[NX3](=[OX1])[OX1-,OX2]",
        "alkene": "[CX3]=[CX3]",
        "alkyne": "[CX2]#[CX1]",
        "aryl_halide": "[c][Cl,Br,I]",
        "halide": "[c,C][F,Cl,Br,I]",
        "primary_oh": "[CH2X4][OX2H1]",
    }
    detected = [
        name
        for name, smarts in smarts_map.items()
        if (pat := Chem.MolFromSmarts(smarts)) is not None and mol.HasSubstructMatch(pat)
    ]
    return {
        "mol_wt": round(Descriptors.MolWt(mol), 2),
        "clogp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "functional_groups": detected,
    }


def get_offline_green_steps(
    molecule_name: str,
    rdkit_profile: Dict[str, Any],
    selected_reactions: Optional[List[str]] = None,
) -> List[dict]:
    keys_to_use: List[str] = []
    if selected_reactions:
        for r in selected_reactions:
            k = normalize_reaction_key(r)
            if k and k not in keys_to_use:
                keys_to_use.append(k)

    if not keys_to_use:
        fgs = set(rdkit_profile.get("functional_groups", []))
        if "amide" in fgs:
            keys_to_use.append("amidation")
        if "ester" in fgs:
            keys_to_use.append("esterification")
        if "primary_oh" in fgs:
            keys_to_use.append("oxidation")
        if "nitro" in fgs or "carbonyl" in fgs:
            keys_to_use.append("reduction")
        if "halide" in fgs:
            keys_to_use.append("suzuki coupling")
        if "aryl_halide" in fgs:
            keys_to_use.append("buchwald-hartwig")
        if not keys_to_use:
            keys_to_use = ["amidation", "esterification", "oxidation", "reduction", "suzuki coupling", "buchwald-hartwig"]

    return [
        {
            "step_name": f"Green Swap Evaluation — {OFFLINE_GREEN_SWAPS[k]['mechanism_type']} ({molecule_name})",
            "mechanism_type": OFFLINE_GREEN_SWAPS[k]["mechanism_type"],
            "mechanism_rationale": OFFLINE_GREEN_SWAPS[k]["mechanism_rationale"],
            "green_candidates": OFFLINE_GREEN_SWAPS[k]["candidates"],
        }
        for k in keys_to_use
    ]


def blend_step_with_offline_swaps(step: dict, max_candidates: int = 5) -> dict:
    raw_mech = step.get("mechanism_type", "")
    key = normalize_reaction_key(raw_mech)
    offline_entry = OFFLINE_GREEN_SWAPS.get(key) if key else None

    canonical_mech = (
        offline_entry["mechanism_type"]
        if offline_entry
        else (raw_mech if raw_mech in CANONICAL_MECHANISMS else "General Transformation")
    )

    combined_rationale = str(step.get("mechanism_rationale", "")).strip()
    if offline_entry and offline_entry["mechanism_rationale"] not in combined_rationale:
        combined_rationale = (
            f"{combined_rationale} | *Reference Mechanism Anchor:* {offline_entry['mechanism_rationale']}"
            if combined_rationale
            else offline_entry["mechanism_rationale"]
        )

    merged: List[dict] = []
    seen_names = set()

    def _add_cand(c_dict: dict, default_origin: str):
        norm_name = re.sub(r"\s+", " ", str(c_dict.get("name", "")).strip().lower())
        if not norm_name or norm_name in seen_names:
            return
        seen_names.add(norm_name)
        item = dict(c_dict)
        item["origin"] = item.get("origin") or default_origin
        merged.append(item)

    for llm_c in step.get("green_candidates", []):
        _add_cand(llm_c, "LLM")
    if offline_entry:
        for off_c in offline_entry.get("candidates", []):
            _add_cand(off_c, "OFFLINE_GREEN_SWAPS")

    return {
        **step,
        "mechanism_type": canonical_mech,
        "mechanism_rationale": combined_rationale,
        "green_candidates": merged[:max_candidates],
    }


def score_and_format_green_steps(steps_data: List[dict], rdkit_profile: Dict[str, Any]) -> str:
    out = [RUBRIC_MARKDOWN, "---"]
    fg_set = set(rdkit_profile.get("functional_groups", []))

    for idx, step in enumerate(steps_data, start=1):
        step_title = step.get("step_name", f"Step {idx}")
        mech_type = step.get("mechanism_type", "General Transformation")
        mech_notes = step.get("mechanism_rationale", "")

        candidates: List[GreenCandidate] = []
        for raw_c in step.get("green_candidates", []):
            origin = raw_c.get("origin", "LLM")
            scores = clamp_scores(raw_c.get("scores", {}))
            caveat = raw_c.get("caveat", "Validate on actual substrate before scale-up.")
            evidence = str(raw_c.get("evidence_level", "HEURISTIC")).upper()
            if evidence not in ("LITERATURE-SUPPORTED", "HEURISTIC"):
                evidence = "HEURISTIC"
            confidence = str(raw_c.get("confidence", "Medium")).capitalize()
            if confidence not in ("High", "Medium", "Low"):
                confidence = "Medium"

            cond_lower = str(raw_c.get("reagents_conditions", "")).lower()
            if "reduct" in mech_type.lower() and "alkene" in fg_set and ("h2" in cond_lower or "hydrogen" in cond_lower):
                scores["selectivity"] = max(0.0, scores["selectivity"] - 3.0)
                caveat += " (RDKit alert: alkene motif detected; verify C=C vs. target chemoselectivity.)"

            candidates.append(
                GreenCandidate(
                    name=raw_c.get("name", "Alternative Protocol"),
                    origin=origin,
                    mechanism_role=raw_c.get("mechanism_role", mech_type),
                    reagents_conditions=raw_c.get("reagents_conditions", "Specify conditions"),
                    solvent_tier=raw_c.get("solvent_tier", "ACS GCI / CHEM21 check needed"),
                    scores=scores,
                    source_ref=raw_c.get("source_ref", "Green chemistry guide / reaction literature"),
                    evidence_level=evidence,
                    confidence=confidence,
                    caveat=caveat,
                    hard_block=raw_c.get("hard_block"),
                )
            )

        candidates.sort(key=lambda c: (c.hard_block is not None, -c.total_score))

        out.append(f"### Step {idx}: {step_title}")
        out.append(f"- **Mechanism / Transformation Class:** `{mech_type}`")
        if mech_notes:
            out.append(f"- **Mechanistic & Catalyst Role:** {mech_notes}")

        for c in candidates:
            s = c.scores
            breakdown = (
                f"Cat/Atom: {s['catalysis_atom']}/30 | "
                f"Hazard/Guide: {s['hazard_guide']}/25 | "
                f"Mildness: {s['mildness']}/20 | "
                f"Selectivity: {s['selectivity']}/15 | "
                f"Precedent: {s['precedent']}/10"
            )
            out.append(
                f"\n**{c.priority_tier} — {c.name}** (`{c.total_score}/100`) · *Source: `{c.origin}`* {c.badge}\n"
                f"  - *Mechanism Role:* {c.mechanism_role}\n"
                f"  - *Conditions & Solvent Tier:* {c.reagents_conditions} | *{c.solvent_tier}*\n"
                f"  - *Score Breakdown:* {breakdown}\n"
                f"  - *Reference / Rationale:* {c.source_ref}\n"
                f"  - *Caveat:* {c.caveat}"
            )
        out.append("\n---")

    return "\n".join(out)


# ------------------------------------------------------------------------------
# 5. SOLUBILITY, CACHED CHEM-INFO & RETROSYNTHESIS HELPERS
# ------------------------------------------------------------------------------
def extract_compound_fields(c) -> Optional[Dict[str, Any]]:
    if c is None:
        return None
    smiles = (
        getattr(c, "smiles", None)
        or getattr(c, "isomeric_smiles", None)
        or getattr(c, "connectivity_smiles", None)
        or getattr(c, "canonical_smiles", None)
    )
    try:
        mw_float = float(c.molecular_weight) if getattr(c, "molecular_weight", None) is not None else None
    except (TypeError, ValueError):
        mw_float = None
    if not smiles or mw_float is None or mw_float <= 0:
        return None
    try:
        xlogp_float = float(c.xlogp) if getattr(c, "xlogp", None) is not None else None
    except (TypeError, ValueError):
        xlogp_float = None
    synonyms = getattr(c, "synonyms", None) or []
    return {
        "cid": getattr(c, "cid", None),
        "iupac_name": getattr(c, "iupac_name", None),
        "molecular_weight": mw_float,
        "xlogp": xlogp_float,
        "smiles": smiles,
        "synonyms": list(synonyms[:10]),
    }


def is_drug(compound_data: Dict[str, Any]) -> bool:
    non_drugs = {
        "water", "methane", "benzene", "toluene", "hexane", "cyclohexane",
        "sodium chloride", "table salt", "ethanol", "methanol", "acetone",
    }
    syns = [str(s).strip().lower() for s in (compound_data.get("synonyms") or [])[:5]]
    if syns and syns[0] in non_drugs:
        return False
    return True


@st.cache_data(ttl=3600, show_spinner=False)
def cached_pubchem_lookup(mol_name: str) -> Dict[str, Any]:
    cleaned = mol_name.strip()
    key = cleaned.lower()

    compounds = []
    for attempt in range(2):
        try:
            compounds = pcp.get_compounds(cleaned, "name") or []
            if not compounds and key != cleaned:
                compounds = pcp.get_compounds(key, "name") or []
            if compounds:
                break
        except Exception:
            if attempt == 0:
                time.sleep(0.8)

    for comp in compounds:
        parsed = extract_compound_fields(comp)
        if parsed is not None:
            return {"status": "ok", "data": parsed, "source": "pubchem"}

    if key in KNOWN_DRUG_FALLBACKS:
        return {"status": "ok", "data": KNOWN_DRUG_FALLBACKS[key], "source": "fallback"}

    return {"status": "not_found", "data": None, "source": "none"}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_sdf_data(cid: Optional[int]) -> Optional[str]:
    if not cid:
        return None
    for rec_type in ("3d", "2d"):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF/?record_type={rec_type}"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return response.read().decode("utf-8")
        except Exception:
            continue
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_wiki_summary(drug_name: str) -> str:
    try:
        return wikipedia.summary(drug_name, sentences=3, auto_suggest=False)
    except DisambiguationError as e:
        try:
            return wikipedia.summary(e.options[0], sentences=3)
        except Exception:
            return "Multiple Wikipedia entries found for this query."
    except PageError:
        return "No detailed summary found on Wikipedia for this query."
    except Exception:
        return "Could not retrieve literature text from Wikipedia."


def generate_solubility_curves(mol_weight: float, logp: float):
    concentrations = np.linspace(0, 100, 50)
    solubility_conc = (1 - np.exp(-0.1 * concentrations)) * (50.0 / (logp + 1.5))

    temps = np.linspace(273, 373, 50)
    enthalpy_sol = 25000.0 + (mol_weight * 10.0)
    solubility_temp = np.exp(-(enthalpy_sol / 8.314) * (1.0 / temps - 1.0 / 298.15)) * (10.0 ** (-logp))

    pressures = np.linspace(1, 100, 50)
    molar_volume = mol_weight / 1.2
    solubility_press = np.exp((molar_volume * (pressures - 1.0)) / (8.314 * 298.15)) * (10.0 ** (-logp))

    ph_vals = np.linspace(2.0, 12.0, 100)
    s0 = max(1e-4, 0.05 * (10.0 ** (-0.5 * logp)))
    pka_est = 4.5
    solubility_ph = s0 * (1.0 + 10.0 ** (ph_vals - pka_est))

    return concentrations, solubility_conc, temps, solubility_temp, pressures, solubility_press, ph_vals, solubility_ph


def get_aquatic_summary(drug_name: str, compound_data: Dict[str, Any]) -> str:
    wiki_summary = cached_wiki_summary(drug_name)
    iupac = compound_data.get("iupac_name") or drug_name
    mw = compound_data.get("molecular_weight")
    xlogp = compound_data.get("xlogp")

    return (
        f"### 💧 Aqueous Solubility & Aquatic Behaviour Summary\n\n"
        f"**Chemical Identity:** `{iupac}` | **MW:** {mw} g/mol\n\n"
        f"**Aquatic Behavior:** Estimated XLogP3 is **{xlogp if xlogp is not None else 'Unknown'}**, "
        f"influencing aqueous solubility, ionization/pH partitioning, and environmental/biological transport.\n\n"
        f"**Reference Summary:**\n> {wiki_summary}"
    )


def generate_disconnection_gif(target_smiles: str, synthon_smiles: str, output_gif_path: str) -> Optional[str]:
    try:
        target_mol = Chem.MolFromSmiles(target_smiles)
        synthon_mol = Chem.MolFromSmiles(synthon_smiles)
        if not target_mol or not synthon_mol:
            return None

        frames: List[Image.Image] = []
        img_target = Draw.MolToImage(target_mol, size=(240, 240))
        frames.extend([img_target] * 2)

        img_highlight = Draw.MolToImage(
            target_mol,
            size=(240, 240),
            highlightAtoms=list(range(target_mol.GetNumAtoms())),
            highlightColor=(1, 0.4, 0.4),
        )
        frames.extend([img_highlight] * 2)

        img_synthons = Draw.MolToImage(synthon_mol, size=(240, 240))
        frames.extend([img_synthons] * 3)

        frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], duration=500, loop=0)
        return output_gif_path
    except Exception:
        return None


def extract_text_and_json(raw_response) -> tuple[str, dict]:
    if isinstance(raw_response, list):
        parts = [block["text"] if isinstance(block, dict) and "text" in block else str(block) for block in raw_response]
        text_result = "\n".join(parts)
    else:
        text_result = str(raw_response)

    parsed_json: dict = {"pathways": [], "green_steps": []}
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", text_result)
    if json_match:
        raw_json = json_match.group(1).strip()
        text_result = text_result.replace(json_match.group(0), "").strip()
        try:
            data = json.loads(raw_json)
            if isinstance(data, list):
                parsed_json["pathways"] = data
            elif isinstance(data, dict):
                parsed_json["pathways"] = data.get("pathways", [])
                parsed_json["green_steps"] = data.get("green_steps", [])
        except json.JSONDecodeError:
            pass

    return text_result, parsed_json


def get_gemini_retrosynthesis(molecule_name: str, smiles: str, rdkit_profile: Dict[str, Any], model_name: str):
    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is not configured.")

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)

    template = """
    You are a synthetic organic chemist and green chemistry specialist. Analyze:
    Name: {molecule}
    SMILES: {smiles}
    RDKit Descriptors & Motifs: {rdkit_profile}

    Provide:
    ## 1. Pathway A: Classical/Industrial Route (highest commercial feasibility)
    ## 2. Pathway B: Alternative Green Chemistry Route (mild conditions / eco-friendly)
    ## 3. Pathway C: Convergent/Novel FGI Route
    ## 4. Synthetic Equivalents & Market Availability (cheapest / accessible commercial precursors)
    ## 5. Comprehensive Feasibility Comparison Table
    | Metric | Pathway A | Pathway B | Pathway C |
    |---|---|---|---|
    | Number of Synthetic Steps | ... | ... | ... |
    | Commercial Availability of Starting Materials | ... | ... | ... |
    | Atom Economy / Green Metrics | ... | ... | ... |
    | Total Estimated Yield | ... | ... | ... |
    | Overall Feasibility Score (Out of 10) | ... | ... | ... |
    | Best Suited Application | ... | ... | ... |

    At the VERY END, output a single ```json ... ``` object with:
    - "pathways": list of 3 pathway objects with "pathway_name" and "steps" (each having "step", "title", "target_smiles", "synthon_smiles")
    - "green_steps": list of 2-4 step objects, each with:
      - "step_name": string
      - "mechanism_type": one of [{canonical_list}]
      - "mechanism_rationale": string
      - "green_candidates": list of 2 objects with "name", "origin": "LLM", "mechanism_role", "reagents_conditions",
        "solvent_tier", "scores": {{"catalysis_atom": 0-30, "hazard_guide": 0-25, "mildness": 0-20, "selectivity": 0-15, "precedent": 0-10}},
        "source_ref", "evidence_level": "LITERATURE-SUPPORTED" | "HEURISTIC", "confidence": "High" | "Medium" | "Low", "caveat"
    """

    prompt = PromptTemplate(
        input_variables=["molecule", "smiles", "rdkit_profile", "canonical_list"],
        template=template,
    )
    chain = prompt | llm | StrOutputParser()
    raw_result = chain.invoke(
        {
            "molecule": molecule_name,
            "smiles": smiles,
            "rdkit_profile": json.dumps(rdkit_profile),
            "canonical_list": ", ".join(CANONICAL_MECHANISMS),
        }
    )
    return extract_text_and_json(raw_result)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_gemini_synthesis(mol_name: str, smiles: str, profile_json: str, model_name: str):
    return get_gemini_retrosynthesis(mol_name, smiles, json.loads(profile_json), model_name)


def build_and_store_molecule_analysis(
    mol_name: str,
    model_name: str,
    selected_rxns: Optional[List[str]],
    include_wiki: bool,
) -> Dict[str, Any]:
    """Runs full analysis for a molecule and stores it in session_state for reuse."""
    lookup = cached_pubchem_lookup(mol_name)
    if lookup["status"] != "ok" or not lookup["data"]:
        return {"error": "There is not enough data to show."}

    comp_data = lookup["data"]
    if not is_drug(comp_data):
        return {"error": "The given molecule is not a drug."}

    smiles = comp_data["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "There is not enough data to show."}

    profile = get_rdkit_profile(mol)
    safe_logp = float(comp_data["xlogp"]) if comp_data.get("xlogp") is not None else 2.0
    curves = generate_solubility_curves(float(comp_data["molecular_weight"]), safe_logp)
    aquatic_md = get_aquatic_summary(mol_name, comp_data) if include_wiki else (
        f"**Chemical Identity:** `{comp_data.get('iupac_name') or mol_name}` | "
        f"**MW:** {comp_data['molecular_weight']} g/mol | **XLogP3:** {comp_data.get('xlogp', 'Unknown')}"
    )

    report_text = ""
    parsed_json: dict = {"pathways": [], "green_steps": []}
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            report_text, parsed_json = cached_gemini_synthesis(
                mol_name,
                smiles,
                json.dumps(profile),
                model_name,
            )
        except Exception as e:
            report_text = f"⚠️ Gemini synthesis fallback (`{e}`). Using built-in green chemistry rankings."

    raw_green_steps = parsed_json.get("green_steps") or []
    if raw_green_steps:
        steps_to_score = [blend_step_with_offline_swaps(s) for s in raw_green_steps]
    else:
        steps_to_score = get_offline_green_steps(mol_name, profile, selected_reactions=selected_rxns)

    green_md = score_and_format_green_steps(steps_to_score, profile)

    record = {
        "query": mol_name,
        "source": lookup.get("source", "pubchem"),
        "comp_data": comp_data,
        "smiles": smiles,
        "profile": profile,
        "curves": curves,
        "aquatic_md": aquatic_md,
        "report_text": report_text,
        "green_md": green_md,
        "pathways": parsed_json.get("pathways") or [],
    }
    st.session_state["saved_molecules"][mol_name.strip().lower()] = record
    return record


# ------------------------------------------------------------------------------
# 6. MAIN STREAMLIT DASHBOARD
# ------------------------------------------------------------------------------
st.markdown("<h1 class='main-title'>🧪 AI Retrosynthetic & Aqueous Solubility Workbench</h1>", unsafe_allow_html=True)

try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

with st.sidebar:
    st.write(f"👤 Logged in as: **{st.session_state['username']}**")
    st.markdown(f"🛡️ **Role:** `{'Administrator' if st.session_state['username'] == ADMIN_USERNAME else 'Standard User'}`")
    if st.button("🚪 Logout", key="btn_logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.rerun()

    st.markdown("---")
    st.header("⚙️ Gemini & Cache Controls")
    if not os.environ.get("GOOGLE_API_KEY"):
        manual_key = st.text_input("Gemini API Key (if not in Secrets)", type="password", key="manual_gemini_key")
        if manual_key.strip():
            os.environ["GOOGLE_API_KEY"] = manual_key.strip()

    selected_model = st.selectbox(
        "Select Gemini Model",
        ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview"],
        help="Gemini model used for multi-pathway retrosynthesis and green-swap evaluation.",
        key="gemini_model_select",
    )
    mechanism_filter = st.selectbox(
        "Green Swap Mechanism Filter",
        ["Auto"] + CANONICAL_MECHANISMS,
        key="green_swap_mechanism_filter",
    )
    fast_mode = st.checkbox(
        "⚡ Fast Mode (Skip GIFs & Wiki)",
        value=True,
        help="Skips slower Wikipedia lookups and multi-frame GIF rendering for rapid testing.",
        key="fast_mode_checkbox",
    )

    if st.button("🧹 Clear Cache & Saved Data", key="btn_clear_cache"):
        st.cache_data.clear()
        st.session_state["saved_molecules"] = {}
        st.success("Cache and saved molecule session data cleared.")
        st.rerun()

    if st.session_state["saved_molecules"]:
        st.markdown("---")
        st.subheader("💾 Saved Molecules")
        saved_keys = list(st.session_state["saved_molecules"].keys())
        picked_saved = st.selectbox(
            "Load previously analyzed molecule",
            options=[""] + saved_keys,
            format_func=lambda k: "— Select saved molecule —" if k == "" else k.title(),
            key="saved_mol_picker",
        )
        if picked_saved:
            st.session_state["active_mol_query"] = st.session_state["saved_molecules"][picked_saved]["query"]

    if st.session_state["username"] == ADMIN_USERNAME:
        st.markdown("---")
        with st.expander("📊 Admin Panel: User Database"):
            st.write("☁️ **Connected to Supabase Cloud**")
            try:
                response = supabase.table("users").select("username, id").execute()
                registered_users = response.data
                if registered_users:
                    st.success(f"Total Permanent Accounts: {len(registered_users)}")
                    for idx, user in enumerate(registered_users, 1):
                        st.text(f"{idx}. {user['username']} (ID: {user['id']})")
                else:
                    st.info("0 users are registered yet.")
            except Exception as db_err:
                st.error(f"Database error: {db_err}")

col1, col2 = st.columns([3, 1])
with col1:
    target_input = st.text_input(
        "Enter Drug / Organic Molecule Name:",
        value=st.session_state["active_mol_query"],
        placeholder="e.g., Ibuprofen, Aspirin, Paracetamol",
        key="target_mol_input",
    )
with col2:
    st.write(" ")
    st.write(" ")
    analyze_btn = st.button("🚀 Analyze & Save", key="btn_analyze_save")

if analyze_btn:
    mol_name = target_input.strip()
    if not mol_name:
        st.warning("Please enter a molecule name.")
        st.stop()

    st.session_state["active_mol_query"] = mol_name
    selected_rxns = None if mechanism_filter == "Auto" else [mechanism_filter]

    with st.spinner(f"🔬 Analyzing and caching '{mol_name}'..."):
        result = build_and_store_molecule_analysis(
            mol_name=mol_name,
            model_name=selected_model,
            selected_rxns=selected_rxns,
            include_wiki=not fast_mode,
        )

    if "error" in result:
        st.error(result["error"])
        st.stop()

    comp_data = result["comp_data"]
    smiles = result["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    profile = result["profile"]
    c_x, c_y, t_x, t_y, p_x, p_y, ph_x, ph_y = result["curves"]

    st.success(f"Saved '{mol_name}' in session & cache (source: `{result['source']}`).")
    st.subheader(f"Molecule Identified: {comp_data.get('iupac_name') or mol_name}")
    st.code(f"SMILES: {smiles}")
    st.markdown(
        f"**RDKit Profile:** MW = `{profile['mol_wt']}` g/mol | cLogP = `{profile['clogp']}` | "
        f"TPSA = `{profile['tpsa']}` Å² | Motifs = `{', '.join(profile['functional_groups']) or 'None flagged'}`"
    )

    c_2d, c_3d = st.columns(2)
    with c_2d:
        st.image(Draw.MolToImage(mol, size=(300, 300)), caption="2D Structure (RDKit)", width=260)
    with c_3d:
        if HAS_PY3DMOL:
            sdf_data = cached_sdf_data(comp_data.get("cid"))
            if sdf_data:
                viewer = py3Dmol.view(width=420, height=280)
                viewer.addModel(sdf_data, "sdf")
                viewer.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.28}})
                viewer.setBackgroundColor("#0E1117")
                viewer.zoomTo()
                st_html(viewer._make_html(), height=290)
            else:
                st.info("3D conformer unavailable for this entry.")
        else:
            st.info("Install `py3Dmol` for an interactive 3D conformer viewer.")

    st.markdown("---")
    st.header("📈 Solubility & Thermodynamic Curves")
    g1, g2 = st.columns(2)
    with g1:
        f1 = go.Figure(go.Scatter(x=c_x, y=c_y, mode="lines", line=dict(color="#00FFCC", width=3)))
        f1.update_layout(title="Solubility vs Concentration", xaxis_title="Conc (mg/mL)", yaxis_title="Solubility (mg/mL)", template="plotly_dark", height=320)
        st.plotly_chart(f1, use_container_width=True)

        f3 = go.Figure(go.Scatter(x=p_x, y=p_y, mode="lines", line=dict(color="#FFFF00", width=3)))
        f3.update_layout(title="Solubility vs Pressure", xaxis_title="Pressure (atm)", yaxis_title="Solubility (mol/L)", template="plotly_dark", height=320)
        st.plotly_chart(f3, use_container_width=True)
    with g2:
        f2 = go.Figure(go.Scatter(x=t_x, y=t_y, mode="lines", line=dict(color="#FF00CC", width=3)))
        f2.update_layout(title="Solubility vs Temp (Van't Hoff)", xaxis_title="Temp (K)", yaxis_title="Solubility (mol/L)", template="plotly_dark", height=320)
        st.plotly_chart(f2, use_container_width=True)

        f4 = go.Figure(go.Scatter(x=ph_x, y=ph_y, mode="lines", line=dict(color="#33CCFF", width=3)))
        f4.update_layout(title="Solubility vs pH (Henderson–Hasselbalch)", xaxis_title="pH", yaxis_title="Solubility (mol/L)", yaxis_type="log", template="plotly_dark", height=320)
        st.plotly_chart(f4, use_container_width=True)

    st.markdown(result["aquatic_md"])

    st.markdown("---")
    st.header("🧬 Multi-Pathway Retrosynthesis & Green Swap Ranking")
    if result["report_text"]:
        st.markdown(result["report_text"])
    st.markdown(result["green_md"])

    if not fast_mode and result["pathways"]:
        st.header("🎬 Animated Disconnection Simulations")
        for path_idx, pathway in enumerate(result["pathways"], 1):
            st.subheader(f"📌 {pathway.get('pathway_name', f'Pathway {path_idx}')}")
            steps = pathway.get("steps", [])
            if steps:
                cols = st.columns(len(steps))
                for idx, step_info in enumerate(steps):
                    with cols[idx % len(cols)]:
                        st.write(f"**Step {step_info.get('step', idx + 1)}:** {step_info.get('title', '')}")
                        t_smi = step_info.get("target_smiles", "")
                        s_smi = step_info.get("synthon_smiles", "")
                        if t_smi and s_smi:
                            gif_path = f"temp_path{path_idx}_step{idx}.gif"
                            out_gif = generate_disconnection_gif(t_smi, s_smi, gif_path)
                            if out_gif:
                                st.image(out_gif)

    full_export = "\n\n".join(
        part for part in [result["aquatic_md"], result["report_text"], result["green_md"]] if part
    )
    safe_name = "".join([c for c in mol_name if c.isalnum() or c == " "]).strip().replace(" ", "_")
    st.download_button(
        label="📥 Download Full Report (.md)",
        data=full_export,
        file_name=f"{safe_name}_Solubility_and_Green_Retrosynthesis.md",
        mime="text/markdown",
        key="btn_download_report",
    )
