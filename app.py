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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
from supabase import Client, create_client

try:
    import py3Dmol
    from streamlit.components.v1 import html as st_html

    HAS_PY3DMOL = True
except ImportError:
    HAS_PY3DMOL = False

# ------------------------------------------------------------------------------
# 1. PAGE CONFIG & STYLES
# ------------------------------------------------------------------------------
ADMIN_USERNAME = "admin"

st.set_page_config(
    page_title="AI Retrosynthetic & Solubility Workbench",
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
        margin-bottom: 16px;
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
# 2. SUPABASE AUTHENTICATION
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
        res = supabase.table("users").select("password").eq("username", username).execute()
        return bool(res.data and res.data[0]["password"] == make_hash(password))
    except Exception as e:
        st.error(f"Login error: {e}")
        return False


def add_user(username: str, password: str) -> bool:
    try:
        existing = supabase.table("users").select("username").eq("username", username).execute()
        if existing.data:
            return False
        supabase.table("users").insert({"username": username, "password": make_hash(password)}).execute()
        return True
    except Exception as e:
        st.error(f"Registration error: {e}")
        return False


for k, v in {
    "logged_in": False,
    "username": "",
    "saved_molecules": {},
    "active_mol_query": "Paracetamol",
    "last_record": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------------------------------------------------------
# 3. AUTHENTICATION GATE
# ------------------------------------------------------------------------------
if not st.session_state["logged_in"]:
    st.markdown("<h1 class='main-title'>🔐 AI Retrosynthetic Analyst Portal</h1>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])
        with tab_login:
            u = st.text_input("Username", key="login_user")
            p = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", key="btn_login"):
                if check_user(u, p):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = u
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        with tab_signup:
            nu = st.text_input("Choose Username", key="signup_user")
            np_ = st.text_input("Choose Password", type="password", key="signup_pass")
            if st.button("Register", key="btn_signup"):
                if nu.strip() and np_.strip():
                    if add_user(nu.strip(), np_.strip()):
                        st.success("Account created! Switch to Login.")
                    else:
                        st.error("Username already exists.")
                else:
                    st.warning("Please fill in both fields.")
    st.stop()

# ------------------------------------------------------------------------------
# 4. GREEN CHEMISTRY RUBRIC & OFFLINE SWAPS
# ------------------------------------------------------------------------------
CANONICAL_MECHANISMS = [
    "Esterification",
    "Amidation",
    "Oxidation",
    "Reduction",
    "Suzuki Coupling",
    "Buchwald-Hartwig",
]

SCORE_CAPS = {
    "catalysis_atom": 30.0,
    "hazard_guide": 25.0,
    "mildness": 20.0,
    "selectivity": 15.0,
    "precedent": 10.0,
}

OFFLINE_GREEN_SWAPS: Dict[str, Dict[str, Any]] = {
    "esterification": {
        "mechanism_type": "Esterification",
        "mechanism_rationale": "Catalytic acid or lipase/transesterification in greener media (2-MeTHF, EtOAc, alcohols) replacing stoichiometric coupling agents.",
        "candidates": [
            {
                "name": "Heterogeneous / Catalytic Acid in EtOAc or 2-MeTHF",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Acid-catalyzed acyl-oxygen esterification with water removal",
                "reagents_conditions": "Amberlyst-15 or H2SO4 (2 mol%), 2-MeTHF or EtOAc, 50–80 °C",
                "solvent_tier": "CHEM21 Recommended (2-MeTHF / EtOAc)",
                "scores": {"catalysis_atom": 24.0, "hazard_guide": 20.0, "mildness": 14.0, "selectivity": 11.0, "precedent": 9.0},
                "source_ref": "ACS GCI & CHEM21 solvent guides",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Check acid-labile groups and steric hindrance.",
            },
            {
                "name": "Lipase-Catalyzed Esterification / Transesterification",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Enzymatic acyl transfer under mild conditions",
                "reagents_conditions": "Immobilized Lipase B, t-BuOH or 2-MeTHF, 35–50 °C",
                "solvent_tier": "CHEM21 Recommended / Mild bio-media",
                "scores": {"catalysis_atom": 26.0, "hazard_guide": 22.0, "mildness": 18.0, "selectivity": 14.0, "precedent": 7.0},
                "source_ref": "ACS GCI biocatalysis benchmarks",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "Medium",
                "caveat": "Substrate scope and scale cost require check.",
            },
        ],
    },
    "amidation": {
        "mechanism_type": "Amidation",
        "mechanism_rationale": "Lower-waste coupling (T3P) or catalytic boron/ester aminolysis in greener solvents instead of DMF/HATU.",
        "candidates": [
            {
                "name": "T3P Coupling in EtOAc or 2-MeTHF",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Cyclic anhydride carboxylate activation with water-soluble byproduct",
                "reagents_conditions": "T3P (50% in EtOAc/2-MeTHF), base, 0–25 °C",
                "solvent_tier": "ACS GCI Preferred / CHEM21 Recommended",
                "scores": {"catalysis_atom": 20.0, "hazard_guide": 20.0, "mildness": 17.0, "selectivity": 12.0, "precedent": 9.0},
                "source_ref": "ACS GCI reagent guide",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Monitor alpha-chiral centers for epimerization.",
            },
            {
                "name": "Boronic Acid Direct Amidation",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Catalytic boron activation of carboxylic acids",
                "reagents_conditions": "2–5 mol% arylboronic acid, molecular sieves, 2-MeTHF, 60–80 °C",
                "solvent_tier": "CHEM21 Recommended (2-MeTHF)",
                "scores": {"catalysis_atom": 25.0, "hazard_guide": 18.0, "mildness": 14.0, "selectivity": 11.0, "precedent": 6.0},
                "source_ref": "Catalytic direct amidation literature",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "Medium",
                "caveat": "Less suited for weakly nucleophilic anilines.",
            },
        ],
    },
    "oxidation": {
        "mechanism_type": "Oxidation",
        "mechanism_rationale": "Catalytic O2/TEMPO or NaOCl/TEMPO/H2O2 pathways replacing stoichiometric Cr(VI)/MnO2 and chlorinated solvents.",
        "candidates": [
            {
                "name": "Cu / TEMPO Aerobic Oxidation",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Catalytic aerobic alcohol oxidation with water byproduct",
                "reagents_conditions": "1–5 mol% Cu/ligand + TEMPO, air/O2, 2-MeTHF or MeCN, RT",
                "solvent_tier": "CHEM21 Recommended (2-MeTHF) + O2",
                "scores": {"catalysis_atom": 28.0, "hazard_guide": 21.0, "mildness": 17.0, "selectivity": 12.0, "precedent": 9.0},
                "source_ref": "ACS GCI aerobic oxidation precedent",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Check solvent flammability with O2/air.",
            },
            {
                "name": "TEMPO / NaOCl in EtOAc",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Oxoammonium-mediated selective oxidation",
                "reagents_conditions": "1 mol% TEMPO, KBr/NaHCO3, aq. NaOCl, EtOAc, 0–15 °C",
                "solvent_tier": "CHEM21 Recommended (EtOAc / H2O)",
                "scores": {"catalysis_atom": 23.0, "hazard_guide": 18.0, "mildness": 17.0, "selectivity": 12.0, "precedent": 9.0},
                "source_ref": "Anelli oxidation & green solvent swap",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Control pH and temperature to prevent over-oxidation.",
            },
        ],
    },
    "reduction": {
        "mechanism_type": "Reduction",
        "mechanism_rationale": "Transfer hydrogenation, NaBH4 in alcohols, or mild iron reduction avoiding LiAlH4 and toxic aprotic solvents.",
        "candidates": [
            {
                "name": "Catalytic Transfer Hydrogenation in EtOH / i-PrOH",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Hydrogen transfer from alcohol/formate donor without H2 cylinder",
                "reagents_conditions": "Pd/C or Ru/Ir + formate/i-PrOH, EtOH or 2-MeTHF, 40–75 °C",
                "solvent_tier": "CHEM21 Recommended (EtOH, i-PrOH, 2-MeTHF)",
                "scores": {"catalysis_atom": 24.0, "hazard_guide": 21.0, "mildness": 16.0, "selectivity": 11.0, "precedent": 9.0},
                "source_ref": "ACS GCI reduction & CHEM21 guide",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Watch for competing alkene or aryl halide reduction.",
            },
            {
                "name": "NaBH4 in EtOH / Aqueous Alcohol",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Mild carbonyl hydride reduction",
                "reagents_conditions": "NaBH4, EtOH or EtOH/H2O, 0–25 °C",
                "solvent_tier": "CHEM21 Recommended (EtOH / H2O)",
                "scores": {"catalysis_atom": 18.0, "hazard_guide": 19.0, "mildness": 18.0, "selectivity": 12.0, "precedent": 9.0},
                "source_ref": "Standard carbonyl reduction precedent",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Does not reduce esters or amides directly.",
            },
        ],
    },
    "suzuki coupling": {
        "mechanism_type": "Suzuki Coupling",
        "mechanism_rationale": "Pd-catalyzed biaryl coupling in aqueous EtOH or 2-MeTHF instead of DMF/toluene.",
        "candidates": [
            {
                "name": "Aqueous EtOH / 2-MeTHF Suzuki-Miyaura",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Low-loading Pd cross-coupling with arylboronic acids",
                "reagents_conditions": "0.5–2 mol% Pd cat., K2CO3/K3PO4, EtOH/H2O or 2-MeTHF/H2O, 50–80 °C",
                "solvent_tier": "CHEM21 Recommended (EtOH, 2-MeTHF, H2O)",
                "scores": {"catalysis_atom": 24.0, "hazard_guide": 21.0, "mildness": 16.0, "selectivity": 12.0, "precedent": 9.0},
                "source_ref": "ACS GCI Suzuki solvent guide",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Ensure adequate substrate solubility in aqueous-organic mix.",
            },
            {
                "name": "Micellar Aqueous Suzuki Coupling",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Surfactant-enabled coupling in water",
                "reagents_conditions": "1–2 mol% Pd/ligand, ~2% aq. surfactant, base, RT–45 °C",
                "solvent_tier": "Water-dominant medium",
                "scores": {"catalysis_atom": 25.0, "hazard_guide": 22.0, "mildness": 16.0, "selectivity": 10.0, "precedent": 6.0},
                "source_ref": "Lipshutz micellar catalysis literature",
                "evidence_level": "HEURISTIC",
                "confidence": "Medium",
                "caveat": "Verify emulsion workup and substrate compatibility.",
            },
        ],
    },
    "buchwald-hartwig": {
        "mechanism_type": "Buchwald-Hartwig",
        "mechanism_rationale": "Pd or Cu C–N coupling in 2-MeTHF or aqueous/micellar media replacing DMF/dioxane.",
        "candidates": [
            {
                "name": "Pd / Ligand C–N Coupling in 2-MeTHF",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Catalytic aryl/heteroaryl C–N bond formation",
                "reagents_conditions": "0.5–2 mol% Pd + ligand, K3PO4 or Cs2CO3, 2-MeTHF, 60–90 °C",
                "solvent_tier": "CHEM21 Recommended (2-MeTHF)",
                "scores": {"catalysis_atom": 23.0, "hazard_guide": 19.0, "mildness": 15.0, "selectivity": 12.0, "precedent": 9.0},
                "source_ref": "Buchwald-Hartwig & CHEM21 ether guide",
                "evidence_level": "LITERATURE-SUPPORTED",
                "confidence": "High",
                "caveat": "Avoid harsh alkoxides if base-sensitive groups are present.",
            },
            {
                "name": "Cu-Mediated C–N Coupling in EtOH / 2-MeTHF",
                "origin": "OFFLINE_GREEN_SWAPS",
                "mechanism_role": "Lower-cost copper alternative for aryl C–N coupling",
                "reagents_conditions": "2–10 mol% Cu salt + ligand, K2CO3/K3PO4, 2-MeTHF or EtOH, 70–100 °C",
                "solvent_tier": "CHEM21 Recommended (2-MeTHF / EtOH)",
                "scores": {"catalysis_atom": 20.0, "hazard_guide": 18.0, "mildness": 12.0, "selectivity": 10.0, "precedent": 7.0},
                "source_ref": "Copper C–N coupling literature",
                "evidence_level": "HEURISTIC",
                "confidence": "Low",
                "caveat": "Often requires higher temperature/loading than Pd.",
            },
        ],
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

    def __post_init__(self):
        self.scores = clamp_scores(self.scores)

    @property
    def total_score(self) -> float:
        return round(sum(self.scores.values()), 1)

    @property
    def priority_tier(self) -> str:
        s = self.total_score
        if s >= 75.0 and self.evidence_level == "LITERATURE-SUPPORTED" and self.confidence in ("High", "Medium"):
            return "Priority 1"
        if s >= 55.0:
            return "Priority 2"
        return "Priority 3"


def normalize_reaction_key(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(k in t for k in ["buchwald", "hartwig", "c-n", "aminat"]):
        return "buchwald-hartwig"
    if any(k in t for k in ["reduct", "hydrogenat", "nitro", "carbonyl"]):
        return "reduction"
    if any(k in t for k in ["suzuki", "cross-coupling", "c-c", "biaryl"]):
        return "suzuki coupling"
    if any(k in t for k in ["esterif", "ester"]):
        return "esterification"
    if any(k in t for k in ["amidat", "amide"]):
        return "amidation"
    if any(k in t for k in ["oxidat", "oxidize"]):
        return "oxidation"
    return None


def get_rdkit_profile(mol) -> Dict[str, Any]:
    smarts_map = {
        "ester": "[CX3](=O)[OX2][#6]",
        "amide": "[CX3](=O)[NX3]",
        "carbonyl": "[CX3]=[OX1]",
        "nitro": "[NX3](=[OX1])[OX1-,OX2]",
        "alkene": "[CX3]=[CX3]",
        "aryl_halide": "[c][Cl,Br,I]",
        "primary_oh": "[CH2X4][OX2H1]",
    }
    detected = [
        name
        for name, smarts in smarts_map.items()
        if (pat := Chem.MolFromSmarts(smarts)) is not None and mol.HasSubstructMatch(pat)
    ]
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mol_wt": round(Descriptors.MolWt(mol), 2),
        "clogp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "functional_groups": detected,
    }


def get_offline_green_steps(mol_name: str, profile: Dict[str, Any], selected_rxns: Optional[List[str]] = None) -> List[dict]:
    keys: List[str] = []
    if selected_rxns:
        for r in selected_rxns:
            k = normalize_reaction_key(r)
            if k and k not in keys:
                keys.append(k)
    if not keys:
        fgs = set(profile.get("functional_groups", []))
        if "amide" in fgs:
            keys.append("amidation")
        if "ester" in fgs:
            keys.append("esterification")
        if "primary_oh" in fgs:
            keys.append("oxidation")
        if "nitro" in fgs or "carbonyl" in fgs:
            keys.append("reduction")
        if "aryl_halide" in fgs:
            keys.append("suzuki coupling")
        if not keys:
            keys = ["amidation", "esterification", "oxidation"]

    return [
        {
            "step_name": f"{OFFLINE_GREEN_SWAPS[k]['mechanism_type']} Swap ({mol_name})",
            "mechanism_type": OFFLINE_GREEN_SWAPS[k]["mechanism_type"],
            "mechanism_rationale": OFFLINE_GREEN_SWAPS[k]["mechanism_rationale"],
            "green_candidates": OFFLINE_GREEN_SWAPS[k]["candidates"],
        }
        for k in keys
    ]


def blend_step_with_offline_swaps(step: dict) -> dict:
    raw_mech = step.get("mechanism_type", "")
    key = normalize_reaction_key(raw_mech)
    off = OFFLINE_GREEN_SWAPS.get(key) if key else None
    merged: List[dict] = []
    seen = set()

    for c in step.get("green_candidates", []):
        nm = re.sub(r"\s+", " ", str(c.get("name", "")).strip().lower())
        if nm and nm not in seen:
            seen.add(nm)
            item = dict(c)
            item["origin"] = item.get("origin") or "LLM"
            merged.append(item)

    if off:
        for c in off.get("candidates", []):
            nm = re.sub(r"\s+", " ", str(c.get("name", "")).strip().lower())
            if nm and nm not in seen:
                seen.add(nm)
                item = dict(c)
                item["origin"] = "OFFLINE_GREEN_SWAPS"
                merged.append(item)

    return {
        **step,
        "mechanism_type": off["mechanism_type"] if off else (raw_mech or "General Transformation"),
        "mechanism_rationale": step.get("mechanism_rationale") or (off["mechanism_rationale"] if off else ""),
        "green_candidates": merged[:4],
    }


def format_green_ranking(steps_data: List[dict], profile: Dict[str, Any]) -> str:
    fg_set = set(profile.get("functional_groups", []))
    lines = ["### 🌱 Ranked Green Chemistry Swaps (0–100 Rubric)"]
    for idx, step in enumerate(steps_data, 1):
        mech = step.get("mechanism_type", "General")
        lines.append(f"\n#### {idx}. {step.get('step_name', mech)} (`{mech}`)")
        if step.get("mechanism_rationale"):
            lines.append(f"*{step['mechanism_rationale']}*")

        cands: List[GreenCandidate] = []
        for raw in step.get("green_candidates", []):
            sc = clamp_scores(raw.get("scores"))
            caveat = str(raw.get("caveat", "Validate on substrate."))
            if "reduct" in mech.lower() and "alkene" in fg_set and "h2" in str(raw.get("reagents_conditions", "")).lower():
                sc["selectivity"] = max(0.0, sc["selectivity"] - 3.0)
                caveat += " (Alkene present: check chemoselectivity.)"
            cands.append(
                GreenCandidate(
                    name=str(raw.get("name", "Alternative")),
                    origin=str(raw.get("origin", "LLM")),
                    mechanism_role=str(raw.get("mechanism_role", mech)),
                    reagents_conditions=str(raw.get("reagents_conditions", "")),
                    solvent_tier=str(raw.get("solvent_tier", "")),
                    scores=sc,
                    source_ref=str(raw.get("source_ref", "Green chemistry guide")),
                    evidence_level=str(raw.get("evidence_level", "HEURISTIC")).upper(),
                    confidence=str(raw.get("confidence", "Medium")).capitalize(),
                    caveat=caveat,
                )
            )
        cands.sort(key=lambda x: -x.total_score)
        for c in cands:
            lines.append(
                f"- **{c.priority_tier} — {c.name}** (`{c.total_score}/100`) · `{c.origin}` · **[{c.evidence_level} | {c.confidence}]**\n"
                f"  - *Conditions / Solvent:* {c.reagents_conditions} | *{c.solvent_tier}*\n"
                f"  - *Ref & Caveat:* {c.source_ref} — {c.caveat}"
            )
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# 5. FAST ORGANIC MOLECULE LOOKUP (ANY MOLECULE: NAME, SMILES, OR CID)
# ------------------------------------------------------------------------------
def resolve_rdkit_mol(query: str) -> Optional[Chem.Mol]:
    """Accepts any valid SMILES directly, or resolves name/CID via PubChem."""
    q = query.strip()
    if not q:
        return None

    # 1. Direct SMILES check (instant, works offline for any organic SMILES)
    mol = Chem.MolFromSmiles(q)
    if mol is not None and mol.GetNumAtoms() > 0:
        return mol

    # 2. PubChem lookup by name or CID
    compounds = []
    for _ in range(2):
        try:
            if q.isdigit():
                compounds = [pcp.Compound.from_cid(int(q))]
            else:
                compounds = pcp.get_compounds(q, "name") or []
            if compounds:
                break
        except Exception:
            time.sleep(0.35)

    for c in compounds:
        smi = (
            getattr(c, "smiles", None)
            or getattr(c, "isomeric_smiles", None)
            or getattr(c, "connectivity_smiles", None)
            or getattr(c, "canonical_smiles", None)
        )
        if smi:
            m = Chem.MolFromSmiles(smi)
            if m is not None and m.GetNumAtoms() > 0:
                m.SetProp("_cid", str(getattr(c, "cid", "") or ""))
                m.SetProp("_iupac", str(getattr(c, "iupac_name", "") or q))
                if getattr(c, "xlogp", None) is not None:
                    m.SetProp("_xlogp", str(c.xlogp))
                return m
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_molecule_core(query: str) -> Optional[Dict[str, Any]]:
    mol = resolve_rdkit_mol(query)
    if mol is None:
        return None
    can_smiles = Chem.MolToSmiles(mol)
    mw = round(Descriptors.MolWt(mol), 2)
    clogp = round(Descriptors.MolLogP(mol), 2)
    xlogp_prop = mol.GetProp("_xlogp") if mol.HasProp("_xlogp") else ""
    try:
        logp = float(xlogp_prop) if xlogp_prop else clogp
    except ValueError:
        logp = clogp

    cid_prop = mol.GetProp("_cid") if mol.HasProp("_cid") else ""
    iupac_prop = mol.GetProp("_iupac") if mol.HasProp("_iupac") else query.strip()
    profile = get_rdkit_profile(mol)

    return {
        "query": query.strip(),
        "name": iupac_prop or query.strip(),
        "cid": int(cid_prop) if cid_prop.isdigit() else None,
        "smiles": can_smiles,
        "mol_wt": mw,
        "logp": logp,
        "profile": profile,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def cached_sdf(cid: Optional[int]) -> Optional[str]:
    if not cid:
        return None
    for rec in ("3d", "2d"):
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF/?record_type={rec}"
            with urllib.request.urlopen(url, timeout=4) as r:
                return r.read().decode("utf-8")
        except Exception:
            continue
    return None


def generate_solubility_curves(mw: float, logp: float):
    conc = np.linspace(0, 100, 40)
    s_conc = (1 - np.exp(-0.1 * conc)) * (50.0 / (abs(logp) + 1.5))

    temps = np.linspace(273, 373, 40)
    s_temp = np.exp(-((25000.0 + mw * 10.0) / 8.314) * (1.0 / temps - 1.0 / 298.15)) * (10.0 ** (-logp))

    press = np.linspace(1, 100, 40)
    s_press = np.exp(((mw / 1.2) * (press - 1.0)) / (8.314 * 298.15)) * (10.0 ** (-logp))

    ph = np.linspace(2.0, 12.0, 60)
    s0 = max(1e-4, 0.05 * (10.0 ** (-0.5 * logp)))
    s_ph = s0 * (1.0 + 10.0 ** (ph - 4.5))
    return conc, s_conc, temps, s_temp, press, s_press, ph, s_ph


# ------------------------------------------------------------------------------
# 6. CONCISE RETROSYNTHESIS PROMPT + GUARANTEED PATHWAYS FALLBACK
# ------------------------------------------------------------------------------
def build_fallback_pathways(mol_name: str, smiles: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures retrosynthetic pathways always render even offline or if LLM omits JSON."""
    fgs = profile.get("functional_groups", [])
    motif_hint = ", ".join(fgs) if fgs else "core carbon/heteroatom framework"
    md = (
        f"### 🛣️ Retrosynthetic Pathways for **{mol_name}** (`{smiles}`)\n"
        f"*Detected motifs:* `{motif_hint}`\n\n"
        f"- **Pathway A (Classical / Industrial):** Disconnect terminal amide/ester/C–heteroatom bond to commercial aromatic/aliphatic building blocks; execute standard acylation/substitution with solvent recovery.\n"
        f"- **Pathway B (Green / Catalytic):** Use catalytic or milder reagents in 2-MeTHF, EtOAc, or aqueous/alcoholic media to lower E-factor and solvent hazard.\n"
        f"- **Pathway C (Convergent / FGI):** Assemble two pre-functionalized fragments first, then apply late-stage functional group interconversion (FGI)."
    )
    pathways = [
        {"pathway_name": "Pathway A: Classical Route", "steps": [{"step": 1, "title": "Key Bond Disconnection", "target_smiles": smiles, "synthon_smiles": smiles}]},
        {"pathway_name": "Pathway B: Green Catalytic Route", "steps": [{"step": 1, "title": "Green Solvent / Catalyst Swap", "target_smiles": smiles, "synthon_smiles": smiles}]},
        {"pathway_name": "Pathway C: Convergent / FGI Route", "steps": [{"step": 1, "title": "Fragment Assembly & FGI", "target_smiles": smiles, "synthon_smiles": smiles}]},
    ]
    return {"markdown": md, "pathways": pathways, "green_steps": []}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_concise_retrosynthesis(mol_name: str, smiles: str, profile_json: str, model_name: str) -> Dict[str, Any]:
    profile = json.loads(profile_json)
    fallback = build_fallback_pathways(mol_name, smiles, profile)
    if not os.environ.get("GOOGLE_API_KEY"):
        return fallback

    models_to_try = [model_name]
    if "gemini-3.5-flash-lite" not in models_to_try:
        models_to_try.append("gemini-3.5-flash-lite")

    prompt = PromptTemplate(
        input_variables=["molecule", "smiles", "profile", "mechs"],
        template=(
            "You are a concise synthetic chemist. Target: {molecule} | SMILES: {smiles} | Profile: {profile}.\n"
            "Write brief bullet points for:\n"
            "1. Pathway A (Classical/Industrial disconnections & forward steps)\n"
            "2. Pathway B (Green/mild disconnections & forward steps)\n"
            "3. Pathway C (Convergent/FGI disconnections & forward steps)\n"
            "4. Commercial starting materials\n"
            "Then end with ONE ```json object with keys:\n"
            "- \"pathways\": [{{\"pathway_name\": \"str\", \"steps\": [{{\"step\": 1, \"title\": \"str\", \"target_smiles\": \"str\", \"synthon_smiles\": \"str\"}}]}}]\n"
            "- \"green_steps\": [{{\"step_name\": \"str\", \"mechanism_type\": \"one of [{mechs}]\", \"mechanism_rationale\": \"str\", "
            "\"green_candidates\": [{{\"name\": \"str\", \"origin\": \"LLM\", \"mechanism_role\": \"str\", \"reagents_conditions\": \"str\", "
            "\"solvent_tier\": \"str\", \"scores\": {{\"catalysis_atom\": 24, \"hazard_guide\": 20, \"mildness\": 16, \"selectivity\": 12, \"precedent\": 8}}, "
            "\"source_ref\": \"str\", \"evidence_level\": \"LITERATURE-SUPPORTED\", \"confidence\": \"High\", \"caveat\": \"str\"}}]}}]\n"
            "Keep text concise for fast response."
        ),
    )

    last_err = None
    for m_name in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(model=m_name, temperature=0.15, max_retries=0)
            raw = (prompt | llm | StrOutputParser()).invoke(
                {
                    "molecule": mol_name,
                    "smiles": smiles,
                    "profile": json.dumps(profile),
                    "mechs": ", ".join(CANONICAL_MECHANISMS),
                }
            )
            text = raw if isinstance(raw, str) else "\n".join(str(x) for x in raw)
            pathways: List[dict] = []
            green_steps: List[dict] = []

            match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                clean_md = text.replace(match.group(0), "").strip()
                try:
                    obj = json.loads(match.group(1))
                    if isinstance(obj, dict):
                        pathways = obj.get("pathways") or []
                        green_steps = obj.get("green_steps") or []
                except Exception:
                    pass
            else:
                clean_md = text.strip()

            return {
                "markdown": clean_md or fallback["markdown"],
                "pathways": pathways or fallback["pathways"],
                "green_steps": green_steps,
            }
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                continue
            break

    note = (
        "ℹ️ *Daily free-tier Gemini quota reached (429).* Showing instant offline retrosynthesis & green-swap rankings.\n\n"
        if last_err and ("429" in str(last_err) or "RESOURCE_EXHAUSTED" in str(last_err))
        else f"⚠️ LLM fallback (`{last_err}`).\n\n"
    )
    return {
        "markdown": note + fallback["markdown"],
        "pathways": fallback["pathways"],
        "green_steps": [],
    }


def generate_disconnection_gif(target_smiles: str, synthon_smiles: str, out_path: str) -> Optional[str]:
    try:
        m1 = Chem.MolFromSmiles(target_smiles)
        m2 = Chem.MolFromSmiles(synthon_smiles)
        if not m1 or not m2:
            return None
        f1 = Draw.MolToImage(m1, size=(220, 220))
        f2 = Draw.MolToImage(m2, size=(220, 220))
        frames = [f1, f1, f2, f2]
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=450, loop=0)
        return out_path
    except Exception:
        return None


# ------------------------------------------------------------------------------
# 7. SIDEBAR & PROGRESSIVE STEP-BY-STEP EXECUTION
# ------------------------------------------------------------------------------
st.markdown("<h1 class='main-title'>🧪 AI Retrosynthetic & Solubility Workbench</h1>", unsafe_allow_html=True)

try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

with st.sidebar:
    st.write(f"👤 User: **{st.session_state['username']}**")
    if st.button("🚪 Logout", key="btn_logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.session_state["last_record"] = None
        st.rerun()

    st.markdown("---")
    if not os.environ.get("GOOGLE_API_KEY"):
        k_in = st.text_input("Gemini API Key", type="password", key="sb_gemini_key")
        if k_in.strip():
            os.environ["GOOGLE_API_KEY"] = k_in.strip()

    selected_model = st.selectbox(
        "Gemini Model",
        ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview"],
        key="sb_model_select",
    )
    mechanism_filter = st.selectbox(
        "Green Swap Focus",
        ["Auto"] + CANONICAL_MECHANISMS,
        key="sb_mech_filter",
    )
    run_gemini = st.checkbox("🤖 Run Gemini Pathways", value=True, key="sb_run_gemini")
    render_gifs = st.checkbox("🎬 Render Step GIFs", value=False, key="sb_render_gifs")

    if st.button("🧹 Clear Cache & Saved", key="btn_clear_cache"):
        st.cache_data.clear()
        st.session_state["saved_molecules"] = {}
        st.session_state["last_record"] = None
        st.rerun()

    if st.session_state["saved_molecules"]:
        st.markdown("---")
        saved_list = list(st.session_state["saved_molecules"].keys())
        chosen = st.selectbox(
            "💾 Saved Molecules",
            [""] + saved_list,
            format_func=lambda x: "— Pick saved —" if x == "" else x,
            key="sb_saved_picker",
        )
        if chosen and chosen in st.session_state["saved_molecules"]:
            st.session_state["last_record"] = st.session_state["saved_molecules"][chosen]

    if st.session_state["username"] == ADMIN_USERNAME:
        with st.expander("🛡️ Admin: Users"):
            try:
                users = supabase.table("users").select("username, id").execute().data or []
                for u_row in users:
                    st.text(f"• {u_row['username']} (#{u_row['id']})")
            except Exception as e:
                st.error(f"DB error: {e}")

c_inp, c_btn = st.columns([3, 1])
with c_inp:
    query_val = st.text_input(
        "Enter Any Organic Molecule Name or SMILES:",
        value=st.session_state["active_mol_query"],
        placeholder="e.g., Ibuprofen, salicylic acid, CC(=O)Nc1ccc(O)cc1, styrene",
        key="main_mol_input",
    )
with c_btn:
    st.write(" ")
    st.write(" ")
    go_btn = st.button("🚀 Run Step-by-Step", key="btn_run_steps")

if go_btn:
    q = query_val.strip()
    if not q:
        st.warning("Enter a molecule name or SMILES.")
        st.stop()

    st.session_state["active_mol_query"] = q
    progress = st.progress(0, text="Step 1/3: Resolving structure & RDKit properties...")

    core = cached_molecule_core(q)
    if not core:
        progress.empty()
        st.error("Could not resolve molecule from name or SMILES.")
        st.stop()

    progress.progress(35, text="Step 2/3: Ranking green chemistry swaps...")
    selected_rxns = None if mechanism_filter == "Auto" else [mechanism_filter]
    base_steps = get_offline_green_steps(core["name"], core["profile"], selected_rxns)

    progress.progress(70, text="Step 3/3: Building retrosynthetic pathways...")
    if run_gemini and os.environ.get("GOOGLE_API_KEY"):
        try:
            retro = cached_concise_retrosynthesis(
                core["name"],
                core["smiles"],
                json.dumps(core["profile"]),
                selected_model,
            )
        except Exception as e:
            fb = build_fallback_pathways(core["name"], core["smiles"], core["profile"])
            retro = {"markdown": f"⚠️ LLM fallback (`{e}`)\n\n{fb['markdown']}", "pathways": fb["pathways"], "green_steps": []}
    else:
        retro = build_fallback_pathways(core["name"], core["smiles"], core["profile"])

    blended_steps = [blend_step_with_offline_swaps(s) for s in (retro.get("green_steps") or base_steps)]
    green_md = format_green_ranking(blended_steps, core["profile"])
    curves = generate_solubility_curves(core["mol_wt"], core["logp"])

    record = {
        "core": core,
        "curves": curves,
        "green_md": green_md,
        "retro_md": retro["markdown"],
        "pathways": retro["pathways"],
    }
    st.session_state["saved_molecules"][q.lower()] = record
    st.session_state["last_record"] = record
    progress.progress(100, text="Done! Saved for instant reuse.")
    time.sleep(0.2)
    progress.empty()

rec = st.session_state.get("last_record")
if rec:
    core = rec["core"]
    prof = core["profile"]
    mol = Chem.MolFromSmiles(core["smiles"])

    # Stage 1 Output: Identity, Structure & Solubility
    st.subheader(f"🔬 {core['name']} (`{prof['formula']}`)")
    st.code(f"SMILES: {core['smiles']}")
    st.caption(
        f"MW: {core['mol_wt']} g/mol | Est. LogP: {core['logp']} | TPSA: {prof['tpsa']} Å² | "
        f"Motifs: {', '.join(prof['functional_groups']) or 'None'}"
    )

    s1, s2 = st.columns([1, 2])
    with s1:
        if mol:
            st.image(Draw.MolToImage(mol, size=(240, 240)), caption="2D Structure", width=220)
        if HAS_PY3DMOL and core.get("cid"):
            sdf = cached_sdf(core["cid"])
            if sdf:
                v = py3Dmol.view(width=320, height=220)
                v.addModel(sdf, "sdf")
                v.setStyle({"stick": {"radius": 0.15}})
                v.zoomTo()
                st_html(v._make_html(), height=230)

    with s2:
        conc, s_conc, temps, s_temp, press, s_press, ph, s_ph = rec["curves"]
        g1, g2 = st.columns(2)
        with g1:
            f1 = go.Figure(go.Scatter(x=conc, y=s_conc, mode="lines", line=dict(color="#00FFCC", width=2.5)))
            f1.update_layout(title="Solubility vs Conc", template="plotly_dark", height=230, margin=dict(l=25, r=15, t=35, b=25))
            st.plotly_chart(f1, use_container_width=True)

            f3 = go.Figure(go.Scatter(x=press, y=s_press, mode="lines", line=dict(color="#FFFF00", width=2.5)))
            f3.update_layout(title="Solubility vs Pressure", template="plotly_dark", height=230, margin=dict(l=25, r=15, t=35, b=25))
            st.plotly_chart(f3, use_container_width=True)
        with g2:
            f2 = go.Figure(go.Scatter(x=temps, y=s_temp, mode="lines", line=dict(color="#FF00CC", width=2.5)))
            f2.update_layout(title="Solubility vs Temp", template="plotly_dark", height=230, margin=dict(l=25, r=15, t=35, b=25))
            st.plotly_chart(f2, use_container_width=True)

            f4 = go.Figure(go.Scatter(x=ph, y=s_ph, mode="lines", line=dict(color="#33CCFF", width=2.5)))
            f4.update_layout(title="Solubility vs pH", yaxis_type="log", template="plotly_dark", height=230, margin=dict(l=25, r=15, t=35, b=25))
            st.plotly_chart(f4, use_container_width=True)

    # Stage 2 & Stage 3 Output in Tabs for clean step-by-step reading
    tab_retro, tab_green = st.tabs(["🛣️ Retrosynthetic Pathways", "🌱 Green Swaps"])
    with tab_retro:
        st.markdown(rec["retro_md"])
        if render_gifs and rec.get("pathways"):
            for p_idx, p_obj in enumerate(rec["pathways"], 1):
                st.markdown(f"**{p_obj.get('pathway_name', f'Pathway {p_idx}')}**")
                for s_idx, stp in enumerate(p_obj.get("steps") or [], 1):
                    gif = generate_disconnection_gif(
                        stp.get("target_smiles", ""),
                        stp.get("synthon_smiles", ""),
                        f"gif_p{p_idx}_s{s_idx}.gif",
                    )
                    if gif:
                        st.image(gif, caption=stp.get("title", f"Step {s_idx}"), width=180)

    with tab_green:
        st.markdown(rec["green_md"])
