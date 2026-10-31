from pathlib import Path
from datetime import date
from collections import Counter

from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse, FileResponse, Http404
from django.urls import reverse

import pandas as pd
import numpy as np
import joblib
import vcf
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

# ================= REPORTLAB =================
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)

from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.colors import HexColor


# ======================================================
# HOME
# ======================================================
def home(request):
    return render(request, "home.html")


# ======================================================
# MODEL PATHS  (✅ FIXED)
# ======================================================
BASE_ML = Path(settings.BASE_DIR) / "app1" / "ML"

MODEL_PATH = BASE_ML / "hybrid_quantum_genomic_model.pth"
SCALER_PATH = BASE_ML / "scaler.pkl"
LOOKUP_PATH = BASE_ML / "lookup.pkl"
REVIEW_MAP_PATH = BASE_ML / "review_map.pkl"


# ======================================================
# LOAD ML FILES
# ======================================================
scaler = joblib.load(SCALER_PATH)
lookup = joblib.load(LOOKUP_PATH)
review_map = joblib.load(REVIEW_MAP_PATH)


# ---------------- Quantum Model ----------------
n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


qlayer = qml.qnn.TorchLayer(
    circuit,
    {"weights": (2, n_qubits, 3)}
)

model = nn.Sequential(
    nn.Linear(5, 5),
    qlayer,
    nn.Linear(5, 5)
)

# ✅ CORRECT PATH LOAD
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


# ======================================================
# CONSTANTS
# ======================================================
DISEASES = [
    "Hereditary Cancer Syndrome",
    "Genetic Cardiovascular Disorder",
    "Genetic Diabetes Disorder",
    "Hereditary Neurological Disorder",
    "Other Genetic Disorder"
]

PDF_DIR = Path(settings.MEDIA_ROOT) / "genomic_reports"
PDF_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================
# HELPERS
# ======================================================
def normalize_chromosome(chrom):
    return str(chrom).upper().replace("CHR", "")


def read_vcf(path):
    variants = []
    reader = vcf.Reader(open(path, "r"))

    for rec in reader:
        variants.append({
            "chromosome": normalize_chromosome(rec.CHROM),
            "position": int(rec.POS)
        })

    return variants


def lookup_variant(chrom, pos):
    try:
        row = lookup.loc[(chrom, pos)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row
    except KeyError:
        return None


def build_features(row):

    review_score = review_map.get(str(row["ReviewStatus"]), 0.3)
    submitters = float(row["NumberSubmitters"])

    evidence_strength = (
        0.5 * review_score +
        0.3 * min(submitters, 10) / 10 +
        0.2 * int(pd.notna(row["Guidelines"]))
    )

    return np.array([[

        float(row["ClinSigSimple"]),
        evidence_strength,
        1,
        submitters,
        review_score
    ]])


def predict_qml_scores(features):

    X_scaled = scaler.transform(features)

    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32))
        probs = F.softmax(logits, dim=1).numpy()[0]

    return probs


def categorize_variant(row):

    severity = float(row["ClinSigSimple"])

    review_score = review_map.get(str(row["ReviewStatus"]), 0.3)
    submitters = float(row["NumberSubmitters"])

    evidence_strength = (
        0.5 * review_score +
        0.3 * min(submitters, 10) / 10 +
        0.2 * int(pd.notna(row["Guidelines"]))
    )

    risk_score = 0.6 * severity + 0.4 * evidence_strength

    if risk_score >= 0.75:
        category = "HIGH RISK"
    elif risk_score >= 0.40:
        category = "MODERATE RISK"
    else:
        category = "LOW RISK"

    return risk_score, category










# ======================================================
# MUTATION ANALYSIS
# ======================================================

def mutation_analysis(row, chrom, pos, quantum_prob):

    severity = float(row["ClinSigSimple"])

    if severity >= 0.8:
        classification = "Pathogenic"
    elif severity >= 0.6:
        classification = "Likely Pathogenic"
    elif severity >= 0.3:
        classification = "Uncertain Significance"
    else:
        classification = "Likely Benign"

    mutation_detected = "YES" if severity >= 0.3 else "NO"

    return {
        "Gene": row["GeneSymbol"],
        "Detected": mutation_detected,
        "Type": "Variant Severity Based Mutation",
        "Position": f"chr{chrom}:{pos}",
        "Classification": classification,
        "QuantumProb": quantum_prob
    }


# ==========================================
# FULL GENOMIC PDF REPORT
# ==========================================

def generate_pdf_report(
    patient_id,
    scores,
    diseases,
    matched_variants,
    mutation_inputs,
    quantum_probability
):

    pdf_path = PDF_DIR / f"{patient_id}_Genomic_Report.pdf"

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    elements = []

    title_style = ParagraphStyle(
        "Title",
        fontSize=20,
        alignment=1,
        textColor=HexColor("#1F3C88"),
        spaceAfter=14
    )

    section_style = ParagraphStyle(
        "Section",
        fontSize=14,
        fontName="Helvetica-Bold",
        textColor=HexColor("#2B4DA2"),
        spaceBefore=12,
        spaceAfter=6
    )

    normal_style = ParagraphStyle(
        "Normal",
        fontSize=11,
        spaceAfter=4
    )

    highlight_style = ParagraphStyle(
        "Highlight",
        fontSize=11,
        backColor=HexColor("#FFF2CC")
    )

    # ---------- TITLE ----------
    elements.append(Paragraph("Genomic Diagnostic Report", title_style))

    info_table = Table([
        ["Patient ID:", patient_id],
        ["Test Type:", "Hybrid Quantum Genomic Classification"],
        ["Report Date:", date.today().isoformat()]
    ], colWidths=[2*inch, 4*inch])

    elements.append(info_table)
    elements.append(Spacer(1,12))
    elements.append(HRFlowable(width="100%"))

    # ---------- Prediction ----------
    elements.append(Paragraph("Prediction Results", section_style))

    data = [["Disease", "Probability"]]
    for d, s in zip(diseases, scores):
        data.append([d, f"{s:.3f}"])

    table = Table(data, colWidths=[4*inch,2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey)
    ]))

    elements.append(table)

    # ---------- Variants ----------
    elements.append(Paragraph("Categorized Variants", section_style))

    vdata = [["Gene","Chr","Position","Score","Category"]]

    for v in matched_variants:
        vdata.append([
            v["Gene"],
            v["Chromosome"],
            v["Position"],
            f"{v['Score']:.3f}",
            v["Category"]
        ])

    vtable = Table(vdata, repeatRows=1)
    vtable.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),0.3,colors.grey)
    ]))

    elements.append(vtable)

    # ---------- Mutation Analysis ----------
    elements.append(Paragraph("Mutation Analysis", section_style))

    for row, chrom, pos in mutation_inputs:

        severity = float(row["ClinSigSimple"])
        if severity < 0.6:
            continue

        m = mutation_analysis(row, chrom, pos, quantum_probability)

        elements.append(Paragraph(f"<b>Gene:</b> {m['Gene']}", normal_style))
        elements.append(Paragraph(f"<b>Position:</b> {m['Position']}", normal_style))
        elements.append(Paragraph(
            f"<b>Classification:</b> {m['Classification']}",
            highlight_style
        ))
        elements.append(Paragraph(
            f"<b>Quantum Probability:</b> {m['QuantumProb']:.3f}",
            normal_style
        ))

        elements.append(Spacer(1,8))

    # ---------- Risk Summary ----------
    counts = Counter([v["Category"] for v in matched_variants])

    elements.append(Paragraph("Risk Summary", section_style))

    for k, v in counts.items():
        elements.append(Paragraph(f"{k} : {v}", normal_style))

    elements.append(Spacer(1,10))

    elements.append(Paragraph(
        "Disclaimer: Research-use genomic prediction only.",
        normal_style
    ))

    doc.build(elements)

    return pdf_path


# ======================================================
# ANALYSIS VIEW (FINAL)
# ======================================================
def analysis(request):

    if request.method == "POST":

        action = request.POST.get("action")
        patient_id = request.POST.get("pid")
        vcf_file = request.FILES.get("vcf")

        if not vcf_file:
            return JsonResponse({"success": False, "error": "VCF required"})

        temp_path = Path(settings.MEDIA_ROOT) / vcf_file.name

        with open(temp_path, "wb+") as f:
            for chunk in vcf_file.chunks():
                f.write(chunk)

        variants = read_vcf(temp_path)

        feature_list = []
        matched_variants = []
        mutation_inputs = []

        for v in variants:

            chrom = v["chromosome"]
            pos = v["position"]

            row = lookup_variant(chrom, pos)
            if row is None:
                continue

            risk_score, category = categorize_variant(row)

            matched_variants.append({
                "Gene": row["GeneSymbol"],
                "Chromosome": chrom,
                "Position": pos,
                "Score": risk_score,
                "Category": category
            })

            mutation_inputs.append((row, chrom, pos))
            feature_list.append(build_features(row)[0])

        if not feature_list:
            return JsonResponse({"success": False, "error": "No variants found"})

        if action == "analyze":

            high_risk = [
                {
                    "gene": v["Gene"],
                    "chromosome": v["Chromosome"],
                    "position": v["Position"]
                }
                for v in matched_variants
                if v["Category"] == "HIGH RISK"
            ]

            return JsonResponse({
                "success": True,
                "stage": "high_risk",
                "high_risk": high_risk
            })

        patient_features = np.mean(np.array(feature_list), axis=0).reshape(1, -1)

        scores = predict_qml_scores(patient_features)
        quantum_probability = float(np.max(scores))

        pdf_path = generate_pdf_report(
            patient_id,
            scores,
            DISEASES,
            matched_variants,
            mutation_inputs,
            quantum_probability
        )

        return JsonResponse({
            "success": True,
            "stage": "report",
            "scores": scores.tolist(),
            "diseases": DISEASES,
            "file_url": reverse("download_pdf", args=[pdf_path.name])
        })

    return render(request, "analysis.html")


# ======================================================
# DOWNLOAD PDF
# ======================================================
def download_pdf(request, filename):

    pdf_path = PDF_DIR / filename

    if not pdf_path.exists():
        raise Http404("PDF not found")

    return FileResponse(open(pdf_path, "rb"),
                        as_attachment=True,
                        filename=filename)