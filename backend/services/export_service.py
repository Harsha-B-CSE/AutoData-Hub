import io
import json
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi.responses import FileResponse, Response
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfgen import canvas
from backend.db.session import SessionLocal
from backend.db.models import ProcessingJob, Dataset, QualityReport, BenchmarkLog

class ExportService:
    @staticmethod
    def generate_pdf_report(job_id: str) -> bytes:
        """
        Builds a publication-ready PDF summary of the dataset processing job,
        including initial vs final scorecards, metrics, runtime benchmarks, and recommendations.
        """
        db = SessionLocal()
        try:
            job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if not job:
                raise ValueError(f"Job {job_id} not found.")

            dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
            qr = db.query(QualityReport).filter(QualityReport.job_id == job_id).first()
            benchmarks = db.query(BenchmarkLog).filter(BenchmarkLog.job_id == job_id).all()

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=40,
                leftMargin=40,
                topMargin=40,
                bottomMargin=40
            )

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                "TitleStyle",
                parent=styles["Heading1"],
                fontSize=22,
                leading=26,
                textColor=colors.HexColor("#0f172a"),
                spaceAfter=10
            )
            subtitle_style = ParagraphStyle(
                "SubtitleStyle",
                parent=styles["Normal"],
                fontSize=11,
                textColor=colors.HexColor("#475569"),
                spaceAfter=20
            )
            heading2_style = ParagraphStyle(
                "Heading2Style",
                parent=styles["Heading2"],
                fontSize=14,
                leading=18,
                textColor=colors.HexColor("#1e293b"),
                spaceBefore=14,
                spaceAfter=8
            )
            body_style = ParagraphStyle(
                "BodyStyle",
                parent=styles["Normal"],
                fontSize=10,
                leading=14,
                textColor=colors.HexColor("#334155")
            )

            story = []

            # Header
            story.append(Paragraph("AUTODATA-HUB: Dataset Quality & Performance Audit", title_style))
            story.append(Paragraph(f"Job ID: {job.id} | Engine: {dataset.file_type.capitalize()} | Generated: {job.completed_at or 'N/A'}", subtitle_style))
            story.append(Spacer(1, 10))

            # Executive Scorecard Table
            pre_s = qr.pre_score if qr else 0.0
            post_s = qr.post_score if qr else 0.0
            delta_s = round(post_s - pre_s, 2)
            delta_prefix = "+" if delta_s >= 0 else ""

            score_data = [
                ["Metric", "Pre-Processing", "Post-Processing", "Net Gain"],
                ["Quality Score (0-100)", f"{pre_s:.1f}", f"{post_s:.1f}", f"{delta_prefix}{delta_s:.1f}"],
                ["Dataset File", dataset.original_filename, dataset.file_type.upper(), f"{(dataset.file_size_bytes / 1024):.1f} KB"],
                ["Execution Paradigm", job.execution_mode.upper(), "Status", job.status],
            ]

            t_score = Table(score_data, colWidths=[150, 120, 120, 120])
            t_score.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
            ]))
            story.append(t_score)
            story.append(Spacer(1, 15))

            # Quality Dimensions Breakdown
            if qr and qr.metrics_json:
                metrics = json.loads(qr.metrics_json)
                pre_q = metrics.get("pre_quality", {})
                post_q = metrics.get("post_quality", {})
                pre_c = pre_q.get("contributions", {})
                post_c = post_q.get("contributions", {})
                formula_txt = pre_q.get("formula_display", "Q = 35%·S_comp + 25%·S_val + 20%·S_uniq + 20%·S_snr")

                story.append(Paragraph("Quality Dimensions Breakdown & Mathematical Formula", heading2_style))
                story.append(Paragraph(f"<b>Formula:</b> {formula_txt}", body_style))
                story.append(Spacer(1, 6))

                dim_data = [
                    ["Dimension", "Weight", "Baseline (%)", "Base Contrib", "Optimized (%)", "Opt Contrib"],
                    [
                        "Completeness",
                        "35%",
                        f"{pre_q.get('completeness', 0):.1f}%",
                        f"{pre_c.get('completeness', 0):.1f} pts",
                        f"{post_q.get('completeness', 0):.1f}%",
                        f"{post_c.get('completeness', 0):.1f} pts"
                    ],
                    [
                        "Validity",
                        "25%",
                        f"{pre_q.get('validity', 0):.1f}%",
                        f"{pre_c.get('validity', 0):.1f} pts",
                        f"{post_q.get('validity', 0):.1f}%",
                        f"{post_c.get('validity', 0):.1f} pts"
                    ],
                    [
                        "Uniqueness",
                        "20%",
                        f"{pre_q.get('uniqueness', 0):.1f}%",
                        f"{pre_c.get('uniqueness', 0):.1f} pts",
                        f"{post_q.get('uniqueness', 0):.1f}%",
                        f"{post_c.get('uniqueness', 0):.1f} pts"
                    ],
                    [
                        "Signal-to-Noise (SNR)",
                        "20%",
                        f"{pre_q.get('snr', 0):.1f}%",
                        f"{pre_c.get('snr', 0):.1f} pts",
                        f"{post_q.get('snr', 0):.1f}%",
                        f"{post_c.get('snr', 0):.1f} pts"
                    ],
                    [
                        "Composite Total",
                        "100%",
                        "-",
                        f"{pre_s:.1f} pts",
                        "-",
                        f"{post_s:.1f} pts"
                    ],
                ]
                t_dim = Table(dim_data, colWidths=[130, 60, 80, 80, 80, 80])
                t_dim.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("BACKGROUND", (0, 1), (-1, -2), colors.HexColor("#ffffff")),
                    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f1f5f9")),
                    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ]))
                story.append(t_dim)
                story.append(Spacer(1, 15))

                # Column-level Imputation Strategies (if present)
                imp_strats = metrics.get("imputation_strategies", {})
                null_cols = [k for k, v in imp_strats.items() if v.get("null_count", 0) > 0]
                if null_cols:
                    story.append(Paragraph("Column-Level Imputation Strategy Audit", heading2_style))
                    imp_rows = [["Column", "Data Type", "Nulls", "Exact Imputation Strategy Applied"]]
                    for col_n in null_cols[:12]:
                        c_info = imp_strats[col_n]
                        imp_rows.append([
                            col_n,
                            c_info.get("data_type", ""),
                            f"{c_info.get('null_count', 0)} ({c_info.get('null_percent', 0):.1f}%)",
                            Paragraph(c_info.get("description", c_info.get("strategy", "")), body_style),
                        ])
                    t_imp = Table(imp_rows, colWidths=[110, 80, 80, 240])
                    t_imp.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]))
                    story.append(t_imp)
                    story.append(Spacer(1, 15))

            # Runtime Performance Benchmarks
            if benchmarks:
                story.append(Paragraph("Performance Benchmarks & Execution Telemetry", heading2_style))
                bench_data = [["Pipeline Stage", "Latency (ms)", "Memory Delta (MB)", "CPU (%)"]]
                total_ms = 0.0
                for b in benchmarks:
                    total_ms += b.latency_ms
                    bench_data.append([
                        b.stage_name,
                        f"{b.latency_ms:.1f} ms",
                        f"{b.memory_delta_mb:+.1f} MB",
                        f"{b.cpu_percent:.1f}%"
                    ])
                bench_data.append(["Total Execution Latency", f"{total_ms:.1f} ms", "-", "-"])

                t_bench = Table(bench_data, colWidths=[180, 110, 110, 110])
                t_bench.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f1f5f9")),
                ]))
                story.append(t_bench)
                story.append(Spacer(1, 15))

            # Recommendations & Interventions
            if qr and qr.recommendations_json:
                recs = json.loads(qr.recommendations_json)
                if recs:
                    story.append(Paragraph("Applied Recommendations & Explainable Audit", heading2_style))
                    rec_rows = [["Target", "Diagnosis", "Applied Intervention"]]
                    for r in recs[:8]: # top 8
                        rec_rows.append([
                            r.get("target", "Dataset"),
                            Paragraph(r.get("diagnosis", ""), body_style),
                            Paragraph(r.get("recommendation", ""), body_style),
                        ])
                    t_rec = Table(rec_rows, colWidths=[110, 200, 200])
                    t_rec.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#475569")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ]))
                    story.append(t_rec)

            doc.build(story)
            return buffer.getvalue()
        finally:
            db.close()
