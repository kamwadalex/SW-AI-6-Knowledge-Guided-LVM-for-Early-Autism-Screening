# app/services/pdf_report.py
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
from typing import Dict
import io
import logging

logger = logging.getLogger(__name__)

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.title_style = ParagraphStyle(
            'Title',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=20,
            alignment=1  # Center
        )
        
        # Section header style
        self.section_style = ParagraphStyle(
            'Section',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#3498db'),
            spaceBefore=15,
            spaceAfter=10
        )
        
        # Normal text style
        self.normal_style = ParagraphStyle(
            'Normal',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            spaceAfter=8
        )
    
    def _get_risk_color(self, score: float) -> colors.Color:
        """Get color based on risk score"""
        if score <= 3.0:
            return colors.HexColor('#27ae60')  # Green
        elif score <= 5.0:
            return colors.HexColor('#f39c12')  # Orange
        elif score <= 7.0:
            return colors.HexColor('#e74c3c')  # Red
        else:
            return colors.HexColor('#c0392b')  # Dark Red
    
    def _create_header(self) -> list:
        """Create PDF header"""
        header = []
        
        # Title
        title = Paragraph("Autism Screening Report", self.title_style)
        header.append(title)
        header.append(Spacer(1, 0.2*inch))
        
        # Generation date
        date_text = f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        date_para = Paragraph(f"<para align='center'>{date_text}</para>", self.normal_style)
        header.append(date_para)
        header.append(Spacer(1, 0.3*inch))
        
        return header
    
    def _create_summary_section(self, result: Dict) -> list:
        """Create summary section with main score"""
        section = []
        
        title = Paragraph("Executive Summary", self.section_style)
        section.append(title)
        
        pred = result['prediction']
        score = pred['final_score']
        severity = pred['severity']
        confidence = pred['confidence']
        
        # Score display
        score_color = self._get_risk_color(score)
        score_text = f"<para align='center'><b>Final Score: {score:.2f}/10</b></para>"
        section.append(Paragraph(score_text, self.normal_style))
        section.append(Spacer(1, 0.1*inch))
        
        # Severity and confidence
        info_data = [
            ['Severity Level', severity],
            ['Confidence', f'{confidence*100:.1f}%'],
            ['Risk Assessment', self._get_risk_label(score)]
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        section.append(info_table)
        
        return section
    
    def _create_component_analysis(self, components: Dict) -> list:
        """Create component scores section"""
        section = []
        
        title = Paragraph("Component Analysis", self.section_style)
        section.append(title)
        
        # Create table with component scores
        table_data = [
            ['Component', 'Score', 'Percentage', 'Contribution']
        ]
        
        for comp_name, score in components.items():
            comp_display = comp_name.replace('_', ' ').title()
            percentage = f'{(score/10)*100:.1f}%'
            contribution = self._get_contribution_label(score)
            
            table_data.append([
                comp_display,
                f'{score:.2f}',
                percentage,
                contribution
            ])
        
        comp_table = Table(table_data, colWidths=[2*inch, 1*inch, 1.5*inch, 2*inch])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        section.append(comp_table)
        
        return section
    
    def _create_interpretation_section(self, interpretation: Dict) -> list:
        """Create interpretation section"""
        section = []
        
        title = Paragraph("Score Interpretation", self.section_style)
        section.append(title)
        
        # Severity description
        severity_text = f"<b>Severity: {interpretation['severity']}</b>"
        section.append(Paragraph(severity_text, self.normal_style))
        
        desc_text = interpretation.get('description', '')
        section.append(Paragraph(desc_text, self.normal_style))
        section.append(Spacer(1, 0.1*inch))
        
        # Recommendation
        rec = interpretation.get('recommendation', '')
        section.append(Paragraph(f"<b>Recommendation:</b> {rec}", self.normal_style))
        
        return section
    
    def _create_knowledge_section(self, explanation: Dict) -> list:
        """Create knowledge-guided explanation section"""
        section = []
        
        title = Paragraph("Knowledge-Guided Analysis", self.section_style)
        section.append(title)
        
        # Base explanation
        base_exp = explanation.get('base_explanation', '')
        section.append(Paragraph(base_exp, self.normal_style))
        section.append(Spacer(1, 0.2*inch))
        
        # Risk level
        risk_level = explanation.get('risk_level', 'Unknown')
        risk_text = f"<b>Overall Risk Level: {risk_level}</b>"
        section.append(Paragraph(risk_text, self.normal_style))
        section.append(Spacer(1, 0.2*inch))
        
        # Dominant models
        dominant = explanation.get('dominant_models', [])
        if dominant:
            models_text = "Models driving this assessment: " + ", ".join(dominant)
            section.append(Paragraph(models_text, self.normal_style))
            section.append(Spacer(1, 0.2*inch))
        
        # Domains
        domains = explanation.get('domains', [])
        if domains:
            section.append(Paragraph("<b>Diagnostic Domains:</b>", self.normal_style))
            
            for domain in domains:
                domain_name = domain.get('domain', 'Unknown Domain')
                domain_desc = domain.get('description', '')
                domain_rec = domain.get('severity_interpretation', '')
                
                section.append(Spacer(1, 0.1*inch))
                section.append(Paragraph(f"<b>• {domain_name}</b>", self.normal_style))
                section.append(Paragraph(f"  {domain_desc}", self.normal_style))
                if domain_rec:
                    section.append(Paragraph(f"  <i>{domain_rec}</i>", self.normal_style))
        
        return section
    
    def _create_recommendations_section(self, explanation: Dict) -> list:
        """Create clinical recommendations section"""
        section = []
        
        title = Paragraph("Clinical Recommendations", self.section_style)
        section.append(title)
        
        recommendations = explanation.get('clinical_recommendations', [])
        
        for i, rec in enumerate(recommendations, 1):
            rec_text = f"{i}. {rec}"
            section.append(Paragraph(rec_text, self.normal_style))
        
        return section
    
    def _create_metadata_section(self, metadata: Dict) -> list:
        """Create processing metadata section"""
        section = []
        
        title = Paragraph("Processing Information", self.section_style)
        section.append(title)
        
        # Create metadata table
        meta_data = [
            ['Processing Time', f"{metadata.get('total_processing_time', 0):.2f} seconds"],
            ['Timestamp', metadata.get('timestamp', 'N/A')],
            ['Knowledge Guidance', 'Enabled' if metadata.get('knowledge_guidance', False) else 'Disabled']
        ]
        
        if 'model_versions' in metadata:
            meta_data.append(['Pipeline Version', 'v1.0.0'])
        
        meta_table = Table(meta_data, colWidths=[2.5*inch, 4*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
        ]))
        section.append(meta_table)
        
        return section
    
    def _get_risk_label(self, score: float) -> str:
        """Get risk assessment label"""
        if score <= 3.0:
            return "Low Risk"
        elif score <= 5.0:
            return "Medium Risk"
        elif score <= 7.0:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _get_contribution_label(self, score: float) -> str:
        """Get contribution level"""
        if score >= 7.0:
            return "High Contribution"
        elif score >= 4.0:
            return "Moderate Contribution"
        else:
            return "Low Contribution"
    
    def generate_pdf(self, result: Dict) -> bytes:
        """Generate PDF report from result data"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter,
                                   rightMargin=0.5*inch, leftMargin=0.5*inch,
                                   topMargin=0.5*inch, bottomMargin=0.5*inch)
            
            # Build PDF content
            story = []
            
            # Header
            story.extend(self._create_header())
            
            # Summary
            story.extend(self._create_summary_section(result))
            story.append(Spacer(1, 0.3*inch))
            
            # Component Analysis
            if 'component_analysis' in result:
                story.extend(self._create_component_analysis(result['component_analysis']))
                story.append(Spacer(1, 0.3*inch))
            
            # Interpretation
            if 'interpretation' in result:
                story.extend(self._create_interpretation_section(result['interpretation']))
                story.append(Spacer(1, 0.3*inch))
            
            # Knowledge-guided explanation
            if 'knowledge_guided_explanation' in result:
                story.extend(self._create_knowledge_section(result['knowledge_guided_explanation']))
                story.append(Spacer(1, 0.3*inch))
            
            # Recommendations
            if 'knowledge_guided_explanation' in result:
                story.extend(self._create_recommendations_section(result['knowledge_guided_explanation']))
                story.append(Spacer(1, 0.3*inch))
            
            # Metadata
            if 'processing_metadata' in result:
                story.extend(self._create_metadata_section(result['processing_metadata']))
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            
            return buffer.read()
            
        except Exception as e:
            logger.error(f"PDF generation failed: {str(e)}")
            raise

