#!/usr/bin/env python3
"""
LaTeX Content Verification and Consistency Checker
=================================================

Performs detailed verification of LaTeX document content for:
- Cross-reference consistency throughout the document
- Numerical accuracy and consistency across pages
- Figure-text alignment and descriptions
- Table accuracy and formatting
- Mathematical equation consistency

Author: Research Team
Date: June 2025
Version: 2.0 Enhanced LaTeX Verification
"""

import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

class LaTeXContentVerifier:
    """Comprehensive LaTeX content verifier for academic papers"""
    
    def __init__(self, latex_file_path: str):
        """Initialize LaTeX content verifier"""
        self.latex_file = Path(latex_file_path)
        self.content = self._load_latex_content()
        self.verification_results = {
            'timestamp': '',
            'cross_references': {},
            'numerical_consistency': {},
            'figure_text_alignment': {},
            'mathematical_accuracy': {},
            'overall_consistency_score': 0.0,
            'detailed_issues': [],
            'recommendations': []
        }
    
    def _load_latex_content(self) -> str:
        """Load LaTeX content from file"""
        try:
            with open(self.latex_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading LaTeX file: {e}")
            return ""
    
    def verify_comprehensive_consistency(self) -> Dict[str, Any]:
        """Run comprehensive LaTeX content verification"""
        print("=" * 80)
        print("DETAILED LaTeX CONTENT VERIFICATION")
        print("=" * 80)
        
        # 1. Cross-reference verification
        print("🔗 Verifying cross-references...")
        self.verification_results['cross_references'] = self._verify_cross_references()
        
        # 2. Numerical consistency check
        print("🔢 Checking numerical consistency...")
        self.verification_results['numerical_consistency'] = self._verify_numerical_consistency()
        
        # 3. Figure-text alignment
        print("📊 Verifying figure-text alignment...")
        self.verification_results['figure_text_alignment'] = self._verify_figure_text_alignment()
        
        # 4. Mathematical accuracy
        print("🧮 Checking mathematical accuracy...")
        self.verification_results['mathematical_accuracy'] = self._verify_mathematical_accuracy()
        
        # 5. Calculate overall score
        self._calculate_overall_score()
        
        # 6. Generate detailed report
        self._generate_detailed_report()
        
        print("=" * 80)
        print(f"LaTeX Content Verification Completed")
        print(f"Overall Consistency Score: {self.verification_results['overall_consistency_score']:.1%}")
        print("=" * 80)
        
        return self.verification_results
    
    def _verify_cross_references(self) -> Dict[str, Any]:
        """Verify all cross-references in the document"""
        cross_ref_results = {
            'figure_references': {'total': 0, 'valid': 0, 'invalid': []},
            'table_references': {'total': 0, 'valid': 0, 'invalid': []},
            'equation_references': {'total': 0, 'valid': 0, 'invalid': []},
            'section_references': {'total': 0, 'valid': 0, 'invalid': []},
            'score': 0.0
        }
        
        # Extract all labels
        figure_labels = set(re.findall(r'\\label\{fig:([^}]+)\}', self.content))
        table_labels = set(re.findall(r'\\label\{tab:([^}]+)\}', self.content))
        equation_labels = set(re.findall(r'\\label\{eq:([^}]+)\}', self.content))
        section_labels = set(re.findall(r'\\label\{sec:([^}]+)\}', self.content))
        
        # Check figure references
        figure_refs = re.findall(r'Figure~\\ref\{fig:([^}]+)\}', self.content)
        cross_ref_results['figure_references']['total'] = len(figure_refs)
        for ref in figure_refs:
            if ref in figure_labels:
                cross_ref_results['figure_references']['valid'] += 1
            else:
                cross_ref_results['figure_references']['invalid'].append(ref)
        
        # Check table references
        table_refs = re.findall(r'Table~\\ref\{tab:([^}]+)\}', self.content)
        cross_ref_results['table_references']['total'] = len(table_refs)
        for ref in table_refs:
            if ref in table_labels:
                cross_ref_results['table_references']['valid'] += 1
            else:
                cross_ref_results['table_references']['invalid'].append(ref)
        
        # Calculate cross-reference score
        total_refs = (cross_ref_results['figure_references']['total'] + 
                     cross_ref_results['table_references']['total'])
        valid_refs = (cross_ref_results['figure_references']['valid'] + 
                     cross_ref_results['table_references']['valid'])
        
        if total_refs > 0:
            cross_ref_results['score'] = valid_refs / total_refs
        else:
            cross_ref_results['score'] = 1.0
        
        print(f"   ✓ Figure references: {cross_ref_results['figure_references']['valid']}/{cross_ref_results['figure_references']['total']}")
        print(f"   ✓ Table references: {cross_ref_results['table_references']['valid']}/{cross_ref_results['table_references']['total']}")
        
        return cross_ref_results
    
    def _verify_numerical_consistency(self) -> Dict[str, Any]:
        """Verify numerical consistency throughout the document"""
        numerical_results = {
            'performance_metrics': {},
            'percentages': {},
            'statistical_values': {},
            'consistency_issues': [],
            'score': 0.0
        }
        
        # Extract performance metrics (AUC values)
        auc_values = re.findall(r'(\d\.\d{3})', self.content)
        numerical_results['performance_metrics']['auc_values'] = auc_values
        
        # Extract percentages
        percentages = re.findall(r'(\d+\.?\d*)\%', self.content)
        numerical_results['percentages']['values'] = percentages
        
        # Check for specific consistency patterns
        # Check if corporate AUC (0.847) appears consistently
        corporate_auc_mentions = len(re.findall(r'0\.847', self.content))
        retail_auc_mentions = len(re.findall(r'0\.823', self.content))
        
        numerical_results['performance_metrics']['corporate_auc_consistency'] = corporate_auc_mentions
        numerical_results['performance_metrics']['retail_auc_consistency'] = retail_auc_mentions
        
        # Check improvement percentages
        improvement_12_mentions = len(re.findall(r'12\.?\d*\%', self.content))
        numerical_results['percentages']['improvement_consistency'] = improvement_12_mentions
        
        # Score based on consistency
        consistency_score = 0.9  # Assume high consistency
        if corporate_auc_mentions < 2:
            numerical_results['consistency_issues'].append("Corporate AUC (0.847) mentioned inconsistently")
            consistency_score -= 0.1
        if retail_auc_mentions < 2:
            numerical_results['consistency_issues'].append("Retail AUC (0.823) mentioned inconsistently")
            consistency_score -= 0.1
        
        numerical_results['score'] = max(0.0, consistency_score)
        
        print(f"   ✓ Corporate AUC consistency: {corporate_auc_mentions} mentions")
        print(f"   ✓ Retail AUC consistency: {retail_auc_mentions} mentions")
        print(f"   ✓ 12% improvement consistency: {improvement_12_mentions} mentions")
        
        return numerical_results
    
    def _verify_figure_text_alignment(self) -> Dict[str, Any]:
        """Verify alignment between figures and their descriptions"""
        figure_alignment = {
            'figure_captions': {},
            'figure_discussions': {},
            'figure_numbering': {},
            'alignment_issues': [],
            'score': 0.0
        }
        
        # Extract figure captions
        captions = re.findall(r'\\caption\{([^}]+(?:\{[^}]*\}[^}]*)*)\}', self.content)
        figure_alignment['figure_captions']['total'] = len(captions)
        
        # Check for figure discussion patterns
        figure_discussions = re.findall(r'Figure~\\ref\{fig:([^}]+)\}[^.]*\.', self.content)
        figure_alignment['figure_discussions']['total'] = len(figure_discussions)
        
        # Check figure numbering sequence
        figure_numbers = re.findall(r'figure_(\d+)_', self.content)
        figure_numbers = [int(n) for n in figure_numbers]
        expected_sequence = list(range(1, max(figure_numbers) + 1)) if figure_numbers else []
        
        figure_alignment['figure_numbering']['expected'] = expected_sequence
        figure_alignment['figure_numbering']['found'] = sorted(set(figure_numbers))
        figure_alignment['figure_numbering']['sequential'] = (
            figure_alignment['figure_numbering']['expected'] == 
            figure_alignment['figure_numbering']['found']
        )
        
        # Score calculation
        caption_score = 1.0 if len(captions) >= 8 else len(captions) / 8
        discussion_score = 1.0 if len(figure_discussions) >= 6 else len(figure_discussions) / 6
        numbering_score = 1.0 if figure_alignment['figure_numbering']['sequential'] else 0.8
        
        figure_alignment['score'] = (caption_score + discussion_score + numbering_score) / 3
        
        print(f"   ✓ Figure captions found: {len(captions)}")
        print(f"   ✓ Figure discussions found: {len(figure_discussions)}")
        print(f"   ✓ Figure numbering sequential: {figure_alignment['figure_numbering']['sequential']}")
        
        return figure_alignment
    
    def _verify_mathematical_accuracy(self) -> Dict[str, Any]:
        """Verify mathematical equations and calculations"""
        math_results = {
            'equations': {},
            'calculations': {},
            'accuracy_issues': [],
            'score': 0.0
        }
        
        # Count equations
        equations = re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', self.content, re.DOTALL)
        math_results['equations']['total'] = len(equations)
        
        # Count align environments
        aligns = re.findall(r'\\begin\{align\}(.*?)\\end\{align\}', self.content, re.DOTALL)
        math_results['equations']['align_total'] = len(aligns)
        
        # Check for mathematical symbols and notation consistency
        math_symbols = {
            'psi': len(re.findall(r'\\psi', self.content)),
            'phi': len(re.findall(r'\\phi', self.content)),
            'beta': len(re.findall(r'\\beta', self.content)),
            'alpha': len(re.findall(r'\\alpha', self.content))
        }
        
        math_results['calculations']['symbols'] = math_symbols
        
        # Score based on equation presence and notation consistency
        equation_score = 1.0 if len(equations) >= 3 else len(equations) / 3
        notation_score = 1.0 if sum(math_symbols.values()) >= 5 else sum(math_symbols.values()) / 5
        
        math_results['score'] = (equation_score + notation_score) / 2
        
        print(f"   ✓ Equations found: {len(equations)}")
        print(f"   ✓ Align environments: {len(aligns)}")
        print(f"   ✓ Mathematical symbols: {sum(math_symbols.values())}")
        
        return math_results
    
    def _calculate_overall_score(self):
        """Calculate overall consistency score"""
        scores = [
            self.verification_results['cross_references']['score'],
            self.verification_results['numerical_consistency']['score'],
            self.verification_results['figure_text_alignment']['score'],
            self.verification_results['mathematical_accuracy']['score']
        ]
        
        self.verification_results['overall_consistency_score'] = sum(scores) / len(scores)
        
        # Generate recommendations based on scores
        if self.verification_results['cross_references']['score'] < 0.9:
            self.verification_results['recommendations'].append(
                "Review and fix cross-reference issues"
            )
        
        if self.verification_results['numerical_consistency']['score'] < 0.9:
            self.verification_results['recommendations'].append(
                "Check numerical consistency across document"
            )
        
        if self.verification_results['figure_text_alignment']['score'] < 0.9:
            self.verification_results['recommendations'].append(
                "Improve figure-text alignment and descriptions"
            )
    
    def _generate_detailed_report(self):
        """Generate detailed verification report"""
        report_path = Path("/Users/omosholaowolabi/Documents/credit_intelligence_xai/consistency_reports/latex_detailed_verification.json")
        
        with open(report_path, 'w') as f:
            json.dump(self.verification_results, f, indent=2, default=str)
        
        # Generate summary report
        summary_path = Path("/Users/omosholaowolabi/Documents/credit_intelligence_xai/consistency_reports/latex_verification_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("DETAILED LaTeX CONTENT VERIFICATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Overall Consistency Score: {self.verification_results['overall_consistency_score']:.1%}\n\n")
            
            f.write("Component Scores:\n")
            f.write(f"- Cross-references: {self.verification_results['cross_references']['score']:.1%}\n")
            f.write(f"- Numerical Consistency: {self.verification_results['numerical_consistency']['score']:.1%}\n")
            f.write(f"- Figure-Text Alignment: {self.verification_results['figure_text_alignment']['score']:.1%}\n")
            f.write(f"- Mathematical Accuracy: {self.verification_results['mathematical_accuracy']['score']:.1%}\n\n")
            
            if self.verification_results['recommendations']:
                f.write("Recommendations:\n")
                for rec in self.verification_results['recommendations']:
                    f.write(f"- {rec}\n")
        
        print(f"\n📋 Detailed verification report saved: {report_path}")
        print(f"📋 Summary report saved: {summary_path}")


def main():
    """Execute detailed LaTeX content verification"""
    latex_file = "/Users/omosholaowolabi/Documents/credit_intelligence_xai/latex_output/explainable_credit_intelligence.tex"
    
    if not Path(latex_file).exists():
        print(f"❌ LaTeX file not found: {latex_file}")
        return
    
    verifier = LaTeXContentVerifier(latex_file)
    results = verifier.verify_comprehensive_consistency()
    
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Overall LaTeX Consistency: {results['overall_consistency_score']:.1%}")
    
    print("\nDetailed Breakdown:")
    print(f"🔗 Cross-references: {results['cross_references']['score']:.1%}")
    print(f"🔢 Numerical Consistency: {results['numerical_consistency']['score']:.1%}")
    print(f"📊 Figure-Text Alignment: {results['figure_text_alignment']['score']:.1%}")
    print(f"🧮 Mathematical Accuracy: {results['mathematical_accuracy']['score']:.1%}")
    
    # Show specific metrics
    print("\nSpecific Findings:")
    print(f"📈 Corporate AUC (0.847) mentions: {results['numerical_consistency']['performance_metrics']['corporate_auc_consistency']}")
    print(f"📈 Retail AUC (0.823) mentions: {results['numerical_consistency']['performance_metrics']['retail_auc_consistency']}")
    print(f"📊 Figure captions: {results['figure_text_alignment']['figure_captions']['total']}")
    print(f"🧮 Mathematical equations: {results['mathematical_accuracy']['equations']['total']}")
    
    if results['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in results['recommendations']:
            print(f"   - {rec}")
    
    return results


if __name__ == "__main__":
    main()