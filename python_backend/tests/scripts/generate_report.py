#!/usr/bin/env python3
"""
Generate test report from test results
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import argparse

class TestReportGenerator:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.test_results = []
        self.summary = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'duration': 0.0
        }
        
    def parse_junit_xml(self, xml_file):
        """Parse JUnit XML test results"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Handle both testsuite and testsuites root
            if root.tag == 'testsuites':
                testsuites = root.findall('testsuite')
            else:
                testsuites = [root]
            
            for testsuite in testsuites:
                suite_name = testsuite.get('name', 'Unknown')
                
                for testcase in testsuite.findall('testcase'):
                    test_result = {
                        'suite': suite_name,
                        'name': testcase.get('name'),
                        'classname': testcase.get('classname'),
                        'time': float(testcase.get('time', 0)),
                        'status': 'passed'
                    }
                    
                    # Check for failures
                    failure = testcase.find('failure')
                    if failure is not None:
                        test_result['status'] = 'failed'
                        test_result['failure_message'] = failure.get('message', '')
                        test_result['failure_type'] = failure.get('type', '')
                        test_result['failure_text'] = failure.text
                    
                    # Check for errors
                    error = testcase.find('error')
                    if error is not None:
                        test_result['status'] = 'error'
                        test_result['error_message'] = error.get('message', '')
                        test_result['error_type'] = error.get('type', '')
                        test_result['error_text'] = error.text
                    
                    # Check for skipped
                    skipped = testcase.find('skipped')
                    if skipped is not None:
                        test_result['status'] = 'skipped'
                        test_result['skip_message'] = skipped.get('message', '')
                    
                    self.test_results.append(test_result)
                    
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
    
    def collect_results(self):
        """Collect all test results from the results directory"""
        xml_files = list(self.results_dir.glob('*.xml'))
        
        if not xml_files:
            print(f"No XML result files found in {self.results_dir}")
            return
        
        for xml_file in xml_files:
            print(f"Processing {xml_file.name}...")
            self.parse_junit_xml(xml_file)
        
        # Calculate summary
        for result in self.test_results:
            self.summary['total'] += 1
            self.summary['duration'] += result['time']
            
            if result['status'] == 'passed':
                self.summary['passed'] += 1
            elif result['status'] == 'failed':
                self.summary['failed'] += 1
            elif result['status'] == 'skipped':
                self.summary['skipped'] += 1
            elif result['status'] == 'error':
                self.summary['errors'] += 1
    
    def generate_html_report(self, output_file):
        """Generate HTML test report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Triton Python Backend Test Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #333;
            color: white;
            padding: 20px;
            border-radius: 5px;
        }}
        .summary {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .summary-item {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .skipped {{ color: #ffc107; }}
        .error {{ color: #fd7e14; }}
        table {{
            width: 100%;
            background-color: white;
            border-collapse: collapse;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #333;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .status-passed {{ background-color: #d4edda; }}
        .status-failed {{ background-color: #f8d7da; }}
        .status-skipped {{ background-color: #fff3cd; }}
        .status-error {{ background-color: #f8d7da; }}
        .platform-info {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Triton Python Backend Test Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="platform-info">
        <h2>Platform Information</h2>
        <p><strong>OS:</strong> macOS</p>
        <p><strong>Architecture:</strong> {self.get_architecture()}</p>
        <p><strong>Python Version:</strong> {sys.version.split()[0]}</p>
    </div>
    
    <div class="summary">
        <div class="summary-item">
            <h3>Total Tests</h3>
            <h2>{self.summary['total']}</h2>
        </div>
        <div class="summary-item">
            <h3 class="passed">Passed</h3>
            <h2 class="passed">{self.summary['passed']}</h2>
        </div>
        <div class="summary-item">
            <h3 class="failed">Failed</h3>
            <h2 class="failed">{self.summary['failed']}</h2>
        </div>
        <div class="summary-item">
            <h3 class="skipped">Skipped</h3>
            <h2 class="skipped">{self.summary['skipped']}</h2>
        </div>
        <div class="summary-item">
            <h3 class="error">Errors</h3>
            <h2 class="error">{self.summary['errors']}</h2>
        </div>
        <div class="summary-item">
            <h3>Duration</h3>
            <h2>{self.summary['duration']:.2f}s</h2>
        </div>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Suite</th>
            <th>Test Name</th>
            <th>Status</th>
            <th>Duration</th>
            <th>Details</th>
        </tr>
"""
        
        for result in sorted(self.test_results, key=lambda x: (x['suite'], x['name'])):
            status_class = f"status-{result['status']}"
            details = ""
            
            if result['status'] == 'failed':
                details = f"<small>{result.get('failure_message', '')}</small>"
            elif result['status'] == 'error':
                details = f"<small>{result.get('error_message', '')}</small>"
            elif result['status'] == 'skipped':
                details = f"<small>{result.get('skip_message', '')}</small>"
            
            html_content += f"""
        <tr class="{status_class}">
            <td>{result['suite']}</td>
            <td>{result['name']}</td>
            <td class="{result['status']}">{result['status'].upper()}</td>
            <td>{result['time']:.3f}s</td>
            <td>{details}</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <div style="margin-top: 40px; text-align: center; color: #666;">
        <p>Triton Python Backend Test Suite - macOS Compatibility</p>
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def generate_json_report(self, output_file):
        """Generate JSON test report"""
        report = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'platform': 'macOS',
                'architecture': self.get_architecture(),
                'python_version': sys.version.split()[0]
            },
            'summary': self.summary,
            'results': self.test_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    def generate_markdown_report(self, output_file):
        """Generate Markdown test report"""
        md_content = f"""# Triton Python Backend Test Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Platform Information
- **OS:** macOS
- **Architecture:** {self.get_architecture()}
- **Python Version:** {sys.version.split()[0]}

## Summary
- **Total Tests:** {self.summary['total']}
- **Passed:** {self.summary['passed']} ✅
- **Failed:** {self.summary['failed']} ❌
- **Skipped:** {self.summary['skipped']} ⏭️
- **Errors:** {self.summary['errors']} 🚫
- **Total Duration:** {self.summary['duration']:.2f}s

## Test Results

| Suite | Test Name | Status | Duration |
|-------|-----------|--------|----------|
"""
        
        for result in sorted(self.test_results, key=lambda x: (x['suite'], x['name'])):
            status_emoji = {
                'passed': '✅',
                'failed': '❌',
                'skipped': '⏭️',
                'error': '🚫'
            }.get(result['status'], '❓')
            
            md_content += f"| {result['suite']} | {result['name']} | {status_emoji} {result['status'].upper()} | {result['time']:.3f}s |\n"
        
        # Add failed test details
        failed_tests = [r for r in self.test_results if r['status'] in ['failed', 'error']]
        if failed_tests:
            md_content += "\n## Failed Test Details\n\n"
            for test in failed_tests:
                md_content += f"### {test['suite']} - {test['name']}\n"
                if test['status'] == 'failed':
                    md_content += f"**Failure:** {test.get('failure_message', 'Unknown')}\n"
                    if 'failure_text' in test:
                        md_content += f"```\n{test['failure_text']}\n```\n"
                else:
                    md_content += f"**Error:** {test.get('error_message', 'Unknown')}\n"
                    if 'error_text' in test:
                        md_content += f"```\n{test['error_text']}\n```\n"
                md_content += "\n"
        
        with open(output_file, 'w') as f:
            f.write(md_content)
    
    def get_architecture(self):
        """Get system architecture"""
        import platform
        machine = platform.machine()
        if machine == 'arm64':
            return 'Apple Silicon (ARM64)'
        elif machine == 'x86_64':
            return 'Intel (x86_64)'
        return machine
    
    def print_summary(self):
        """Print test summary to console"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests:  {self.summary['total']}")
        print(f"Passed:       {self.summary['passed']} ({self.summary['passed']/self.summary['total']*100:.1f}%)")
        print(f"Failed:       {self.summary['failed']}")
        print(f"Skipped:      {self.summary['skipped']}")
        print(f"Errors:       {self.summary['errors']}")
        print(f"Duration:     {self.summary['duration']:.2f}s")
        print("="*60)
        
        if self.summary['failed'] > 0 or self.summary['errors'] > 0:
            print("\nFAILED TESTS:")
            for result in self.test_results:
                if result['status'] in ['failed', 'error']:
                    print(f"  - {result['suite']}.{result['name']}")

def main():
    parser = argparse.ArgumentParser(description='Generate test report from results')
    parser.add_argument('results_dir', help='Directory containing test result XML files')
    parser.add_argument('--format', choices=['html', 'json', 'markdown', 'all'], 
                       default='all', help='Output format')
    
    args = parser.parse_args()
    
    generator = TestReportGenerator(args.results_dir)
    generator.collect_results()
    
    if not generator.test_results:
        print("No test results found!")
        return 1
    
    # Generate reports
    results_path = Path(args.results_dir)
    
    if args.format in ['html', 'all']:
        html_file = results_path / 'test_report.html'
        generator.generate_html_report(html_file)
        print(f"HTML report generated: {html_file}")
    
    if args.format in ['json', 'all']:
        json_file = results_path / 'test_report.json'
        generator.generate_json_report(json_file)
        print(f"JSON report generated: {json_file}")
    
    if args.format in ['markdown', 'all']:
        md_file = results_path / 'test_report.md'
        generator.generate_markdown_report(md_file)
        print(f"Markdown report generated: {md_file}")
    
    # Always print summary
    generator.print_summary()
    
    # Return non-zero if tests failed
    return 1 if generator.summary['failed'] > 0 or generator.summary['errors'] > 0 else 0

if __name__ == '__main__':
    sys.exit(main())