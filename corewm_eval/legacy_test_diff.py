"""Compare exact failing node IDs from two pytest JUnit reports."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def _result(path):
  root = ET.parse(path).getroot()
  cases = root.findall('.//testcase')
  failures = set()
  passed = set()
  for case in cases:
    nodeid = f"{case.attrib['classname']}::{case.attrib['name']}"
    if case.find('failure') is not None or case.find('error') is not None:
      failures.add(nodeid)
    elif case.find('skipped') is None:
      passed.add(nodeid)
  return failures, passed, len(cases)


def compare(baseline, current, output):
  baseline_fail, baseline_pass, baseline_total = _result(baseline)
  current_fail, current_pass, current_total = _result(current)
  result = {
      'base_commit': '133c3c3a6075e63a678553c40917fe2ae6e5a6df',
      'baseline_total': baseline_total,
      'baseline_failed_count': len(baseline_fail),
      'baseline_passed_count': len(baseline_pass),
      'current_total': current_total,
      'current_failed_count': len(current_fail),
      'current_passed_count': len(current_pass),
      'baseline_failing_node_ids': sorted(baseline_fail),
      'current_failing_node_ids': sorted(current_fail),
      'newly_failing_node_ids': sorted(current_fail - baseline_fail),
      'previously_failing_now_passing_node_ids': sorted(
          baseline_fail - current_fail),
  }
  result['acceptance_new_failures_empty'] = not result['newly_failing_node_ids']
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
  return result


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--baseline', required=True)
  parser.add_argument('--current', required=True)
  parser.add_argument(
      '--output', default='paper_artifacts/pretraining_audit/legacy_test_diff.json')
  args = parser.parse_args(argv)
  result = compare(args.baseline, args.current, args.output)
  print(json.dumps({
      key: value for key, value in result.items()
      if not key.endswith('_node_ids')}, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
