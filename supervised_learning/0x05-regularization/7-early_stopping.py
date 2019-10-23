#!/usr/bin/env python3
"""Fix comment"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Fix comment"""
    check = opt_cost - cost > threshold
    if not check:
        count += 1
    else:
        count = 0
    return (count >= patience, count)
