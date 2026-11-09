#!/usr/bin/env python3
"""Authorized eight-run QM9 completion batch, with a 56-hour exception."""
from batch2_cluster import main

TASKS = [('B1',1), ('B1',2), ('A0',2), ('A1',1), ('A1',2),
         ('M0',2), ('M1',1), ('M1',2)]

if __name__ == '__main__':
    main(tasks=TASKS, batch_name='batch3', budget_hours=56)
