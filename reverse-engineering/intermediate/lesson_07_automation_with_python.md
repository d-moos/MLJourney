# Lesson 7: Automation with Python – Scripting Analysis Tasks

## Overview

**Python automation** allows you to automate repetitive reverse engineering tasks. You can write scripts to:
- Parse PE files
- Analyze binaries
- Extract information
- Perform batch analysis

## What You'll Learn

By the end of this lesson, you will understand:

- **How to use Python for reverse engineering**
- **How to parse PE files** with pefile
- **How to analyze binaries** with Binary Ninja's Python API
- **How to automate analysis tasks**
- **How to write helper scripts**

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1-6 of this course
- Be comfortable with Python
- Understand PE file structure

## Python Libraries for Reverse Engineering

### pefile

`pefile` is a Python library for parsing PE files.

```python
import pefile

pe = pefile.PE('binary.exe')
print(pe.DOS_HEADER)
print(pe.NT_HEADERS)
print(pe.SECTIONS)
```

### Binary Ninja Python API

Binary Ninja has a Python API for analyzing binaries.

```python
import binaryninja

bv = binaryninja.BinaryViewType.get_view_of_file('binary.exe')
for func in bv.functions:
    print(func.name)
```

### capstone

`capstone` is a disassembly engine.

```python
from capstone import *

md = Cs(CS_ARCH_X86, CS_MODE_64)
for insn in md.disasm(code, 0x1000):
    print(f"{insn.address:x}: {insn.mnemonic} {insn.op_str}")
```

## Exercises

### Exercise 1: Parse a PE File with pefile

**Objective**: Learn to parse PE files with Python.

**Steps**:
1. Write a Python script that:
   - Opens a PE file
   - Prints the DOS header
   - Prints the NT headers
   - Prints all sections
2. Run the script on a binary
3. Document your findings

**Verification**: Your script should successfully parse the PE file.

### Exercise 2: Analyze a Binary with Binary Ninja API

**Objective**: Learn to use Binary Ninja's Python API.

**Steps**:
1. Write a Python script that:
   - Opens a binary with Binary Ninja
   - Lists all functions
   - For each function, prints the name and address
2. Run the script on a binary
3. Document your findings

**Verification**: Your script should list all functions.

### Exercise 3: Automate Analysis Tasks

**Objective**: Learn to automate analysis.

**Steps**:
1. Write a Python script that:
   - Analyzes multiple binaries
   - Extracts information from each
   - Generates a report
2. Run the script on multiple binaries
3. Document your findings

**Verification**: Your script should generate a report.

## Solutions

### Solution 1: Parse a PE File with pefile

A simple PE parser:

```python
import pefile

pe = pefile.PE('binary.exe')
print(f"Image Base: {hex(pe.OPTIONAL_HEADER.ImageBase)}")
print(f"Entry Point: {hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint)}")
for section in pe.SECTIONS:
    print(f"{section.Name}: {hex(section.VirtualAddress)} - {hex(section.VirtualAddress + section.VirtualSize)}")
```

### Solution 2: Analyze a Binary with Binary Ninja API

A simple Binary Ninja analyzer:

```python
import binaryninja

bv = binaryninja.BinaryViewType.get_view_of_file('binary.exe')
for func in bv.functions:
    print(f"{func.name}: {hex(func.start)}")
```

### Solution 3: Automate Analysis Tasks

A simple automation script:

```python
import os
import pefile

for filename in os.listdir('.'):
    if filename.endswith('.exe'):
        pe = pefile.PE(filename)
        print(f"{filename}: {hex(pe.OPTIONAL_HEADER.ImageBase)}")
```

## Summary

You now understand Python automation for reverse engineering. You can:

- Parse PE files with pefile
- Analyze binaries with Binary Ninja API
- Automate analysis tasks
- Write helper scripts
- Perform batch analysis

In the next lesson, you'll complete a capstone project.
