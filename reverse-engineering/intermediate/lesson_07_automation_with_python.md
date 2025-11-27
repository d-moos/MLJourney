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

Here's a comprehensive PE parser that extracts and displays key information from a PE file:

```python
import pefile
import sys

def analyze_pe(filename):
    """Analyze a PE file and print detailed information."""
    try:
        pe = pefile.PE(filename)

        # Print basic information
        print(f"=== PE File Analysis: {filename} ===\n")

        # DOS Header
        print("DOS Header:")
        print(f"  Magic: {hex(pe.DOS_HEADER.e_magic)} (should be 0x5A4D 'MZ')")
        print(f"  PE Offset: {hex(pe.DOS_HEADER.e_lfanew)}\n")

        # COFF Header
        print("COFF Header:")
        machine_types = {0x14c: 'x86', 0x8664: 'x64'}
        machine = machine_types.get(pe.FILE_HEADER.Machine, 'Unknown')
        print(f"  Machine: {hex(pe.FILE_HEADER.Machine)} ({machine})")
        print(f"  Number of Sections: {pe.FILE_HEADER.NumberOfSections}")
        print(f"  Timestamp: {pe.FILE_HEADER.TimeDateStamp}\n")

        # Optional Header
        print("Optional Header:")
        print(f"  Image Base: {hex(pe.OPTIONAL_HEADER.ImageBase)}")
        print(f"  Entry Point: {hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint)}")
        print(f"  Subsystem: {pe.OPTIONAL_HEADER.Subsystem}")
        print(f"  Size of Image: {hex(pe.OPTIONAL_HEADER.SizeOfImage)}\n")

        # Sections
        print("Sections:")
        for section in pe.SECTIONS:
            name = section.Name.decode('utf-8').rstrip('\x00')
            print(f"  {name}:")
            print(f"    Virtual Address: {hex(section.VirtualAddress)}")
            print(f"    Virtual Size: {hex(section.Misc_VirtualSize)}")
            print(f"    Raw Size: {hex(section.SizeOfRawData)}")
            print(f"    Characteristics: {hex(section.Characteristics)}")

        # Imports
        print("\nImported DLLs:")
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                print(f"  {entry.dll.decode('utf-8')}")
                for imp in entry.imports[:5]:  # Show first 5 imports
                    if imp.name:
                        print(f"    - {imp.name.decode('utf-8')}")
                if len(entry.imports) > 5:
                    print(f"    ... and {len(entry.imports) - 5} more")

        pe.close()

    except Exception as e:
        print(f"Error analyzing {filename}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pe_parser.py <binary.exe>")
        sys.exit(1)

    analyze_pe(sys.argv[1])
```

This script provides a complete overview of the PE file structure, including the DOS header, COFF header, optional header, sections, and imports. It's much more informative than just printing the image base and entry point. You can extend it further by adding export analysis, resource analysis, or relocation analysis.

### Solution 2: Analyze a Binary with Binary Ninja API

Here's a more comprehensive Binary Ninja analysis script that extracts detailed function information:

```python
import binaryninja
import sys

def analyze_binary(filename):
    """Analyze a binary with Binary Ninja and extract function statistics."""
    print(f"=== Binary Ninja Analysis: {filename} ===\n")

    # Open the binary
    bv = binaryninja.BinaryViewType.get_view_of_file(filename)
    if not bv:
        print(f"Failed to open {filename}")
        return

    # Wait for analysis to complete
    bv.update_analysis_and_wait()

    # Print basic information
    print(f"Architecture: {bv.arch.name}")
    print(f"Platform: {bv.platform.name}")
    print(f"Entry Point: {hex(bv.entry_point)}\n")

    # Analyze functions
    print(f"Total Functions: {len(bv.functions)}\n")

    # Collect statistics
    function_stats = []
    for func in bv.functions:
        # Calculate function size
        size = func.highest_address - func.lowest_address

        # Count basic blocks
        num_blocks = len(func.basic_blocks)

        # Count instructions
        num_instructions = sum(len(block) for block in func.basic_blocks)

        # Count calls
        num_calls = len(func.call_sites)

        function_stats.append({
            'name': func.name,
            'address': func.start,
            'size': size,
            'blocks': num_blocks,
            'instructions': num_instructions,
            'calls': num_calls
        })

    # Sort by size (largest first)
    function_stats.sort(key=lambda x: x['size'], reverse=True)

    # Print top 10 largest functions
    print("Top 10 Largest Functions:")
    for i, stat in enumerate(function_stats[:10], 1):
        print(f"{i}. {stat['name']} @ {hex(stat['address'])}")
        print(f"   Size: {stat['size']} bytes, Blocks: {stat['blocks']}, "
              f"Instructions: {stat['instructions']}, Calls: {stat['calls']}")

    # Find functions that call specific APIs
    print("\nFunctions calling interesting APIs:")
    interesting_apis = ['CreateFile', 'WriteFile', 'RegSetValue', 'CreateProcess']

    for func in bv.functions:
        for call_site in func.call_sites:
            # Get the called function
            called_funcs = bv.get_functions_at(call_site.address)
            for called in called_funcs:
                if any(api in called.name for api in interesting_apis):
                    print(f"  {func.name} calls {called.name}")

    bv.file.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bn_analyzer.py <binary.exe>")
        sys.exit(1)

    analyze_binary(sys.argv[1])
```

This script goes beyond just listing functions—it calculates statistics like function size, number of basic blocks, number of instructions, and number of calls. It also identifies functions that call interesting APIs, which is useful for finding potentially malicious or security-relevant code.

### Solution 3: Automate Analysis Tasks

Here's a comprehensive automation script that analyzes multiple binaries and generates a detailed report:

```python
import os
import pefile
import json
from datetime import datetime

def analyze_binary(filename):
    """Analyze a single binary and return results."""
    try:
        pe = pefile.PE(filename)

        # Extract basic information
        machine_types = {0x14c: 'x86', 0x8664: 'x64'}
        machine = machine_types.get(pe.FILE_HEADER.Machine, 'Unknown')

        # Calculate entropy for each section (indicator of packing/encryption)
        section_info = []
        for section in pe.SECTIONS:
            name = section.Name.decode('utf-8').rstrip('\x00')
            entropy = section.get_entropy()
            section_info.append({
                'name': name,
                'virtual_size': section.Misc_VirtualSize,
                'raw_size': section.SizeOfRawData,
                'entropy': round(entropy, 2)
            })

        # Extract imports
        imported_dlls = []
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode('utf-8')
                imports = [imp.name.decode('utf-8') for imp in entry.imports if imp.name]
                imported_dlls.append({
                    'dll': dll_name,
                    'count': len(imports),
                    'functions': imports[:10]  # First 10 functions
                })

        # Check for suspicious characteristics
        suspicious = []

        # High entropy sections (possible packing)
        high_entropy_sections = [s for s in section_info if s['entropy'] > 7.0]
        if high_entropy_sections:
            suspicious.append(f"High entropy sections: {[s['name'] for s in high_entropy_sections]}")

        # Suspicious imports
        suspicious_apis = ['VirtualAlloc', 'CreateRemoteThread', 'WriteProcessMemory',
                          'RegSetValue', 'CreateProcess']
        for dll_info in imported_dlls:
            for api in suspicious_apis:
                if any(api in func for func in dll_info['functions']):
                    suspicious.append(f"Imports {api} from {dll_info['dll']}")

        pe.close()

        return {
            'filename': filename,
            'architecture': machine,
            'image_base': hex(pe.OPTIONAL_HEADER.ImageBase),
            'entry_point': hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint),
            'sections': section_info,
            'imported_dlls': imported_dlls,
            'suspicious': suspicious,
            'timestamp': pe.FILE_HEADER.TimeDateStamp
        }

    except Exception as e:
        return {
            'filename': filename,
            'error': str(e)
        }

def generate_report(results, output_file='analysis_report.json'):
    """Generate a JSON report of all analyzed binaries."""
    report = {
        'analysis_date': datetime.now().isoformat(),
        'total_binaries': len(results),
        'results': results
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {output_file}")

def main(directory='.'):
    """Analyze all PE files in a directory."""
    print(f"=== Automated Binary Analysis ===")
    print(f"Scanning directory: {directory}\n")

    results = []

    # Find all .exe and .dll files
    for filename in os.listdir(directory):
        if filename.endswith(('.exe', '.dll')):
            print(f"Analyzing {filename}...")
            result = analyze_binary(os.path.join(directory, filename))
            results.append(result)

            # Print summary
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Architecture: {result['architecture']}")
                print(f"  Sections: {len(result['sections'])}")
                print(f"  Imported DLLs: {len(result['imported_dlls'])}")
                if result['suspicious']:
                    print(f"  ⚠️  Suspicious indicators: {len(result['suspicious'])}")
                    for indicator in result['suspicious']:
                        print(f"     - {indicator}")
            print()

    # Generate report
    generate_report(results)

    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total binaries analyzed: {len(results)}")
    successful = len([r for r in results if 'error' not in r])
    print(f"Successful analyses: {successful}")
    suspicious_count = len([r for r in results if 'error' not in r and r['suspicious']])
    print(f"Binaries with suspicious indicators: {suspicious_count}")

if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    main(directory)
```

## Summary

You now understand Python automation for reverse engineering. You can:

- Parse PE files with pefile
- Analyze binaries with Binary Ninja API
- Automate analysis tasks
- Write helper scripts
- Perform batch analysis

In the next lesson, you'll complete a capstone project.
