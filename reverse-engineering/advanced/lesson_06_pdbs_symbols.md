# Lesson 6: PDBs and Symbols – Using Debug Information

## Overview

PDB files (Program Database) contain debug information that makes reverse engineering much easier. They include:
- Function names and addresses
- Variable names and types
- Source file information
- Line number information

Understanding PDBs helps you:
- Analyze binaries with debug information
- Use symbol servers to download PDBs
- Understand how debug information is stored
- Recognize when PDBs are available

## What You'll Learn

By the end of this lesson, you will understand:

- PDB file format
- How to load PDBs in analysis tools
- How to use symbol servers
- How to extract information from PDBs
- How to use PDB information for analysis

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1-5 of this course
- Understand PE file structure
- Be comfortable with Binary Ninja and x64dbg

## PDB File Format

PDB files are structured databases that contain:
- Streams: Different types of information
- Stream 0: Old directory (deprecated)
- Stream 1: PDB information
- Stream 2: Type information
- Stream 3: Debug information
- Stream 4+: Module information

## Loading PDBs in Analysis Tools

### Binary Ninja

Binary Ninja can automatically load PDBs:
1. Place the PDB file in the same directory as the binary
2. Open the binary in Binary Ninja
3. Binary Ninja automatically loads the PDB
4. Function names and types are displayed

### x64dbg

x64dbg can load PDBs:
1. File → Load PDB
2. Select the PDB file
3. x64dbg loads the debug information
4. Function names are displayed in the disassembly

### Ghidra

Ghidra can load PDBs:
1. File → Load PDB
2. Select the PDB file
3. Ghidra loads the debug information

## Symbol Servers

Symbol servers are repositories of PDB files. Microsoft maintains a public symbol server.

### Using Microsoft Symbol Server

https://msdl.microsoft.com/download/symbols

### Downloading PDBs

1. Get the binary's GUID and age (from the PE header)
2. Construct the URL: https://msdl.microsoft.com/download/symbols/binary.pdb/GUID/binary.pdb
3. Download the PDB

### Tools for Downloading PDBs

- SymChk: Microsoft's symbol checker
- PDBDownloader: Third-party tool
- Binary Ninja: Can automatically download PDBs

## Extracting Information from PDBs

### Using pdbparse

pdbparse is a Python library for parsing PDB files

### Using Ghidra's PDB Analyzer

Ghidra has a built-in PDB analyzer that extracts:
- Function names
- Variable names
- Type information
- Source file information

## Exercises

### Exercise 1: Load a PDB in Binary Ninja

Objective: Learn to load PDBs.

Steps:
1. Find a binary with a corresponding PDB file
2. Place the PDB in the same directory as the binary
3. Open the binary in Binary Ninja
4. Verify that function names are displayed
5. Document your findings

Verification: Function names should be displayed.

### Exercise 2: Download a PDB from Symbol Server

Objective: Learn to download PDBs.

Steps:
1. Find a Windows system binary (e.g., kernel32.dll)
2. Get the binary's GUID and age
3. Download the PDB from Microsoft's symbol server
4. Load the PDB in Binary Ninja
5. Analyze the binary with debug information

Verification: You should be able to download and load the PDB.

### Exercise 3: Extract Information from a PDB

Objective: Learn to extract PDB information.

Steps:
1. Use pdbparse to parse a PDB file
2. Extract function names and addresses
3. Extract variable names and types
4. Create a report of the extracted information

Verification: You should be able to extract PDB information.

## Solutions

### Solution 1: Load a PDB in Binary Ninja

To load a PDB:
1. Place the PDB in the same directory as the binary
2. Open the binary in Binary Ninja
3. Binary Ninja automatically loads the PDB
4. Function names are displayed

### Solution 2: Download a PDB from Symbol Server

To download a PDB:
1. Get the binary's GUID and age
2. Construct the URL
3. Download the PDB
4. Load in Binary Ninja

### Solution 3: Extract Information from a PDB

To extract PDB information:
1. Use pdbparse to parse the PDB
2. Iterate through streams
3. Extract function names, addresses, and types
4. Create a report

## Summary

You now understand PDBs and symbols. You can:

- Load PDBs in analysis tools
- Download PDBs from symbol servers
- Extract information from PDBs
- Use debug information for analysis
- Understand PDB file format

In the next lesson, you'll learn about custom tools.
