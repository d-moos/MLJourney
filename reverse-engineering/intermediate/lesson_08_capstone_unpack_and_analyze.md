# Lesson 8: Capstone Project – Unpack and Analyze a Real Binary

## Overview

In this capstone project, you'll bring together everything you've learned in the intermediate course. You'll:
1. Find a packed binary
2. Unpack it manually
3. Analyze the unpacked binary
4. Identify vulnerabilities or malicious behavior
5. Document your findings

## Project Requirements

### Part 1: Find a Packed Binary

Find a packed binary. Options include:
- Download a packed sample from a malware repository (like VirusShare)
- Create your own packed binary using UPX or similar
- Use a crackme from crackmes.one

### Part 2: Unpack the Binary

1. Identify that the binary is packed
2. Find the Original Entry Point (OEP)
3. Dump the unpacked code from memory
4. Rebuild the IAT using Scylla
5. Verify the unpacked binary works

### Part 3: Analyze the Unpacked Binary

1. Open the unpacked binary in Binary Ninja
2. Identify the main functions
3. Analyze the control flow
4. Identify any vulnerabilities or malicious behavior
5. Document your findings

### Part 4: Document Your Work

Create a comprehensive report that includes:

1. **Overview**: What the binary does
2. **Packing Analysis**: How you identified the packing
3. **Unpacking Process**: How you unpacked the binary
4. **Analysis**: What you found in the unpacked binary
5. **Vulnerabilities/Malware**: Any security issues or malicious behavior
6. **IoCs**: Indicators of compromise (if applicable)
7. **Lessons Learned**: What you learned from this project

## Detailed Steps

### Step 1: Find a Packed Binary

Options:
- Download from VirusShare (requires registration)
- Create with UPX: `upx -9 binary.exe -o packed.exe`
- Use a crackme from crackmes.one

### Step 2: Unpack the Binary

1. Open in Binary Ninja and identify packing
2. Open in x64dbg
3. Find the OEP by stepping through the packer code
4. Set a breakpoint at the OEP
5. Dump the unpacked code
6. Use Scylla to rebuild the IAT
7. Test the unpacked binary

### Step 3: Analyze the Unpacked Binary

1. Open in Binary Ninja
2. Identify main functions
3. Analyze control flow
4. Look for vulnerabilities
5. Look for malicious behavior

### Step 4: Document Your Work

Write a detailed report with:
- Screenshots of the analysis process
- Descriptions of what you found
- Explanations of vulnerabilities or malicious behavior
- IoCs (if applicable)
- Reflection on what you learned

## Exercises

### Exercise 1: Identify Packing

**Objective**: Confirm the binary is packed.

**Steps**:
1. Open the binary in Binary Ninja
2. Analyze the sections
3. Check the entropy
4. Look at the imports
5. Document signs of packing

**Verification**: You should be able to confirm packing.

### Exercise 2: Unpack the Binary

**Objective**: Successfully unpack the binary.

**Steps**:
1. Open in x64dbg
2. Find the OEP
3. Dump the unpacked code
4. Rebuild the IAT
5. Verify the unpacked binary works

**Verification**: The unpacked binary should run correctly.

### Exercise 3: Analyze and Document

**Objective**: Analyze the unpacked binary and document findings.

**Steps**:
1. Analyze the unpacked binary
2. Identify main functions
3. Look for vulnerabilities or malicious behavior
4. Create a comprehensive report
5. Include screenshots and explanations

**Verification**: Your report should be detailed and well-documented.

## Solutions

### Solution 1: Identify Packing

When you analyze a packed binary, you should see:
- Few imports
- High entropy in data sections
- Small code section
- Large data section

### Solution 2: Unpack the Binary

The unpacking process:
1. Identify the packer
2. Find the OEP
3. Dump from memory
4. Rebuild the IAT
5. Verify functionality

### Solution 3: Analyze and Document

Your report should include:
- Overview of what the binary does
- Packing analysis
- Unpacking process
- Analysis of unpacked code
- Any vulnerabilities or malicious behavior
- IoCs
- Lessons learned

## Summary

You've completed the intermediate course! You can now:

- Understand Windows internals
- Recognize and analyze exceptions and TLS
- Unpack simple packers
- Recognize and deobfuscate obfuscated code
- Identify vulnerability patterns
- Analyze malware
- Automate analysis with Python
- Complete a full unpacking and analysis project

Congratulations! You're ready to move on to the advanced course.
