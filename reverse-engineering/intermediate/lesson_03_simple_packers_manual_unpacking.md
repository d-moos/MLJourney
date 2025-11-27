# Lesson 3: Simple Packers and Manual Unpacking – Reversing Packed Binaries

## Overview

A **packer** is a tool that compresses and encrypts a binary to make it harder to reverse engineer. When a packed binary runs, it unpacks itself in memory and executes the original code.

Understanding packers is important because:
- Many malware samples are packed
- Game protections use packers
- Understanding unpacking helps you analyze protected binaries
- Manual unpacking is a fundamental skill

## What You'll Learn

By the end of this lesson, you will understand:

- **How packers work** (stub + encrypted payload)
- **How to identify packed binaries**
- **How to find the Original Entry Point (OEP)**
- **How to dump unpacked code from memory**
- **How to rebuild the IAT** (Import Address Table)
- **How to use tools like Scylla** for IAT reconstruction

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1-2 of this course
- Understand PE file structure
- Be comfortable with x64dbg

## How Packers Work

A packer works in three stages:

1. **Stub**: The packer's code that unpacks the original binary
2. **Encrypted Payload**: The original binary, encrypted
3. **Unpacking**: The stub decrypts the payload and executes it

### Packer Structure

```
Packed Binary:
  [Packer Stub]
  [Encrypted Original Binary]
  [Packer Data]
```

When the packed binary runs:
1. The packer stub executes
2. The stub decrypts the original binary in memory
3. The stub jumps to the Original Entry Point (OEP)
4. The original binary executes

## Identifying Packed Binaries

### Signs of Packing

1. **Few imports**: Packed binaries often have very few imported functions
2. **Suspicious entry point**: The entry point might be in a section called `.packed` or `.stub`
3. **Entropy**: Packed sections have high entropy (look random)
4. **Small code section**: The `.text` section is very small
5. **Large data section**: There's a large section with encrypted data

### Using Binary Ninja to Detect Packing

In Binary Ninja:
1. Look at the sections
2. Check the entropy of each section
3. Look for sections with high entropy (likely encrypted)
4. Check the imports (few imports = likely packed)

## Finding the Original Entry Point (OEP)

The **Original Entry Point (OEP)** is where the original binary starts executing after unpacking.

### Method 1: Breakpoint on Entry Point

1. Open the packed binary in x64dbg
2. Set a breakpoint at the entry point
3. Run the program
4. Step through the packer code
5. Look for a `jmp` or `call` to a new address (likely the OEP)

### Method 2: Look for Suspicious Jumps

1. In Binary Ninja, look at the entry point
2. Look for a large `jmp` or `call` instruction
3. This might jump to the OEP

### Method 3: Use Entropy Analysis

1. The packer stub has low entropy (it's code)
2. The encrypted payload has high entropy
3. After unpacking, the original code has low entropy
4. Look for where entropy changes

## Dumping Unpacked Code

Once you find the OEP, you can dump the unpacked code from memory:

1. Set a breakpoint at the OEP
2. Run the program until the breakpoint
3. In x64dbg, right-click on the code
4. Select "Dump to file"
5. Save the unpacked code

## Rebuilding the IAT

When you dump unpacked code, the IAT might be incomplete or incorrect. You need to rebuild it.

### Using Scylla

Scylla is a tool that automatically rebuilds the IAT:

1. In x64dbg, when the program is at the OEP
2. Open Scylla (it's a plugin for x64dbg)
3. Click "IAT Autosearch"
4. Scylla finds the IAT
5. Click "Get Imports"
6. Scylla rebuilds the IAT
7. Save the fixed binary

## Exercises

### Exercise 1: Identify a Packed Binary

**Objective**: Learn to identify packed binaries.

**Steps**:
1. Download a simple packed binary (or create one using UPX)
2. Open it in Binary Ninja
3. Analyze the sections
4. Check the entropy
5. Look at the imports
6. Document signs of packing

**Verification**: You should be able to identify that the binary is packed.

### Exercise 2: Find the OEP

**Objective**: Learn to find the Original Entry Point.

**Steps**:
1. Open a packed binary in x64dbg
2. Set a breakpoint at the entry point
3. Run the program
4. Step through the packer code
5. Look for a `jmp` or `call` to a new address
6. Document the OEP address

**Verification**: You should find the OEP address.

### Exercise 3: Dump and Rebuild

**Objective**: Learn to dump and rebuild a packed binary.

**Steps**:
1. Open a packed binary in x64dbg
2. Find the OEP
3. Set a breakpoint at the OEP
4. Run the program
5. Dump the unpacked code to a file
6. Use Scylla to rebuild the IAT
7. Verify the unpacked binary works

**Verification**: The unpacked binary should run correctly.

## Solutions

### Solution 1: Identify a Packed Binary

When you analyze a packed binary, you should see:
- Few imports (maybe just LoadLibraryA, GetProcAddress)
- High entropy in the data section
- Small code section
- Large data section

### Solution 2: Find the OEP

When you step through the packer code, you should see:
- Decryption loops
- Memory allocation
- A final `jmp` or `call` to the OEP

### Solution 3: Dump and Rebuild

After dumping and rebuilding:
1. The unpacked binary should have the correct imports
2. The unpacked binary should run correctly
3. You can analyze the unpacked binary in Binary Ninja

## Summary

You now understand packers and unpacking. You can:

- Identify packed binaries
- Find the Original Entry Point
- Dump unpacked code from memory
- Rebuild the IAT using Scylla
- Analyze unpacked binaries

In the next lesson, you'll learn about control flow obfuscation.
