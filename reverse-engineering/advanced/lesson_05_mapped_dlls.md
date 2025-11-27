# Lesson 5: Manually Mapped DLLs – Analyzing Injected Code

## Overview

**Manually mapped DLLs** are DLLs that are loaded without using the standard Windows loader. Instead, the code manually:
1. Allocates memory
2. Copies the DLL into memory
3. Resolves imports
4. Calls the entry point

This technique is used by:
- Malware to hide DLL loading
- Game cheats to inject code
- Rootkits to hide from detection

## What You'll Learn

By the end of this lesson, you will understand:

- **How manually mapped DLLs work**
- **How to identify manually mapped DLLs**
- **How to analyze manually mapped DLLs**
- **How to reconstruct the DLL**
- **How to extract the DLL from memory**

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1-4 of this course
- Understand DLL structure
- Understand memory layout

## How Manually Mapped DLLs Work

### Steps

1. **Allocate memory**: Allocate memory for the DLL
2. **Copy DLL**: Copy the DLL into the allocated memory
3. **Resolve imports**: Manually resolve imported functions
4. **Fix relocations**: Apply relocations for the new base address
5. **Call entry point**: Call the DLL's entry point

### Example Code

```c
// Allocate memory
PVOID dll_base = VirtualAlloc(NULL, dll_size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);

// Copy DLL
memcpy(dll_base, dll_data, dll_size);

// Resolve imports
// ... (manually resolve each imported function)

// Fix relocations
// ... (apply relocations)

// Call entry point
typedef BOOL (*DllMainFunc)(HINSTANCE, DWORD, LPVOID);
DllMainFunc dll_main = (DllMainFunc)((PBYTE)dll_base + entry_point_offset);
dll_main((HINSTANCE)dll_base, DLL_PROCESS_ATTACH, NULL);
```

## Identifying Manually Mapped DLLs

### Signs of Manual Mapping

1. **Unusual memory allocation**: Large allocations with EXECUTE permission
2. **Manual import resolution**: Calls to GetProcAddress or similar
3. **Relocation fixing**: Code that modifies memory based on base address
4. **No LoadLibrary**: The DLL is not loaded via LoadLibrary

### Finding Manually Mapped DLLs

1. **Monitor memory allocations**: Use tools like ProcMon
2. **Look for suspicious code**: Look for manual import resolution
3. **Analyze memory**: Look for PE headers in allocated memory
4. **Use debugger**: Set breakpoints on VirtualAlloc and trace execution

## Analyzing Manually Mapped DLLs

### Steps

1. **Find the DLL in memory**: Locate the manually mapped DLL
2. **Dump the DLL**: Extract the DLL from memory
3. **Fix the DLL**: Rebuild the PE header if necessary
4. **Analyze the DLL**: Use Binary Ninja or similar tools

### Dumping the DLL

1. **Find the base address**: Where is the DLL loaded?
2. **Find the size**: How large is the DLL?
3. **Dump the memory**: Extract the DLL from memory
4. **Save to file**: Save the DLL to a file

## Exercises

### Exercise 1: Identify Manually Mapped DLLs

**Objective**: Learn to identify manually mapped DLLs.

**Steps**:
1. Create a program that manually maps a DLL
2. Run it in a debugger
3. Identify the manually mapped DLL
4. Document your findings

**Verification**: You should be able to identify the DLL.

### Exercise 2: Dump a Manually Mapped DLL

**Objective**: Learn to dump manually mapped DLLs.

**Steps**:
1. Create a program that manually maps a DLL
2. Run it in a debugger
3. Find the DLL in memory
4. Dump the DLL to a file
5. Verify the dumped DLL is valid

**Verification**: The dumped DLL should be valid.

### Exercise 3: Analyze a Manually Mapped DLL

**Objective**: Learn to analyze manually mapped DLLs.

**Steps**:
1. Dump a manually mapped DLL
2. Open it in Binary Ninja
3. Analyze the DLL
4. Understand what it does

**Verification**: You should understand the DLL's functionality.

## Solutions

### Solution 1: Identify Manually Mapped DLLs

Manually mapped DLLs typically:
- Are allocated with VirtualAlloc
- Have PE headers in memory
- Are not in the module list
- Have manual import resolution

### Solution 2: Dump a Manually Mapped DLL

To dump a manually mapped DLL:
1. Find the base address
2. Find the size
3. Dump the memory
4. Save to a file

### Solution 3: Analyze a Manually Mapped DLL

To analyze a manually mapped DLL:
1. Dump the DLL
2. Open in Binary Ninja
3. Analyze the code
4. Understand the functionality

## Summary

You now understand manually mapped DLLs. You can:

- Identify manually mapped DLLs
- Dump manually mapped DLLs
- Analyze manually mapped DLLs
- Understand injection techniques
- Detect and analyze injected code

In the next lesson, you'll learn about PDBs and symbols.
